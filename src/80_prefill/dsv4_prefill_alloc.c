/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_alloc.c - block buffers and the per-token scratch array. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

static void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n ? n : 1);
    if (!q) fprintf(stderr, "prefill: out of memory (%zu bytes)\n", n);
    return q;
}

/* Grow one block buffer to `cap` rows of `per` floats/ints. Old contents are
 * preserved; new rows are not initialised because every caller writes the rows
 * it uses before reading them. */
static int grow_f(float **pp, int cap, size_t per)
{
    const size_t bytes = (size_t)cap * per * sizeof(float);
    float *q = (float *)xrealloc(*pp, bytes);
    if (!q) return -1;
    *pp = q;
    return 0;
}

static int grow_i(int **pp, int cap, size_t per)
{
    const size_t bytes = (size_t)cap * per * sizeof(int);
    int *q = (int *)xrealloc(*pp, bytes);
    if (!q) return -1;
    *pp = q;
    return 0;
}

/* Per-token prefill state. Sized from what the block path actually reads:
 * attention's compressor/indexer working rows plus the one accumulator that
 * must survive the whole layer. Sizing this like the decode DSV4Scratch wasted
 * several times the memory and made a long prompt look far more expensive than
 * it is. */
static int tok_init(DSV4PrefillTok *s, const DSV4Cfg *c, int max_pos)
{
    memset(s, 0, sizeof *s);
    const size_t hd    = (size_t)c->head_dim;
    const size_t nidx  = (size_t)c->sliding_window + (size_t)c->index_topk;
    /* Only a ratio-4 layer runs the indexer, so n_compressed <= max_pos/4. */
    const size_t ncomp = (size_t)max_pos / 4 + 1;
    /* attn_scratch is block-wide (see DSV4Prefill): it is consumed inside the
     * serial per-token loop, so it must not be charged to every token. */
    const size_t nf    = hd + ncomp
                       + (size_t)c->topk * (size_t)c->hidden;

    float *a = (float *)calloc(nf, sizeof(float));
    if (!a) return -1;
    s->arena = a;

    float *p = a;
    s->kv = p;           p += hd;
    s->idx_scores = p;   p += ncomp;
    s->expert_acc = p;   p += (size_t)c->topk * (size_t)c->hidden;
    s->idx_cap = (int)ncomp;
    s->expert_acc_stride = c->hidden;

    s->idxs = (int *)calloc(nidx, sizeof(int));
    if (!s->idxs) { free(a); memset(s, 0, sizeof *s); return -1; }
    return 0;
}

int dsv4_prefill_reserve(DSV4Prefill *p, int n)
{
    if (!p || n <= 0) return -1;
    if (n <= p->cap) return 0;

    const DSV4Cfg *c = p->c;
    const int cap = n, hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const int mix_hc = (2 + hc) * hc;
    const size_t qr_n = (size_t)(d > c->q_lora ? d : c->q_lora);
    const size_t hdall = (size_t)c->n_heads * c->head_dim;
    const size_t inter = (size_t)c->moe_inter * c->n_shared;
    const size_t rinter = (size_t)c->moe_inter;
    const size_t og = (size_t)c->o_groups * c->o_lora;

    if (grow_f(&p->H, cap, hcd) ||
        grow_f(&p->x1, cap, (size_t)d) ||
        grow_f(&p->resid, cap, hcd) ||
        grow_f(&p->post, cap, (size_t)hc) ||
        grow_f(&p->comb, cap, (size_t)hc * hc) ||
        grow_f(&p->mixes, cap, (size_t)mix_hc) ||
        grow_f(&p->qr, cap, qr_n) ||
        grow_f(&p->q, cap, hdall) ||
        grow_f(&p->kv, cap, (size_t)c->head_dim) ||
        grow_f(&p->o, cap, hdall) ||
        grow_f(&p->ogrp, cap, og) ||
        grow_f(&p->xg, cap, hdall) ||
        grow_f(&p->yg, cap, og) ||
        grow_f(&p->gate, cap, (size_t)c->n_experts) ||
        grow_f(&p->gate_orig, cap, (size_t)c->n_experts) ||
        grow_i(&p->topk_idx, cap, (size_t)c->topk) ||
        grow_f(&p->topk_w, cap, (size_t)c->topk) ||
        grow_f(&p->sh_gate, cap, inter) ||
        grow_f(&p->sh_up, cap, inter) ||
        grow_f(&p->sh_mid, cap, inter) ||
        grow_f(&p->sh_out, cap, (size_t)d) ||
        grow_f(&p->out, cap, (size_t)d) ||
        grow_i(&p->ex_tok, cap, (size_t)c->topk) ||
        grow_i(&p->ex_k, cap, (size_t)c->topk) ||
        grow_f(&p->ex_x, cap, (size_t)d) ||
        grow_f(&p->ex_gate, cap, rinter) ||
        grow_f(&p->ex_up, cap, rinter) ||
        grow_f(&p->ex_mid, cap, rinter) ||
        grow_f(&p->ex_out, cap, (size_t)d))
        return -1;

    /* The compressor and indexer projections are position independent, so
     * they run as one block-wide GEMM each. Sizes mirror dsv4_scratch_init:
     * see DSV4LayerState's comp_kv_in / idx_q / idx_w fields. */
    if (grow_f(&p->comp_kv_b, cap, 2 * (size_t)c->head_dim) ||
        grow_f(&p->comp_sc_b, cap, 2 * (size_t)c->head_dim) ||
        grow_f(&p->idx_qb, cap,
               (size_t)c->index_n_heads * (size_t)c->index_head_dim) ||
        grow_f(&p->idx_wb, cap, (size_t)c->index_n_heads))
        return -1;

    if (p->scratch_cap < cap) {
        DSV4PrefillTok *ns = (DSV4PrefillTok *)xrealloc(
            p->scratch, (size_t)cap * sizeof *ns);
        if (!ns) return -1;
        p->scratch = ns;
        for (int t = p->scratch_cap; t < cap; t++) {
            if (tok_init(&p->scratch[t], c, p->opt.max_pos) != 0) {
                for (int j = p->scratch_cap; j < t; j++) {
                    free(p->scratch[j].arena);
                    free(p->scratch[j].idxs);
                    memset(&p->scratch[j], 0, sizeof p->scratch[j]);
                }
                return -1;
            }
        }
        p->scratch_cap = cap;
    }

    p->cap = cap;
    const size_t per_tok = 2 * hcd + 5 * (size_t)d + (size_t)hc +
        (size_t)hc * hc + (size_t)mix_hc + qr_n + 3 * hdall +
        (size_t)c->head_dim + 2 * og + 2 * (size_t)c->n_experts +
        4 * (size_t)c->topk + 3 * inter + 3 * rinter +
        4 * (size_t)c->head_dim +
        (size_t)c->index_n_heads * (size_t)c->index_head_dim +
        (size_t)c->index_n_heads;
    const size_t nidx = (size_t)c->sliding_window + (size_t)c->index_topk;
    const size_t tok_f = (size_t)c->head_dim + (size_t)p->opt.max_pos / 4 + 1
                       + nidx + (size_t)c->topk * (size_t)d;
    const size_t tok_bytes = sizeof(DSV4PrefillTok) + tok_f * sizeof(float)
                           + nidx * sizeof(int);
    fprintf(stderr, "prefill: whole-prompt block, %d token(s), %.1f MB block + "
                    "%.1f MB token state (%.1f KB/token); matmul tile <=%d rows\n",
            cap, (double)cap * (double)per_tok * sizeof(float) / 1048576.0,
            (double)cap * (double)tok_bytes / 1048576.0,
            (double)tok_bytes / 1024.0, DSV4_MAX_BATCH);
    return 0;
}

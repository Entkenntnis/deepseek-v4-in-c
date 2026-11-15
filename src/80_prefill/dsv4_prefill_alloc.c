/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_alloc.c - the block buffers and the per-token scratch array. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

static void *xcalloc(size_t n)
{
    void *p = calloc(1, n ? n : 1);
    if (!p) fprintf(stderr, "prefill: out of memory (%zu bytes)\n", n);
    return p;
}

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

    if (p->scratch_cap < cap) {
        DSV4Scratch *ns = (DSV4Scratch *)xrealloc(
            p->scratch, (size_t)cap * sizeof *ns);
        if (!ns) return -1;
        p->scratch = ns;
        for (int t = p->scratch_cap; t < cap; t++) {
            memset(&p->scratch[t], 0, sizeof p->scratch[t]);
            if (dsv4_scratch_init(&p->scratch[t], c, p->opt.max_pos) != 0) {
                for (int j = p->scratch_cap; j <= t; j++)
                    dsv4_scratch_free(&p->scratch[j]);
                return -1;
            }
        }
        p->scratch_cap = cap;
    }

    p->cap = cap;
    const size_t per_tok = 2 * hcd + 5 * (size_t)d + (size_t)hc +
        (size_t)hc * hc + (size_t)mix_hc + qr_n + 3 * hdall +
        (size_t)c->head_dim + 2 * og + 2 * (size_t)c->n_experts +
        4 * (size_t)c->topk + 3 * inter + 3 * rinter;
    fprintf(stderr, "prefill: whole-prompt block, %d token(s), %.1f MB block "
                    "buffers; matmul tile <=%d rows\n",
            cap, (double)cap * (double)per_tok * sizeof(float) / 1048576.0,
            DSV4_MAX_BATCH);
    return 0;
}

int dsv4_prefill_init(DSV4Prefill **out, const DSV4Cfg *c, DSV4Trunk *tr,
                      DSV4LayerState *lst, const DSV4ExpertSrc *src,
                      const DSV4ModelW *mw, const DSV4PrefillOpts *opt)
{
    DSV4Prefill *p = (DSV4Prefill *)xcalloc(sizeof *p);
    if (!p) return -1;
    p->c = c; p->tr = tr; p->lst = lst; p->src = src; p->mw = mw;
    p->opt = *opt;

    /* Only the block-independent buffers live here. Everything scaled by the
     * prompt length is allocated by dsv4_prefill_reserve() once the caller
     * knows how many tokens this block has. */
    p->xh = xcalloc((size_t)c->hidden * sizeof(float));
    p->ex_count = (int *)xcalloc((size_t)c->n_experts * sizeof(int));
    p->ex_order = (int *)xcalloc((size_t)c->n_experts * sizeof(int));
    p->ex_off = (int *)xcalloc(((size_t)c->n_experts + 1) * sizeof(int));
    p->ex_fill = (int *)xcalloc((size_t)c->n_experts * sizeof(int));
    p->ex_seen = (int *)xcalloc((size_t)c->n_experts * sizeof(int));
    if (!p->xh || !p->ex_count || !p->ex_order || !p->ex_off ||
        !p->ex_fill || !p->ex_seen) {
        dsv4_prefill_free(p);
        return -1;
    }

    fprintf(stderr, "prefill: whole-prompt block; kernels tile at <=%d rows\n",
            DSV4_MAX_BATCH);
    *out = p;
    return 0;
}

void dsv4_prefill_free(DSV4Prefill *p)
{
    if (!p) return;
    if (p->scratch) {
        for (int t = 0; t < p->scratch_cap; t++)
            dsv4_scratch_free(&p->scratch[t]);
        free(p->scratch);
    }
    free(p->H); free(p->x1); free(p->resid); free(p->post); free(p->comb);
    free(p->mixes); free(p->qr); free(p->q); free(p->kv); free(p->o);
    free(p->ogrp); free(p->xg); free(p->yg); free(p->gate);
    free(p->gate_orig); free(p->topk_idx); free(p->topk_w); free(p->sh_gate);
    free(p->sh_up); free(p->sh_mid); free(p->sh_out);
    free(p->out); free(p->xh);
    free(p->ex_tok); free(p->ex_k); free(p->ex_count); free(p->ex_order);
    free(p->ex_off); free(p->ex_fill); free(p->ex_seen);
    free(p->ex_x); free(p->ex_gate); free(p->ex_up); free(p->ex_mid);
    free(p->ex_out);
    free(p);
}

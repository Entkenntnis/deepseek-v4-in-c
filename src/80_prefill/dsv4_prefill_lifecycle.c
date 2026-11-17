/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_lifecycle.c - create, release and free a prefill context. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"
#include "dsv4_prefill_vk.h"

static void *xcalloc(size_t n)
{
    void *p = calloc(1, n ? n : 1);
    if (!p) fprintf(stderr, "prefill: out of memory (%zu bytes)\n", n);
    return p;
}

static void tok_free(DSV4PrefillTok *s)
{
    if (!s) return;
    free(s->arena);
    free(s->idxs);
    memset(s, 0, sizeof *s);
}

int dsv4_prefill_init(DSV4Prefill **out, const DSV4Cfg *c, DSV4Trunk *tr,
                      DSV4LayerState *lst, const DSV4ExpertSrc *src,
                      const DSV4ModelW *mw, const DSV4PrefillOpts *opt)
{
    if (!src || !src->begin || !src->end) {
        fprintf(stderr, "prefill: the expert source must provide begin/end; "
                        "the block path has no per-token fallback\n");
        return -1;
    }
    DSV4Prefill *p = (DSV4Prefill *)xcalloc(sizeof *p);
    if (!p) return -1;
    p->c = c; p->tr = tr; p->lst = lst; p->src = src; p->mw = mw;
    p->opt = *opt;
    dsv4_prefill_instr_init(p);

    /* Only the block-independent buffers live here. Everything scaled by the
     * prompt length is allocated by dsv4_prefill_reserve() once the caller
     * knows how many tokens this block has. */
    p->xh = xcalloc((size_t)c->hidden * sizeof(float));
    /* One score row per head, for the head-parallel sparse attention. */
    p->attn_scratch = xcalloc(
        (size_t)c->n_heads *
        ((size_t)c->sliding_window + (size_t)c->index_topk) * sizeof(float));
    p->ex_count = (int *)xcalloc((size_t)c->n_experts * sizeof(int));
    p->ex_order = (int *)xcalloc((size_t)c->n_experts * sizeof(int));
    p->ex_off = (int *)xcalloc(((size_t)c->n_experts + 1) * sizeof(int));
    p->ex_fill = (int *)xcalloc((size_t)c->n_experts * sizeof(int));
    p->ex_seen = (int *)xcalloc((size_t)c->n_experts * sizeof(int));
    if (!p->xh || !p->attn_scratch || !p->ex_count || !p->ex_order ||
        !p->ex_off || !p->ex_fill || !p->ex_seen) {
        dsv4_prefill_free(p);
        return -1;
    }

    fprintf(stderr, "prefill: whole-prompt block; kernels tile at <=%d rows\n",
            DSV4_MAX_BATCH);

    const char *cpu_only = getenv("DSV4_PREFILL_CPU");
    if (cpu_only && *cpu_only && strcmp(cpu_only, "0")) {
        fprintf(stderr, "prefill: DSV4_PREFILL_CPU set; using CPU kernels\n");
    } else if (dsv4_prefill_vk_init(&p->vk, c, tr, (DSV4Cache *)src->ctx)
               != 0) {
        p->vk = NULL;   /* dsv4_prefill_vk_init already logged */
    }

    *out = p;
    return 0;
}

/* Drop everything scaled by the block length. Called after each turn: prefill
 * is the only user of these buffers, and the next turn's allocation costs
 * milliseconds against a prefill that costs seconds. Keeps p->cap at 0 so a
 * long prompt cannot become a permanent high-water mark. */
void dsv4_prefill_release(DSV4Prefill *p)
{
    if (!p) return;
    if (p->scratch) {
        for (int t = 0; t < p->scratch_cap; t++)
            tok_free(&p->scratch[t]);
        free(p->scratch);
        p->scratch = NULL;
    }
    free(p->H); free(p->x1); free(p->resid); free(p->post); free(p->comb);
    free(p->mixes); free(p->qr); free(p->q); free(p->kv); free(p->o);
    free(p->ogrp); free(p->xg); free(p->yg); free(p->gate);
    free(p->gate_orig); free(p->topk_idx); free(p->topk_w); free(p->sh_gate);
    free(p->sh_up); free(p->sh_mid); free(p->sh_out);
    free(p->out);
    free(p->ex_tok); free(p->ex_k);
    free(p->ex_x); free(p->ex_gate); free(p->ex_up); free(p->ex_mid);
    free(p->ex_out);
    free(p->comp_kv_b); free(p->comp_sc_b);
    free(p->idx_qb); free(p->idx_wb);

    p->H = p->x1 = p->resid = p->post = p->comb = p->mixes = NULL;
    p->qr = p->q = p->kv = p->o = p->ogrp = p->xg = p->yg = NULL;
    p->gate = p->gate_orig = NULL; p->topk_idx = NULL; p->topk_w = NULL;
    p->sh_gate = p->sh_up = p->sh_mid = p->sh_out = NULL;
    p->out = NULL;
    p->ex_tok = p->ex_k = NULL; p->ex_x = NULL;
    p->ex_gate = p->ex_up = p->ex_mid = p->ex_out = NULL;
    p->comp_kv_b = p->comp_sc_b = NULL;
    p->idx_qb = p->idx_wb = NULL;
    p->cap = 0;
    p->scratch_cap = 0;
    p->n = 0;
}

void dsv4_prefill_free(DSV4Prefill *p)
{
    if (!p) return;
    if (p->instr_fp) fclose(p->instr_fp);
    p->instr_fp = NULL;
    dsv4_prefill_vk_free(p->vk);
    p->vk = NULL;
    dsv4_prefill_release(p);
    free(p->xh);
    free(p->attn_scratch);
    free(p->ex_count); free(p->ex_order); free(p->ex_off);
    free(p->ex_fill); free(p->ex_seen);
    free(p);
}

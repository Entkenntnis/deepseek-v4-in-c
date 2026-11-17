/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer_state.c - per-layer KV/compressor state and per-token scratch. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_layer.h"

int dsv4_state_init(DSV4LayerState *st, const DSV4Cfg *c, int layer, int max_pos)
{
    memset(st, 0, sizeof *st);
    /* dsv4_compress_ratio returns -1 for a layer outside the map, and -1 is
     * TRUTHY: `ratio ? ... : 0` would take the compressed branch and compute a
     * negative row count. Clamp so "no compressor" has one spelling. */
    int ratio = dsv4_compress_ratio(c, layer);
    if (ratio < 0) ratio = 0;
    const int coff = (ratio == 4) ? 2 : 1;
    const int hd   = c->head_dim;

    const size_t ncomp = ratio ? (size_t)(max_pos / ratio + 1) : 0;
    const size_t kvn   = ((size_t)c->sliding_window + ncomp) * (size_t)hd;
    const size_t cn    = ratio ? (size_t)coff * ratio * coff * hd : 0;

    float *a = (float *)calloc(kvn + 2 * cn, sizeof(float));
    if (!a) return -1;
    st->arena = a;
    st->kv_cache = a;
    if (ratio) {
        st->comp_kv    = a + kvn;
        st->comp_score = a + kvn + cn;
        /* -INFINITY, not zero: an unfilled slot must contribute nothing to the
         * pooling softmax. Zeros would give it uniform weight. */
        for (size_t i = 0; i < cn; i++) st->comp_score[i] = -INFINITY;
    }
    return 0;
}

void dsv4_state_free(DSV4LayerState *st)
{
    if (!st) return;
    free(st->arena);
    memset(st, 0, sizeof *st);
}

int dsv4_scratch_init(DSV4Scratch *s, const DSV4Cfg *c, int max_pos)
{
    memset(s, 0, sizeof *s);
    const size_t hc    = (size_t)c->hc_mult;
    const size_t d     = (size_t)c->hidden;
    const size_t heads = (size_t)c->n_heads;
    const size_t hd    = (size_t)c->head_dim;
    const size_t inter = (size_t)c->moe_inter * (size_t)c->n_shared;
    const size_t nidx  = (size_t)c->sliding_window + (size_t)c->index_topk;

    const size_t nf =
          d + hc * d + hc + hc * hc + (2 + hc) * hc
        + d + heads * hd + hd + heads * hd
        + (size_t)c->o_groups * (size_t)c->o_lora
        + (size_t)c->n_experts * 2
        + inter * 3
        + (size_t)c->n_heads * nidx
        + (size_t)c->topk * d
        + d
        + 4u * (size_t)c->head_dim
        + (size_t)c->index_n_heads * (size_t)c->index_head_dim
        + (size_t)c->index_n_heads
        + (size_t)max_pos + 1;

    /* qr must hold max(q_lora, hidden): it carries the q latent and is reused
     * for the hidden-width block output. Sizing it to q_lora alone overflows
     * whenever hidden > q_lora. */
    const size_t qr_n = (d > (size_t)c->q_lora) ? d : (size_t)c->q_lora;

    float *a = (float *)calloc(nf + qr_n, sizeof(float));
    if (!a) return -1;
    s->arena = a;

    float *p = a;
    s->x1 = p;           p += d;
    s->resid = p;        p += hc * d;
    s->post = p;         p += hc;
    s->comb = p;         p += hc * hc;
    s->mixes = p;        p += (2 + hc) * hc;
    s->qr = p;           p += qr_n;
    s->q = p;            p += heads * hd;
    s->kv = p;           p += hd;
    s->o = p;            p += heads * hd;
    s->ogrp = p;         p += (size_t)c->o_groups * (size_t)c->o_lora;
    s->gate_scores = p;  p += c->n_experts;
    s->gate_orig = p;    p += c->n_experts;
    s->expert_gate = p;  p += inter;
    s->expert_up = p;    p += inter;
    s->expert_out = p;   p += inter;
    /* Head-parallel sparse attention: one score row per head. */
    s->attn_scratch = p; p += heads * nidx;
    s->expert_acc = p;   p += (size_t)c->topk * d;
    s->expert_acc_stride = c->hidden;
    s->shared_acc = p;   p += d;
    s->comp_kv_in = p;   p += 2u * (size_t)c->head_dim;
    s->comp_sc_in = p;   p += 2u * (size_t)c->head_dim;
    s->idx_q = p;        p += (size_t)c->index_n_heads * (size_t)c->index_head_dim;
    s->idx_w = p;        p += (size_t)c->index_n_heads;
    s->idx_scores = p;   p += (size_t)max_pos + 1;
    s->idx_cap = max_pos + 1;

    s->idxs = (int *)calloc(nidx, sizeof(int));
    if (!s->idxs) { free(a); s->arena = NULL; return -1; }
    return 0;
}

void dsv4_scratch_free(DSV4Scratch *s)
{
    if (!s) return;
    free(s->arena);
    free(s->idxs);
    memset(s, 0, sizeof *s);
}

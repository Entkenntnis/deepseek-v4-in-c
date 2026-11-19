/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer.c - one decoder layer, decode path. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_cache.h"
#include "dsv4_matmul.h"
#include "dsv4_layer.h"

/* One decoder layer. `h` is [hc_mult][hidden] in and out. */
void dsv4_layer_forward(float *h, const DSV4LayerW *w, const DSV4Cfg *c,
                        DSV4Scratch *s, DSV4Cache *cache,
                        DSV4LayerState *st, const float *cs, const float *sn,
                        int pos, int token_id)
{
    const int hc = c->hc_mult, d = c->hidden;
    const size_t wide = (size_t)hc * d;

    /* ---- attention half ---- */
    memcpy(s->resid, h, wide * sizeof(float));
    dsv4_hc_pre(s->x1, s->post, s->comb, s->mixes, h, &w->hc_attn,
                hc, d, c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
    dsv4_rmsnorm(s->x1, s->x1, w->attn_norm, d, c->rms_eps);

    const int n_idx = dsv4_window_idxs(s->idxs, pos, c->sliding_window);
    dsv4_layer_attention(s->qr, s->x1, w, c, s, st, cs, sn, pos, n_idx);

    dsv4_hc_post(h, s->qr, s->resid, s->post, s->comb, hc, d);

    /* ---- FFN half. The residual is taken HERE, not at the top. ---- */
    memcpy(s->resid, h, wide * sizeof(float));
    dsv4_hc_pre(s->x1, s->post, s->comb, s->mixes, h, &w->hc_ffn,
                hc, d, c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
    dsv4_rmsnorm(s->x1, s->x1, w->ffn_norm, d, c->rms_eps);

    dsv4_layer_moe(s->qr, s->x1, w, c, s, cache, w->layer, token_id);

    dsv4_hc_post(h, s->qr, s->resid, s->post, s->comb, hc, d);
}

void dsv4_layer_moe(float *out, const float *x, const DSV4LayerW *w,
                    const DSV4Cfg *c, DSV4Scratch *s, DSV4Cache *cache,
                    int layer, int token_id)
{
    dsv4_mmq(s->gate_scores, x, &w->moe.gate);
    dsv4_route(s->topk_idx, s->topk_w, s->gate_scores, s->gate_orig,
               w->moe.bias, w->moe.tid2eid, token_id,
               c->n_experts, c->topk, c->routed_scale);

    for (int i = 0; i < c->hidden; i++) out[i] = 0.0f;

    const DSV4ExpertW *ex[DSV4_MAX_TOPK];
    int done[DSV4_MAX_TOPK];
    for (int k = 0; k < c->topk; k++) { ex[k] = NULL; done[k] = 0; }

    dsv4_cache_begin(cache, layer, s->topk_idx, c->topk, ex);

    /* The residents, while the misses are still on the wire. */
    for (int k = 0; k < c->topk; k++) {
        if (!ex[k]) continue;
        dsv4_expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                            x, ex[k], s->expert_gate, s->expert_up,
                            s->expert_out, c->moe_inter, c->swiglu_limit);
        done[k] = 1;
    }

    /* The shared expert does not depend on routing, so it fills the window
     * while the misses are read. */
    dsv4_expert_forward(s->shared_acc, x, &w->moe.shared, s->expert_gate,
                        s->expert_up, s->expert_out,
                        c->moe_inter * c->n_shared, c->swiglu_limit);

    /* A miss whose read has landed can be computed NOW. The per-k accumulator
     * and the k-order sum below keep the result identical. */
    for (;;) {
        int k = -1;
        if (!dsv4_cache_wait(cache, &k)) break;
        if (k < 0 || k >= c->topk || !ex[k]) continue;
        dsv4_expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                            x, ex[k], s->expert_gate, s->expert_up,
                            s->expert_out, c->moe_inter, c->swiglu_limit);
        done[k] = 1;
    }

    dsv4_cache_end(cache, ex);

    for (int k = 0; k < c->topk; k++) {
        if (!ex[k]) {
            fprintf(stderr, "dsv4_layer: layer %d could not obtain routed "
                            "expert %d of %d; refusing to emit a token routed "
                            "through fewer experts than the model specifies\n",
                    layer, k + 1, c->topk);
            abort();
        }
        if (!done[k])
            dsv4_expert_forward(
                s->expert_acc + (size_t)k * s->expert_acc_stride,
                x, ex[k], s->expert_gate, s->expert_up, s->expert_out,
                c->moe_inter, c->swiglu_limit);
    }
    /* IN k ORDER, ALWAYS. */
    for (int k = 0; k < c->topk; k++) {
        const float *acc = s->expert_acc + (size_t)k * s->expert_acc_stride;
        const float wk = s->topk_w[k];
        for (int i = 0; i < c->hidden; i++) out[i] += wk * acc[i];
    }

    /* The shared expert is added UNWEIGHTED, after the routed sum. */
    for (int i = 0; i < c->hidden; i++) out[i] += s->shared_acc[i];
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

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill.c - batched prompt prefill. See dsv4_prefill.h.
 *
 * WHY THIS IS A SEPARATE PATH
 *   Decode is a GEMV and is arranged around one token: weights are read once
 *   per matmul and the SSD is the floor. Prefill is the opposite problem: the
 *   same weight is needed by every prompt token, so the right unit of work is
 *   a batch of tokens through a layer. Keeping the batch code here means the
 *   decode path keeps its bit-exact reduction order and its existing gates.
 *
 * WHAT IS BATCHED, AND WHAT IS NOT
 *   Batched: q/kv/output projections, the shared expert, and the union read of
 *   the layer's routed experts. The dense FP8 matrices can run on the iGPU.
 *   Per token: mHC, RoPE, the compressor's sequential state, the CSA indexer,
 *   sparse attention, routing and SwiGLU. Those are either small or carry a
 *   sequential position dependency, so batching them would add complexity
 *   without the weight-reuse payoff.
 *
 * ROUTED EXPERTS
 *   The batch asks the cache for the union of its top-k ids once per layer,
 *   then evaluates each expert for every token that routed to it. That removes
 *   the repeated 12.75 MB reads that make unbatched prefill slow. The expert
 *   matmuls themselves still call the established CPU kernels; moving those to
 *   the GPU needs the expert slots in Vulkan memory and is a later step.
 */
#define _GNU_SOURCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4_prefill.h"
#include "dsv4_prefill_vk.h"

/* Primitives defined in src/model/dsv4_layer.c and src/core/dsv4_*.c. */
void dsv4_rmsnorm(float *, const float *, const float *, int, float);
void dsv4_swiglu(float *, const float *, const float *, int, float);
void dsv4_route(int *, float *, float *, float *, const float *,
                const int64_t *, int, int, int, float);
void dsv4_mmq(float *, const float *, const DSV4QMat *);
void dsv4_rope_apply(float *, int, int, const float *, const float *, int, int, int);
void dsv4_sparse_attn(float *, const float *, const float *, const float *,
                      const int *, int, int, int, float, float *);
int  dsv4_window_idxs(int *, int, int);
int  dsv4_compress_step(float *, const float *, const float *, const float *,
                        float *, float *, int, int, int);
int  dsv4_compress_rope_pos(int, int);
void dsv4_indexer_score(float *, const float *, const float *, const float *,
                        int, int, int);
int  dsv4_topk(int *, const float *, int, int);
void dsv4_indexer_offset(int *, int, int);
void dsv4_hc_pre(float *, float *, float *, float *, const float *,
                 const DSV4HcW *, int, int, int, float, float);
void dsv4_hc_post(float *, const float *, const float *, const float *,
                  const float *, int, int);

static void pm_check(DSV4Prefill *p, const DSV4PVJob *jobs, int nj);

static double pnow(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

struct DSV4Prefill {
    const DSV4Cfg   *c;
    DSV4Trunk       *tr;
    DSV4LayerState  *lst;
    const DSV4ExpertSrc *src;
    const DSV4ModelW *mw;
    DSV4PrefillOpts  opt;
    DSV4PrefillVk   *vk;
    int              B;

    float *H;          /* [B][hc*hidden] */
    float *x1;         /* [B][hidden] */
    float *resid;      /* [B][hc*hidden] */
    float *post;       /* [B][hc] */
    float *comb;       /* [B][hc*hc] */
    float *mixes;      /* [B][(2+hc)*hc] */
    float *qr;         /* [B][q_lora or hidden] */
    float *q;          /* [B][heads*head_dim] */
    float *kv;         /* [B][head_dim] */
    float *o;          /* [B][heads*head_dim] */
    float *ogrp;       /* [B][o_groups*o_lora] */
    float *xg, *yg;    /* per-group packing */
    float *gate;       /* [B][n_experts] */
    float *gate_orig;  /* [B][n_experts] */
    int   *topk_idx;   /* [B][topk] */
    float *topk_w;     /* [B][topk] */
    float *sh_gate;    /* [B][inter] */
    float *sh_up;      /* [B][inter] */
    float *sh_mid;     /* [B][inter] */
    float *sh_out;     /* [B][hidden] */
    float *expert_acc; /* [B][topk][hidden] */
    float *out;        /* [B][hidden] */
    float *xh;         /* [hidden] embed scratch */
    DSV4Scratch *scratch; /* [B] per-token scratch for the sequential parts */

    double t_total, t_attn, t_moe, t_io, t_bind, t_hc;
    int    vk_fallbacks;
};

static void *xcalloc(size_t n)
{
    void *p = calloc(1, n ? n : 1);
    if (!p) fprintf(stderr, "prefill: out of memory (%zu bytes)\n", n);
    return p;
}

int dsv4_prefill_init(DSV4Prefill **out, const DSV4Cfg *c, DSV4Trunk *tr,
                      DSV4LayerState *lst, const DSV4ExpertSrc *src,
                      const DSV4ModelW *mw, const DSV4PrefillOpts *opt)
{
    DSV4Prefill *p = (DSV4Prefill *)xcalloc(sizeof *p);
    if (!p) return -1;
    p->c = c; p->tr = tr; p->lst = lst; p->src = src; p->mw = mw;
    p->opt = *opt;
    if (p->opt.batch < 1) p->opt.batch = 1;
    if (p->opt.batch > DSV4_PV_NTOK) p->opt.batch = DSV4_PV_NTOK;
    p->B = p->opt.batch;

    const int B = p->B, hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const int mix_hc = (2 + hc) * hc;
    const size_t qr_n = (size_t)(d > c->q_lora ? d : c->q_lora);
    const size_t hdall = (size_t)c->n_heads * c->head_dim;
    const size_t inter = (size_t)c->moe_inter * c->n_shared;
    const size_t og = (size_t)c->o_groups * c->o_lora;

    p->H = xcalloc((size_t)B * hcd * sizeof(float));
    p->x1 = xcalloc((size_t)B * d * sizeof(float));
    p->resid = xcalloc((size_t)B * hcd * sizeof(float));
    p->post = xcalloc((size_t)B * hc * sizeof(float));
    p->comb = xcalloc((size_t)B * (size_t)hc * hc * sizeof(float));
    p->mixes = xcalloc((size_t)B * mix_hc * sizeof(float));
    p->qr = xcalloc((size_t)B * qr_n * sizeof(float));
    p->q = xcalloc((size_t)B * hdall * sizeof(float));
    p->kv = xcalloc((size_t)B * c->head_dim * sizeof(float));
    p->o = xcalloc((size_t)B * hdall * sizeof(float));
    p->ogrp = xcalloc((size_t)B * og * sizeof(float));
    p->xg = xcalloc((size_t)B * c->o_groups * (hdall / c->o_groups) * sizeof(float));
    p->yg = xcalloc((size_t)B * c->o_groups * c->o_lora * sizeof(float));
    p->gate = xcalloc((size_t)B * c->n_experts * sizeof(float));
    p->gate_orig = xcalloc((size_t)B * c->n_experts * sizeof(float));
    p->topk_idx = xcalloc((size_t)B * c->topk * sizeof(int));
    p->topk_w = xcalloc((size_t)B * c->topk * sizeof(float));
    p->sh_gate = xcalloc((size_t)B * inter * sizeof(float));
    p->sh_up = xcalloc((size_t)B * inter * sizeof(float));
    p->sh_mid = xcalloc((size_t)B * inter * sizeof(float));
    p->sh_out = xcalloc((size_t)B * d * sizeof(float));
    p->expert_acc = xcalloc((size_t)B * c->topk * d * sizeof(float));
    p->out = xcalloc((size_t)B * d * sizeof(float));
    p->xh = xcalloc((size_t)d * sizeof(float));
    p->scratch = (DSV4Scratch *)xcalloc((size_t)B * sizeof(DSV4Scratch));

    if (!p->H || !p->x1 || !p->resid || !p->post || !p->comb || !p->mixes ||
        !p->qr || !p->q || !p->kv || !p->o || !p->ogrp || !p->xg || !p->yg ||
        !p->gate || !p->gate_orig || !p->topk_idx || !p->topk_w ||
        !p->sh_gate || !p->sh_up || !p->sh_mid || !p->sh_out ||
        !p->expert_acc || !p->out || !p->xh || !p->scratch) {
        dsv4_prefill_free(p);
        return -1;
    }
    for (int t = 0; t < B; t++)
        if (dsv4_scratch_init(&p->scratch[t], c, p->opt.max_pos) != 0) {
            dsv4_prefill_free(p);
            return -1;
        }

    if (p->opt.gpu) {
        p->vk = dsv4_pv_init();
        if (!p->vk) {
            fprintf(stderr, "prefill: --gpu-prefill requested but Vulkan "
                            "initialisation failed\n");
            dsv4_prefill_free(p);
            return -1;
        }
    }
    fprintf(stderr, "gpu-prefill: %s, batch %d%s\n",
            "batched prompt path", p->B,
            p->vk ? ", Vulkan FP8 dense" : ", CPU");
    *out = p;
    return 0;
}

void dsv4_prefill_free(DSV4Prefill *p)
{
    if (!p) return;
    if (p->scratch) {
        for (int t = 0; t < p->B; t++) dsv4_scratch_free(&p->scratch[t]);
        free(p->scratch);
    }
    free(p->H); free(p->x1); free(p->resid); free(p->post); free(p->comb);
    free(p->mixes); free(p->qr); free(p->q); free(p->kv); free(p->o);
    free(p->ogrp); free(p->xg); free(p->yg); free(p->gate);
    free(p->gate_orig); free(p->topk_idx); free(p->topk_w); free(p->sh_gate);
    free(p->sh_up); free(p->sh_mid); free(p->sh_out); free(p->expert_acc);
    free(p->out); free(p->xh);
    if (p->vk) dsv4_pv_report(p->vk, "prompt");
    dsv4_pv_free(p->vk);
    free(p);
}

/* The one batched-matmul seam. The Vulkan backend only handles FP8 with a
 * 128-wide block and a prepared layer; everything else takes the CPU loop. */
static int pm_n_multi(DSV4Prefill *p, int layer, const DSV4PVJob *jobs, int nj)
{
    if (p->vk && nj > 0) {
        int all = 1;
        for (int j = 0; j < nj; j++)
            if (jobs[j].m->wdt != DSV4_WFP8 || jobs[j].m->blk_c != 128 ||
                jobs[j].n < 1)
                all = 0;
        if (all && dsv4_pv_mmq_fp8_multi(p->vk, layer, jobs, nj) == 0) {
            pm_check(p, jobs, nj);
            return 0;
        }
        p->vk_fallbacks++;
    }
    for (int j = 0; j < nj; j++) {
        const DSV4QMat *m = jobs[j].m;
        for (int t = 0; t < jobs[j].n; t++)
            dsv4_mmq(jobs[j].y + (size_t)t * (size_t)m->rows,
                     jobs[j].x + (size_t)t * (size_t)m->cols, m);
    }
    return 0;
}

/* Debug/gate hook: compare the Vulkan result of one or more jobs against the
 * CPU kernel on the same activations. Set DSV4_PREFILL_CHECK=1. */
static void pm_check(DSV4Prefill *p, const DSV4PVJob *jobs, int nj)
{
    if (!p->vk || !getenv("DSV4_PREFILL_CHECK")) return;
    static int shown = 0;
    if (shown >= 8) return;
    for (int j = 0; j < nj && shown < 8; j++, shown++) {
        const DSV4QMat *m = jobs[j].m;
        const size_t n = (size_t)jobs[j].n * (size_t)m->rows;
        float *tmp = (float *)malloc(n * sizeof(float));
        if (!tmp) return;
        for (int t = 0; t < jobs[j].n; t++)
            dsv4_mmq(tmp + (size_t)t * (size_t)m->rows,
                     jobs[j].x + (size_t)t * (size_t)m->cols, m);
        double e2 = 0.0, r2 = 0.0, mx = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double d = (double)jobs[j].y[i] - (double)tmp[i];
            if (fabs(d) > mx) mx = fabs(d);
            e2 += d * d;
            r2 += (double)tmp[i] * (double)tmp[i];
        }
        fprintf(stderr, "prefill-check %s %dx%d n=%d: L2 %.2e max %.2e\n",
                m->wdt == DSV4_WFP8 ? "fp8" : "other", m->rows, m->cols,
                jobs[j].n, sqrt(e2 / (r2 > 0 ? r2 : 1.0)), mx);
        free(tmp);
    }
}

static int pm_n(DSV4Prefill *p, int layer, float *y, const float *x,
                const DSV4QMat *m, int n)
{
    DSV4PVJob job = { m, x, y, n };
    return pm_n_multi(p, layer, &job, 1);
}

static void expert_forward(float *out, const float *x, const DSV4ExpertW *e,
                           DSV4Scratch *s, int hidden, int inter, float limit)
{
    dsv4_mmq(s->expert_gate, x, &e->w1);
    dsv4_mmq(s->expert_up,   x, &e->w3);
    dsv4_swiglu(s->expert_out, s->expert_gate, s->expert_up, inter, limit);
    dsv4_mmq(out, s->expert_out, &e->w2);
}

/* Attention half for the batch. The projections run as GEMMs; the q/kv norms,
 * RoPE, compressor, indexer and sparse attention are still per token because
 * they carry a sequential position or a tiny working set. */
static int prefill_attention(DSV4Prefill *p, int L, const DSV4LayerW *w,
                             int n, int pos0, const float *cs, const float *sn)
{
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden, hd = c->head_dim;
    const int H = c->n_heads, rd = c->qk_rope, ql = c->q_lora;
    const int comp = w->has_comp;
    DSV4LayerState *st = &p->lst[L];
    float *kvc = st->kv_cache;
    const double t0 = pnow();

    if (pm_n(p, L, p->qr, p->x1, &w->attn.wq_a, n)) return -1;
    if (pm_n(p, L, p->kv, p->x1, &w->attn.wkv, n)) return -1;
    for (int t = 0; t < n; t++)
        dsv4_rmsnorm(p->qr + (size_t)t * ql, p->qr + (size_t)t * ql,
                     w->attn.q_norm, ql, c->rms_eps);
    if (pm_n(p, L, p->q, p->qr, &w->attn.wq_b, n)) return -1;

    for (int t = 0; t < n; t++) {
        const int pos = pos0 + t;
        float *qt = p->q + (size_t)t * (size_t)H * hd;
        for (int h = 0; h < H; h++) {
            float *qh = qt + (size_t)h * hd;
            double sq = 0.0;
            for (int i = 0; i < hd; i++) sq += (double)qh[i] * qh[i];
            const float inv = (float)(1.0 / sqrt(sq / (double)hd +
                                                 (double)c->rms_eps));
            for (int i = 0; i < hd; i++) qh[i] *= inv;
            dsv4_rope_apply(qh, hd, rd, cs, sn, pos, rd / 2, 0);
        }

        float *kvt = p->kv + (size_t)t * hd;
        dsv4_rmsnorm(kvt, kvt, w->attn.kv_norm, hd, c->rms_eps);
        dsv4_rope_apply(kvt, hd, rd, cs, sn, pos, rd / 2, 0);
        memcpy(kvc + (size_t)(pos % c->sliding_window) * hd, kvt,
               (size_t)hd * sizeof(float));

        if (comp) {
            const int ratio = w->compress_ratio;
            dsv4_mmq(p->scratch[t].comp_kv_in, p->x1 + (size_t)t * d,
                     &w->attn.comp.wkv);
            dsv4_mmq(p->scratch[t].comp_sc_in, p->x1 + (size_t)t * d,
                     &w->attn.comp.wgate);
            if (dsv4_compress_step(p->scratch[t].kv,
                                   p->scratch[t].comp_kv_in,
                                   p->scratch[t].comp_sc_in,
                                   w->attn.comp.ape, st->comp_kv,
                                   st->comp_score, pos, ratio, hd)) {
                dsv4_rmsnorm(p->scratch[t].kv, p->scratch[t].kv,
                             w->attn.comp.norm, hd, c->rms_eps);
                dsv4_rope_apply(p->scratch[t].kv, hd, rd, cs, sn,
                                dsv4_compress_rope_pos(pos, ratio), rd / 2, 0);
                memcpy(kvc + (size_t)(c->sliding_window + st->n_compressed) * hd,
                       p->scratch[t].kv, (size_t)hd * sizeof(float));
                st->n_compressed++;
            }
        }

        int n_idx = dsv4_window_idxs(p->scratch[t].idxs, pos,
                                     c->sliding_window);
        const int avail = st->n_compressed;
        if (avail > 0) {
            if (avail > p->scratch[t].idx_cap) {
                fprintf(stderr, "prefill: %d compressed rows but idx_scores "
                                "holds %d\n", avail, p->scratch[t].idx_cap);
                return -1;
            }
            int take = avail;
            if (w->has_idx) {
                dsv4_mmq(p->scratch[t].idx_q, p->qr + (size_t)t * ql,
                         &w->attn.idx.wq_b);
                for (int h = 0; h < c->index_n_heads; h++)
                    dsv4_rope_apply(p->scratch[t].idx_q +
                                        (size_t)h * c->index_head_dim,
                                    c->index_head_dim, rd, cs, sn, pos,
                                    rd / 2, 0);
                dsv4_mmq(p->scratch[t].idx_w, p->x1 + (size_t)t * d,
                         &w->attn.idx.weights_proj);
                dsv4_indexer_score(p->scratch[t].idx_scores,
                                   p->scratch[t].idx_q,
                                   kvc + (size_t)c->sliding_window * hd,
                                   p->scratch[t].idx_w, c->index_n_heads,
                                   c->index_head_dim, avail);
                take = avail < c->index_topk ? avail : c->index_topk;
                dsv4_topk(p->scratch[t].idxs + n_idx,
                          p->scratch[t].idx_scores, avail, take);
            } else {
                for (int i = 0; i < take; i++)
                    p->scratch[t].idxs[n_idx + i] = i;
            }
            dsv4_indexer_offset(p->scratch[t].idxs + n_idx, take,
                                c->sliding_window);
            n_idx += take;
        }

        float *ot = p->o + (size_t)t * (size_t)H * hd;
        dsv4_sparse_attn(ot, qt, kvc, w->attn.sink, p->scratch[t].idxs,
                         H, hd, n_idx, 1.0f / sqrtf((float)hd),
                         p->scratch[t].attn_scratch);
        for (int h = 0; h < H; h++)
            dsv4_rope_apply(ot + (size_t)h * hd, hd, rd, cs, sn, pos,
                            rd / 2, 1);
    }

    /* Grouped low-rank output projection, one GEMM per group. */
    const int gw = H * hd / c->o_groups;
    const int gr = c->o_lora;
    const size_t esz = (w->attn.wo_a.wdt == DSV4_WBF16) ? 2u : 1u;
    const size_t row_bytes = (w->attn.wo_a.wdt == DSV4_WFP4)
                           ? (size_t)gw / 2u : (size_t)gw * esz;
    if (c->o_groups > 16) return -1;
    DSV4QMat slices[16];
    DSV4PVJob jobs[16];
    for (int g = 0; g < c->o_groups; g++) {
        for (int t = 0; t < n; t++)
            memcpy(p->xg + ((size_t)g * p->B + t) * gw,
                   p->o + (size_t)t * (size_t)H * hd + (size_t)g * gw,
                   (size_t)gw * sizeof(float));
        DSV4QMat slice = w->attn.wo_a;
        slice.rows = gr;
        slice.cols = gw;
        slice.w = (const unsigned char *)w->attn.wo_a.w +
                  (size_t)g * gr * row_bytes;
        if (slice.s) {
            const size_t sc_row = (size_t)((gw + slice.blk_c - 1) / slice.blk_c);
            slice.s = (const unsigned char *)w->attn.wo_a.s +
                      (size_t)g * (size_t)(gr / (slice.blk_r ? slice.blk_r : 1))
                      * sc_row;
        }
        slices[g] = slice;
        jobs[g].m = &slices[g];
        jobs[g].x = p->xg + (size_t)g * p->B * gw;
        jobs[g].y = p->yg + (size_t)g * p->B * gr;
        jobs[g].n = n;
    }
    if (pm_n_multi(p, L, jobs, c->o_groups)) return -1;
    for (int g = 0; g < c->o_groups; g++)
        for (int t = 0; t < n; t++)
            memcpy(p->ogrp + (size_t)t * (size_t)c->o_groups * c->o_lora +
                       (size_t)g * gr,
                   p->yg + (size_t)g * p->B * gr + (size_t)t * gr,
                   (size_t)gr * sizeof(float));
    if (pm_n(p, L, p->out, p->ogrp, &w->attn.wo_b, n)) return -1;
    for (int t = 0; t < n; t++)
        dsv4_hc_post(p->H + (size_t)t * (size_t)hc * d,
                     p->out + (size_t)t * d,
                     p->resid + (size_t)t * (size_t)hc * d,
                     p->post + (size_t)t * hc,
                     p->comb + (size_t)t * hc * hc, hc, d);

    p->t_attn += pnow() - t0;
    return 0;
}

static int prefill_moe(DSV4Prefill *p, int L, const DSV4LayerW *w,
                       int n, int pos0, const int32_t *ids)
{
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const int inter = c->moe_inter * c->n_shared;
    const int rinter = c->moe_inter;
    const int topk = c->topk;

    if (pm_n(p, L, p->gate, p->x1, &w->moe.gate, n)) return -1;
    for (int t = 0; t < n; t++) {
        dsv4_route(p->topk_idx + (size_t)t * topk,
                   p->topk_w + (size_t)t * topk,
                   p->gate + (size_t)t * c->n_experts,
                   p->gate_orig + (size_t)t * c->n_experts,
                   w->moe.bias, w->moe.tid2eid, ids[pos0 + t],
                   c->n_experts, topk, c->routed_scale);
    }

    const double tsh = pnow();
    {
        DSV4PVJob two[2] = {
            { &w->moe.shared.w1, p->x1, p->sh_gate, n },
            { &w->moe.shared.w3, p->x1, p->sh_up, n },
        };
        if (pm_n_multi(p, L, two, 2)) return -1;
    }
    for (int t = 0; t < n; t++)
        dsv4_swiglu(p->sh_mid + (size_t)t * inter,
                    p->sh_gate + (size_t)t * inter,
                    p->sh_up + (size_t)t * inter, inter, c->swiglu_limit);
    if (pm_n(p, L, p->sh_out, p->sh_mid, &w->moe.shared.w2, n)) return -1;
    p->t_moe += pnow() - tsh;

    /* Union of the batch's routed experts. B <= 8 and topk <= 6, so this is at
     * most 48 ids and fits the cache batch without chunking. */
    int uex[DSV4_PV_NTOK * DSV4_MAX_TOPK];
    int ucnt = 0;
    for (int t = 0; t < n; t++)
        for (int k = 0; k < topk; k++) {
            const int id = p->topk_idx[(size_t)t * topk + k];
            int seen = 0;
            for (int e = 0; e < ucnt; e++) if (uex[e] == id) { seen = 1; break; }
            if (!seen) uex[ucnt++] = id;
        }

    /* The cache's get_many is a top-k API (it reserves at most DSV4_MAX_TOPK
     * slots per call), so the union goes through in chunks of topk. Each
     * chunk's pointers stay valid while that chunk is computed. */
    const double tio = pnow();
    const double tmm = pnow();
    for (int start = 0; start < ucnt; start += topk) {
        int m = ucnt - start;
        if (m > topk) m = topk;
        const DSV4ExpertW *ex[DSV4_MAX_TOPK];
        for (int e = 0; e < m; e++) ex[e] = NULL;
        if (p->src->get_many) {
            if (p->src->get_many(p->src->ctx, L, uex + start, m, ex) != 0) {
                fprintf(stderr, "prefill: layer %d expert batch failed\n", L);
                return -1;
            }
        } else {
            for (int e = 0; e < m; e++)
                ex[e] = p->src->get(p->src->ctx, L, uex[start + e]);
        }
        for (int e = 0; e < m; e++) if (!ex[e]) {
            fprintf(stderr, "prefill: layer %d expert %d could not be loaded\n",
                    L, uex[start + e]);
            return -1;
        }
        for (int e = 0; e < m; e++) {
            for (int t = 0; t < n; t++)
                for (int k = 0; k < topk; k++)
                    if (p->topk_idx[(size_t)t * topk + k] == uex[start + e])
                        expert_forward(p->expert_acc +
                                           ((size_t)t * topk + k) * (size_t)d,
                                       p->x1 + (size_t)t * d, ex[e],
                                       &p->scratch[t], d, rinter,
                                       c->swiglu_limit);
        }
    }
    p->t_io += pnow() - tio;
    p->t_moe += pnow() - tmm;

    /* The routed sum must stay in k order, then the shared term is added, just
     * as the decode path does. */
    for (int t = 0; t < n; t++) {
        float *o = p->out + (size_t)t * d;
        for (int i = 0; i < d; i++) o[i] = 0.0f;
        for (int k = 0; k < topk; k++) {
            const float wk = p->topk_w[(size_t)t * topk + k];
            const float *acc = p->expert_acc +
                               ((size_t)t * topk + k) * (size_t)d;
            for (int i = 0; i < d; i++) o[i] += wk * acc[i];
        }
        for (int i = 0; i < d; i++) o[i] += p->sh_out[(size_t)t * d + i];
    }

    for (int t = 0; t < n; t++)
        dsv4_hc_post(p->H + (size_t)t * (size_t)hc * d,
                     p->out + (size_t)t * d,
                     p->resid + (size_t)t * (size_t)hc * d,
                     p->post + (size_t)t * hc,
                     p->comb + (size_t)t * hc * hc, hc, d);
    return 0;
}

static int prefill_layer(DSV4Prefill *p, int L, const DSV4LayerW *w,
                         int n, int pos0, const int32_t *ids,
                         const float *cs, const float *sn)
{
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const int mix_hc = (2 + hc) * hc;
    const double thc = pnow();

    for (int t = 0; t < n; t++) {
        memcpy(p->resid + (size_t)t * hcd, p->H + (size_t)t * hcd,
               hcd * sizeof(float));
        dsv4_hc_pre(p->x1 + (size_t)t * d,
                    p->post + (size_t)t * hc,
                    p->comb + (size_t)t * hc * hc,
                    p->mixes + (size_t)t * mix_hc,
                    p->H + (size_t)t * hcd, &w->hc_attn, hc, d,
                    c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
        dsv4_rmsnorm(p->x1 + (size_t)t * d, p->x1 + (size_t)t * d,
                     w->attn_norm, d, c->rms_eps);
    }
    p->t_hc += pnow() - thc;

    if (prefill_attention(p, L, w, n, pos0, cs, sn)) return -1;

    const double thc2 = pnow();
    for (int t = 0; t < n; t++) {
        memcpy(p->resid + (size_t)t * hcd,
               p->H + (size_t)t * hcd, hcd * sizeof(float));
        dsv4_hc_pre(p->x1 + (size_t)t * d,
                    p->post + (size_t)t * hc,
                    p->comb + (size_t)t * hc * hc,
                    p->mixes + (size_t)t * mix_hc,
                    p->H + (size_t)t * hcd, &w->hc_ffn, hc, d,
                    c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
        dsv4_rmsnorm(p->x1 + (size_t)t * d, p->x1 + (size_t)t * d,
                     w->ffn_norm, d, c->rms_eps);
    }
    p->t_hc += pnow() - thc2;

    if (prefill_moe(p, L, w, n, pos0, ids)) return -1;
    return 0;
}

static const unsigned char *trunk_run(DSV4Trunk *tr, int L)
{
    if (L < tr->npin) return tr->pin[L];
    if (tr->slot_of && tr->slot_of[L] >= 0 && tr->ring)
        return tr->ring + (size_t)tr->slot_of[L] * (size_t)tr->slot_bytes;
    return NULL;
}

int dsv4_prefill_prompt(DSV4Prefill *p, const int32_t *ids, int n,
                        const float *cs_d, const float *sn_d,
                        const float *cs_c, const float *sn_c,
                        float *h_out)
{
    if (!p || !ids || n <= 0 || !h_out) return -1;
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const double t0 = pnow();

    int pos = 0;
    int last_m = 1;
    while (pos < n) {
        int m = n - pos;
        if (m > p->B) m = p->B;
        last_m = m;

        for (int t = 0; t < m; t++) {
            dsv4_embed_row(p->xh, p->mw->embed, p->mw->wdt, ids[pos + t], d);
            for (int j = 0; j < hc; j++)
                memcpy(p->H + (size_t)t * hcd + (size_t)j * d, p->xh,
                       (size_t)d * sizeof(float));
        }

        for (int L = 0; L < c->n_layers; L++) {
            DSV4LayerBind lb;
            const double tb = pnow();
            const double io0 = p->tr->load_seconds;
            if (dsv4_trunk_bind(p->tr, c, L, &lb) != 0) return -1;
            const double io = p->tr->load_seconds - io0;
            const double call = pnow() - tb;
            p->t_io += io;
            p->t_bind += call > io ? call - io : 0.0;

            if (p->vk) {
                const unsigned char *run = trunk_run(p->tr, L);
                if (run && L < p->tr->npin)
                    dsv4_pv_prepare_layer(p->vk, L, run,
                                          (size_t)p->tr->lay[L].nbytes);
            }

            const int comp = dsv4_compress_ratio(c, L) != 0;
            const float *cs = comp ? cs_c : cs_d;
            const float *sn = comp ? sn_c : sn_d;
            if (prefill_layer(p, L, &lb.w, m, pos, ids, cs, sn)) return -1;
        }
        pos += m;
    }

    memcpy(h_out, p->H + (size_t)(last_m - 1) * hcd, hcd * sizeof(float));
    p->t_total = pnow() - t0;
    fprintf(stderr,
            "gpu-prefill: %d prompt tokens, %d batch(es) of <=%d, %.2f s "
            "(%.3f s/token); attn %.2f moe %.2f io %.2f bind %.2f hc %.2f%s\n",
            n, (n + p->B - 1) / p->B, p->B, p->t_total,
            p->t_total / (double)n, p->t_attn, p->t_moe, p->t_io,
            p->t_bind, p->t_hc,
            p->vk_fallbacks ? " (some VK fallbacks)" : "");
    return 0;
}

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
 * WHAT IS BATCHED
 *   The dense matrices: q/kv/output projections, the shared expert and the
 *   router. Their weights are read once per chunk and used for every token in
 *   it. The routed experts stay on the per-token path -- at a 32-token prompt
 *   top-6 over 256 experts gives only ~2.9x reuse, and the expert union was
 *   measured to eat the cache hits that made the per-token path cheap.
 *
 * WHAT IS NOT BATCHED, DELIBERATELY
 *   mHC, RoPE, the compressor's sequential state, the CSA indexer, sparse
 *   attention and the routed experts. They either carry a position-ordered
 *   dependency or have no weight-reuse payoff. Only projections are GEMMs.
 */
#define _GNU_SOURCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4_prefill.h"
#include "dsv4_prefill_gemm.h"

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

/* One matrix over `n` token rows. The CPU seam the prefill module is built on. */
typedef struct {
    const DSV4QMat *m;
    const float    *x;
    float          *y;
    int             n;
} DSV4PMJob;

static double pnow(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Fine-grained accounting for one dsv4_prefill_range call.
 *
 * The public profile only knows "attention", "mHC" and a MoE bucket that
 * contains the router, the shared expert and the routed experts at once. That
 * is not enough to tell a kernel problem from a scheduling problem, so the
 * prefill path carries its own timers. `f_*` are GFLOP (2 x MAC) for the kernel
 * family, which is what makes the seconds comparable with bench/matmul_bw:
 *
 *     effective GF/s = f_<family> / seconds in <family>
 *
 * The per-token loops are timed per call site rather than wrapped in one broad
 * region, because the point is to separate them. clock_gettime is ~20 ns and
 * the smallest loop here is microseconds, so the measurement does not perturb
 * what it measures. Timers nest: attn and moe are the broad public buckets,
 * and the sub-fields are the parts inside them. Do not add the columns up and
 * call it the total. */
typedef struct {
    double embed, hc_pre, hc_post, hc_post_moe, norm;
    double proj_qkv, q_norm, q_head, kv_norm;
    double comp_proj, comp_step, win_idx;
    double idx_proj, idx_rope, idx_score;
    double attn_core, attn_deroep;
    double oproj_pack, oproj, oproj_unpack;
    double gate_mm, route, shared_mm, shared_swiglu;
    double expert_io, expert_mm, expert_swiglu, expert_sum;
    double bind, trunk_io;

    double f_qkv, f_oproj, f_gate, f_shared, f_expert, f_comp, f_idx, f_attn;

    uint64_t e_hits, e_misses, e_bytes, trunk_bytes;
    int have_estats;
} DSV4PrefillStats;

struct DSV4Prefill {
    const DSV4Cfg   *c;
    DSV4Trunk       *tr;
    DSV4LayerState  *lst;
    const DSV4ExpertSrc *src;
    const DSV4ModelW *mw;
    DSV4PrefillOpts  opt;
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
    float *out;        /* [B][hidden] */
    float *xh;         /* [hidden] embed scratch */
    DSV4Scratch *scratch; /* [B] per-token scratch for the sequential parts */

    double t_total, t_attn, t_idx, t_gate, t_shared, t_moe;
    double t_io, t_trunk, t_bind, t_hc;

    DSV4PrefillStats st;   /* per-call sub-stage accounting, see above */
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
    if (p->opt.batch > DSV4_MAX_BATCH) p->opt.batch = DSV4_MAX_BATCH;
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
    p->out = xcalloc((size_t)B * d * sizeof(float));
    p->xh = xcalloc((size_t)d * sizeof(float));
    p->scratch = (DSV4Scratch *)xcalloc((size_t)B * sizeof(DSV4Scratch));

    if (!p->H || !p->x1 || !p->resid || !p->post || !p->comb || !p->mixes ||
        !p->qr || !p->q || !p->kv || !p->o || !p->ogrp || !p->xg || !p->yg ||
        !p->gate || !p->gate_orig || !p->topk_idx || !p->topk_w ||
        !p->sh_gate || !p->sh_up || !p->sh_mid || !p->sh_out ||
        !p->out || !p->xh || !p->scratch) {
        dsv4_prefill_free(p);
        return -1;
    }
    for (int t = 0; t < B; t++)
        if (dsv4_scratch_init(&p->scratch[t], c, p->opt.max_pos) != 0) {
            dsv4_prefill_free(p);
            return -1;
        }

    fprintf(stderr, "prefill: CPU dense batch, %d token(s) per chunk\n", p->B);
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
    free(p->sh_up); free(p->sh_mid); free(p->sh_out);
    free(p->out); free(p->xh);
    free(p);
}

/* The one batched-matmul seam. Every job is a dense matrix over `n` token
 * rows; dsv4_mmq_n dispatches to the format's batch kernel, or to the GEMV for
 * n == 1 and for formats that do not have one yet (FP4). */
static int pm_n_multi(DSV4Prefill *p, int layer, const DSV4PMJob *jobs, int nj)
{
    (void)p; (void)layer;
    for (int j = 0; j < nj; j++) {
        if (jobs[j].n < 1) continue;
        dsv4_mmq_n(jobs[j].y, jobs[j].x, jobs[j].m, jobs[j].n);
    }
    return 0;
}

static int pm_n(DSV4Prefill *p, int layer, float *y, const float *x,
                const DSV4QMat *m, int n)
{
    DSV4PMJob job = { m, x, y, n };
    return pm_n_multi(p, layer, &job, 1);
}

/* Timed/flop-accounted wrappers. Every batched dense matmul goes through one
 * of these two so no projection can silently fall out of the report. */
static void pf_flops(double *acc, const DSV4QMat *m, int n)
{
    *acc += 2.0 * (double)m->rows * (double)m->cols * (double)n * 1e-9;
}

static int pm_n_stats(DSV4Prefill *p, float *y, const float *x,
                      const DSV4QMat *m, int n, double *timer, double *flops)
{
    const double t0 = pnow();
    const int rc = pm_n(p, 0, y, x, m, n);
    *timer += pnow() - t0;
    if (flops) pf_flops(flops, m, n);
    return rc;
}

static int pm_multi_stats(DSV4Prefill *p, const DSV4PMJob *jobs, int nj,
                          double *timer, double *flops)
{
    const double t0 = pnow();
    const int rc = pm_n_multi(p, 0, jobs, nj);
    *timer += pnow() - t0;
    if (flops)
        for (int j = 0; j < nj; j++) pf_flops(flops, jobs[j].m, jobs[j].n);
    return rc;
}

/* Routed-expert matmul, timed per matrix so the FP4 MM is not confused with
 * the SwiGLU that sits between w1/w3 and w2. */
static void expert_forward(DSV4Prefill *p, float *out, const float *x,
                           const DSV4ExpertW *e, DSV4Scratch *s,
                           int hidden, int inter, float limit)
{
    double t0 = pnow();
    dsv4_mmq(s->expert_gate, x, &e->w1);
    dsv4_mmq(s->expert_up,   x, &e->w3);
    p->st.expert_mm += pnow() - t0;

    t0 = pnow();
    dsv4_swiglu(s->expert_out, s->expert_gate, s->expert_up, inter, limit);
    p->st.expert_swiglu += pnow() - t0;

    t0 = pnow();
    dsv4_mmq(out, s->expert_out, &e->w2);
    p->st.expert_mm += pnow() - t0;

    /* w1 [inter,hidden], w3 [inter,hidden], w2 [hidden,inter] -> 6*inter*hidden
     * flops per expert forward. */
    p->st.f_expert += 6.0 * (double)inter * (double)hidden * 1e-9;
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

    if (pm_n_stats(p, p->qr, p->x1, &w->attn.wq_a, n,
                   &p->st.proj_qkv, &p->st.f_qkv)) return -1;
    if (pm_n_stats(p, p->kv, p->x1, &w->attn.wkv, n,
                   &p->st.proj_qkv, &p->st.f_qkv)) return -1;
    {
        const double ts = pnow();
        for (int t = 0; t < n; t++)
            dsv4_rmsnorm(p->qr + (size_t)t * ql, p->qr + (size_t)t * ql,
                         w->attn.q_norm, ql, c->rms_eps);
        p->st.q_norm += pnow() - ts;
    }
    if (pm_n_stats(p, p->q, p->qr, &w->attn.wq_b, n,
                   &p->st.proj_qkv, &p->st.f_qkv)) return -1;

    for (int t = 0; t < n; t++) {
        const int pos = pos0 + t;
        float *qt = p->q + (size_t)t * (size_t)H * hd;
        double ts = pnow();
        for (int h = 0; h < H; h++) {
            float *qh = qt + (size_t)h * hd;
            double sq = 0.0;
            for (int i = 0; i < hd; i++) sq += (double)qh[i] * qh[i];
            const float inv = (float)(1.0 / sqrt(sq / (double)hd +
                                                 (double)c->rms_eps));
            for (int i = 0; i < hd; i++) qh[i] *= inv;
            dsv4_rope_apply(qh, hd, rd, cs, sn, pos, rd / 2, 0);
        }
        p->st.q_head += pnow() - ts;

        float *kvt = p->kv + (size_t)t * hd;
        ts = pnow();
        dsv4_rmsnorm(kvt, kvt, w->attn.kv_norm, hd, c->rms_eps);
        dsv4_rope_apply(kvt, hd, rd, cs, sn, pos, rd / 2, 0);
        memcpy(kvc + (size_t)(pos % c->sliding_window) * hd, kvt,
               (size_t)hd * sizeof(float));
        p->st.kv_norm += pnow() - ts;

        if (comp) {
            const int ratio = w->compress_ratio;
            ts = pnow();
            dsv4_mmq(p->scratch[t].comp_kv_in, p->x1 + (size_t)t * d,
                     &w->attn.comp.wkv);
            dsv4_mmq(p->scratch[t].comp_sc_in, p->x1 + (size_t)t * d,
                     &w->attn.comp.wgate);
            p->st.comp_proj += pnow() - ts;
            p->st.f_comp += 2.0 * ((double)w->attn.comp.wkv.rows *
                                   (double)w->attn.comp.wkv.cols +
                                   (double)w->attn.comp.wgate.rows *
                                   (double)w->attn.comp.wgate.cols) * 1e-9;
            ts = pnow();
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
            p->st.comp_step += pnow() - ts;
        }

        ts = pnow();
        int n_idx = dsv4_window_idxs(p->scratch[t].idxs, pos,
                                     c->sliding_window);
        p->st.win_idx += pnow() - ts;
        const int avail = st->n_compressed;
        if (avail > 0) {
            if (avail > p->scratch[t].idx_cap) {
                fprintf(stderr, "prefill: %d compressed rows but idx_scores "
                                "holds %d\n", avail, p->scratch[t].idx_cap);
                return -1;
            }
            int take = avail;
            if (w->has_idx) {
                const double ti = pnow();
                ts = pnow();
                dsv4_mmq(p->scratch[t].idx_q, p->qr + (size_t)t * ql,
                         &w->attn.idx.wq_b);
                p->st.idx_proj += pnow() - ts;
                ts = pnow();
                for (int h = 0; h < c->index_n_heads; h++)
                    dsv4_rope_apply(p->scratch[t].idx_q +
                                        (size_t)h * c->index_head_dim,
                                    c->index_head_dim, rd, cs, sn, pos,
                                    rd / 2, 0);
                p->st.idx_rope += pnow() - ts;
                ts = pnow();
                dsv4_mmq(p->scratch[t].idx_w, p->x1 + (size_t)t * d,
                         &w->attn.idx.weights_proj);
                p->st.idx_proj += pnow() - ts;
                p->st.f_idx += 2.0 * ((double)w->attn.idx.wq_b.rows *
                                      (double)w->attn.idx.wq_b.cols +
                                      (double)w->attn.idx.weights_proj.rows *
                                      (double)w->attn.idx.weights_proj.cols)
                             * 1e-9;
                ts = pnow();
                dsv4_indexer_score(p->scratch[t].idx_scores,
                                   p->scratch[t].idx_q,
                                   kvc + (size_t)c->sliding_window * hd,
                                   p->scratch[t].idx_w, c->index_n_heads,
                                   c->index_head_dim, avail);
                take = avail < c->index_topk ? avail : c->index_topk;
                dsv4_topk(p->scratch[t].idxs + n_idx,
                          p->scratch[t].idx_scores, avail, take);
                p->st.idx_score += pnow() - ts;
                p->t_idx += pnow() - ti;
            } else {
                ts = pnow();
                for (int i = 0; i < take; i++)
                    p->scratch[t].idxs[n_idx + i] = i;
                p->st.idx_score += pnow() - ts;
            }
            dsv4_indexer_offset(p->scratch[t].idxs + n_idx, take,
                                c->sliding_window);
            n_idx += take;
        }

        float *ot = p->o + (size_t)t * (size_t)H * hd;
        ts = pnow();
        dsv4_sparse_attn(ot, qt, kvc, w->attn.sink, p->scratch[t].idxs,
                         H, hd, n_idx, 1.0f / sqrtf((float)hd),
                         p->scratch[t].attn_scratch);
        p->st.attn_core += pnow() - ts;
        p->st.f_attn += 4.0 * (double)H * (double)hd * (double)n_idx * 1e-9;

        ts = pnow();
        for (int h = 0; h < H; h++)
            dsv4_rope_apply(ot + (size_t)h * hd, hd, rd, cs, sn, pos,
                            rd / 2, 1);
        p->st.attn_deroep += pnow() - ts;
    }

    /* Grouped low-rank output projection, one GEMM per group. */
    const int gw = H * hd / c->o_groups;
    const int gr = c->o_lora;
    const size_t esz = (w->attn.wo_a.wdt == DSV4_WBF16) ? 2u : 1u;
    const size_t row_bytes = (w->attn.wo_a.wdt == DSV4_WFP4)
                           ? (size_t)gw / 2u : (size_t)gw * esz;
    if (c->o_groups > 16) return -1;
    DSV4QMat slices[16];
    DSV4PMJob jobs[16];
    double ts = pnow();
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
    p->st.oproj_pack += pnow() - ts;
    if (pm_multi_stats(p, jobs, c->o_groups,
                       &p->st.oproj, &p->st.f_oproj)) return -1;
    ts = pnow();
    for (int g = 0; g < c->o_groups; g++)
        for (int t = 0; t < n; t++)
            memcpy(p->ogrp + (size_t)t * (size_t)c->o_groups * c->o_lora +
                       (size_t)g * gr,
                   p->yg + (size_t)g * p->B * gr + (size_t)t * gr,
                   (size_t)gr * sizeof(float));
    p->st.oproj_unpack += pnow() - ts;
    if (pm_n_stats(p, p->out, p->ogrp, &w->attn.wo_b, n,
                   &p->st.oproj, &p->st.f_oproj)) return -1;
    ts = pnow();
    for (int t = 0; t < n; t++)
        dsv4_hc_post(p->H + (size_t)t * (size_t)hc * d,
                     p->out + (size_t)t * d,
                     p->resid + (size_t)t * (size_t)hc * d,
                     p->post + (size_t)t * hc,
                     p->comb + (size_t)t * hc * hc, hc, d);
    p->st.hc_post += pnow() - ts;

    p->t_attn += pnow() - t0;
    return 0;
}

static int prefill_moe(DSV4Prefill *p, int L, const DSV4LayerW *w,
                       int n, const int32_t *ids)
{
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const int inter = c->moe_inter * c->n_shared;
    const int rinter = c->moe_inter;
    const int topk = c->topk;

    const double tg = pnow();
    if (pm_n_stats(p, p->gate, p->x1, &w->moe.gate, n,
                   &p->st.gate_mm, &p->st.f_gate)) return -1;
    {
        const double tr = pnow();
        for (int t = 0; t < n; t++) {
            dsv4_route(p->topk_idx + (size_t)t * topk,
                       p->topk_w + (size_t)t * topk,
                       p->gate + (size_t)t * c->n_experts,
                       p->gate_orig + (size_t)t * c->n_experts,
                       w->moe.bias, w->moe.tid2eid, ids[t],
                       c->n_experts, topk, c->routed_scale);
        }
        p->st.route += pnow() - tr;
    }
    p->t_gate += pnow() - tg;

    const double tsh = pnow();
    {
        DSV4PMJob two[2] = {
            { &w->moe.shared.w1, p->x1, p->sh_gate, n },
            { &w->moe.shared.w3, p->x1, p->sh_up, n },
        };
        if (pm_multi_stats(p, two, 2,
                           &p->st.shared_mm, &p->st.f_shared)) return -1;
    }
    {
        const double ts = pnow();
        for (int t = 0; t < n; t++)
            dsv4_swiglu(p->sh_mid + (size_t)t * inter,
                        p->sh_gate + (size_t)t * inter,
                        p->sh_up + (size_t)t * inter, inter, c->swiglu_limit);
        p->st.shared_swiglu += pnow() - ts;
    }
    if (pm_n_stats(p, p->sh_out, p->sh_mid, &w->moe.shared.w2, n,
                   &p->st.shared_mm, &p->st.f_shared)) return -1;
    p->t_shared += pnow() - tsh;

    /* ROUTED EXPERTS STAY PER TOKEN.
     *
     * This is the one place the batched path deliberately does not union the
     * batch's top-k requests. A 32-token chunk asks for 192 experts but only
     * ~65 are distinct, so the union saves ~3x reads -- and then destroys the
     * cache hits that let the per-token path overlap half its reads with
     * resident-expert matmuls. The measured result was +50% exposed I/O.
     *
     * Layer-major order already gives the useful part for free: all tokens of
     * one layer run back to back, so a duplicate request from the next token
     * hits the cache without any union. The routed sum stays in k order per
     * token, exactly as in dsv4_layer.c. */
    for (int t = 0; t < n; t++) {
        DSV4Scratch *s = &p->scratch[t];
        const float *x = p->x1 + (size_t)t * d;
        float *o = p->out + (size_t)t * d;
        const int *tidx = p->topk_idx + (size_t)t * topk;
        const float *tw = p->topk_w + (size_t)t * topk;
        const DSV4ExpertW *ex[DSV4_MAX_TOPK];
        int done[DSV4_MAX_TOPK];
        for (int k = 0; k < topk; k++) { ex[k] = NULL; done[k] = 0; }

        const int overlapped = (p->src->begin && p->src->end);
        double ti0 = pnow();
        if (overlapped)         p->src->begin(p->src->ctx, L, tidx, topk, ex);
        else if (p->src->get_many)
                                p->src->get_many(p->src->ctx, L, tidx, topk, ex);
        else for (int k = 0; k < topk; k++)
                                ex[k] = p->src->get(p->src->ctx, L, tidx[k]);
        {
            const double td = pnow() - ti0;
            p->t_io += td;
            p->st.expert_io += td;
        }
        if (overlapped) {
            double tm0 = pnow();
            for (int k = 0; k < topk; k++) {
                if (!ex[k]) continue;
                expert_forward(p, s->expert_acc +
                                   (size_t)k * s->expert_acc_stride,
                               x, ex[k], s, d, rinter, c->swiglu_limit);
                done[k] = 1;
            }
            p->t_moe += pnow() - tm0;

            if (p->src->wait_any) {
                for (;;) {
                    int k = -1;
                    ti0 = pnow();
                    const int got = p->src->wait_any(p->src->ctx, &k);
                    const double td = pnow() - ti0;
                    p->t_io += td;
                    p->st.expert_io += td;
                    if (!got) break;
                    if (k < 0 || k >= topk || !ex[k]) continue;
                    tm0 = pnow();
                    expert_forward(p, s->expert_acc +
                                       (size_t)k * s->expert_acc_stride,
                                   x, ex[k], s, d, rinter, c->swiglu_limit);
                    p->t_moe += pnow() - tm0;
                    done[k] = 1;
                }
            }

            ti0 = pnow();
            p->src->end(p->src->ctx, ex);
            const double td = pnow() - ti0;
            p->t_io += td;
            p->st.expert_io += td;
        }

        double tm0 = pnow();
        for (int k = 0; k < topk; k++) {
            if (!ex[k]) {
                fprintf(stderr, "prefill: layer %d could not obtain routed "
                                "expert %d of %d for token %d; refusing to "
                                "emit a token routed through fewer experts\n",
                        L, k + 1, topk, t);
                return -1;
            }
            if (!done[k])
                expert_forward(p, s->expert_acc +
                                   (size_t)k * s->expert_acc_stride,
                               x, ex[k], s, d, rinter, c->swiglu_limit);
        }
        {
            const double tsum = pnow();
            for (int i = 0; i < d; i++) o[i] = 0.0f;
            for (int k = 0; k < topk; k++) {
                const float wk = tw[k];
                const float *acc = s->expert_acc + (size_t)k * s->expert_acc_stride;
                for (int i = 0; i < d; i++) o[i] += wk * acc[i];
            }
            for (int i = 0; i < d; i++) o[i] += p->sh_out[(size_t)t * d + i];
            p->st.expert_sum += pnow() - tsum;
        }
        p->t_moe += pnow() - tm0;
    }

    {
        const double ts = pnow();
        for (int t = 0; t < n; t++)
            dsv4_hc_post(p->H + (size_t)t * (size_t)hc * d,
                         p->out + (size_t)t * d,
                         p->resid + (size_t)t * (size_t)hc * d,
                         p->post + (size_t)t * hc,
                         p->comb + (size_t)t * hc * hc, hc, d);
        p->st.hc_post_moe += pnow() - ts;
    }
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
        double ts = pnow();
        memcpy(p->resid + (size_t)t * hcd, p->H + (size_t)t * hcd,
               hcd * sizeof(float));
        dsv4_hc_pre(p->x1 + (size_t)t * d,
                    p->post + (size_t)t * hc,
                    p->comb + (size_t)t * hc * hc,
                    p->mixes + (size_t)t * mix_hc,
                    p->H + (size_t)t * hcd, &w->hc_attn, hc, d,
                    c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
        p->st.hc_pre += pnow() - ts;
        ts = pnow();
        dsv4_rmsnorm(p->x1 + (size_t)t * d, p->x1 + (size_t)t * d,
                     w->attn_norm, d, c->rms_eps);
        p->st.norm += pnow() - ts;
    }
    p->t_hc += pnow() - thc;

    if (prefill_attention(p, L, w, n, pos0, cs, sn)) return -1;

    const double thc2 = pnow();
    for (int t = 0; t < n; t++) {
        double ts = pnow();
        memcpy(p->resid + (size_t)t * hcd,
               p->H + (size_t)t * hcd, hcd * sizeof(float));
        dsv4_hc_pre(p->x1 + (size_t)t * d,
                    p->post + (size_t)t * hc,
                    p->comb + (size_t)t * hc * hc,
                    p->mixes + (size_t)t * mix_hc,
                    p->H + (size_t)t * hcd, &w->hc_ffn, hc, d,
                    c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
        p->st.hc_pre += pnow() - ts;
        ts = pnow();
        dsv4_rmsnorm(p->x1 + (size_t)t * d, p->x1 + (size_t)t * d,
                     w->ffn_norm, d, c->rms_eps);
        p->st.norm += pnow() - ts;
    }
    p->t_hc += pnow() - thc2;

    if (prefill_moe(p, L, w, n, ids)) return -1;
    return 0;
}

static void report_progress(DSV4Prefill *p, const char *stage,
                            int chunk, int chunks, int layer, int layers,
                            int tokens_done, int tokens_total, int chunk_tokens,
                            double elapsed, double progress)
{
    if (!p->opt.on_progress) return;
    DSV4PrefillProgress pr;
    pr.stage = stage;
    pr.chunk = chunk;
    pr.chunks = chunks;
    pr.layer = layer;
    pr.layers = layers;
    pr.tokens_done = tokens_done;
    pr.tokens_total = tokens_total;
    pr.chunk_tokens = chunk_tokens;
    /* The batch the caller actually ran, not the configured cap. A short
     * prompt gets one shrunken chunk (3 tokens, not 16), and a long prompt's
     * last chunk is whatever is left. The TUI prints this value, so reporting
     * the cap here made it lie at both ends. */
    pr.batch = chunk_tokens > 0 ? chunk_tokens : p->B;
    pr.elapsed = elapsed;
    pr.progress = progress < 0.0 ? 0.0 : (progress > 1.0 ? 1.0 : progress);
    p->opt.on_progress(p->opt.progress_ctx, &pr);
}

static void pf_row(const char *name, double sec, double total, int n,
                   double flop)
{
    if (sec <= 0.0 && flop <= 0.0) return;
    const double pct = total > 0.0 ? 100.0 * sec / total : 0.0;
    const double mst = n > 0 ? 1e3 * sec / (double)n : 0.0;
    if (flop > 0.0) {
        const double gfs = sec > 0.0 ? flop / sec : 0.0;
        fprintf(stderr, "    %-26s %8.3f s %6.1f%% %9.2f ms %9.2f GF %7.1f GF/s\n",
                name, sec, pct, mst, flop, gfs);
    } else {
        fprintf(stderr, "    %-26s %8.3f s %6.1f%% %9.2f ms\n",
                name, sec, pct, mst);
    }
}

static void pf_report(DSV4Prefill *p, int n, int nchunks, int first_m)
{
    const DSV4PrefillStats *s = &p->st;
    const double T = p->t_total;
    const double dense_sec = s->proj_qkv + s->oproj + s->gate_mm + s->shared_mm;
    const double dense_gf  = s->f_qkv + s->f_oproj + s->f_gate + s->f_shared;

    fprintf(stderr,
            "  prefill stages: %d tokens, %d chunk(s) of <=%d, %.3f s, "
            "%.3f s/token\n",
            n, nchunks, first_m, T, n > 0 ? T / (double)n : 0.0);
    fprintf(stderr, "    %-26s %8s %7s %10s %10s %8s\n",
            "stage", "sec", "%", "ms/token", "GFLOP", "GF/s");

    fprintf(stderr, "  [embed / mHC / norms]\n");
    pf_row("embed", s->embed, T, n, 0.0);
    pf_row("hc_pre (attn+ffn)", s->hc_pre, T, n, 0.0);
    pf_row("rmsnorm (attn+ffn)", s->norm, T, n, 0.0);
    pf_row("hc_post (attention)", s->hc_post, T, n, 0.0);
    pf_row("hc_post (ffn)", s->hc_post_moe, T, n, 0.0);

    fprintf(stderr, "  [attention]\n");
    pf_row("qkv projections", s->proj_qkv, T, n, s->f_qkv);
    pf_row("q norm (q_lora)", s->q_norm, T, n, 0.0);
    pf_row("q head norm + rope", s->q_head, T, n, 0.0);
    pf_row("kv norm + rope + store", s->kv_norm, T, n, 0.0);
    pf_row("compressor projections", s->comp_proj, T, n, s->f_comp);
    pf_row("compress_step", s->comp_step, T, n, 0.0);
    pf_row("window index build", s->win_idx, T, n, 0.0);
    pf_row("indexer projections", s->idx_proj, T, n, s->f_idx);
    pf_row("indexer rope", s->idx_rope, T, n, 0.0);
    pf_row("indexer score + top-k", s->idx_score, T, n, 0.0);
    pf_row("sparse attention", s->attn_core, T, n, s->f_attn);
    pf_row("attention output de-rope", s->attn_deroep, T, n, 0.0);
    pf_row("o-proj pack", s->oproj_pack, T, n, 0.0);
    pf_row("o-proj group GEMMs", s->oproj, T, n, s->f_oproj);
    pf_row("o-proj unpack", s->oproj_unpack, T, n, 0.0);

    fprintf(stderr, "  [MoE]\n");
    pf_row("router matmul", s->gate_mm, T, n, s->f_gate);
    pf_row("router top-k", s->route, T, n, 0.0);
    pf_row("shared expert matmuls", s->shared_mm, T, n, s->f_shared);
    pf_row("shared expert swiglu", s->shared_swiglu, T, n, 0.0);
    pf_row("routed expert I/O", s->expert_io, T, n, 0.0);
    pf_row("routed expert matmuls", s->expert_mm, T, n, s->f_expert);
    pf_row("routed expert swiglu", s->expert_swiglu, T, n, 0.0);
    pf_row("routed weighted sum", s->expert_sum, T, n, 0.0);

    fprintf(stderr, "  [trunk / bind]\n");
    pf_row("trunk read", s->trunk_io, T, n, 0.0);
    pf_row("trunk bind", s->bind, T, n, 0.0);

    fprintf(stderr, "  [kernel summary]\n");
    pf_row("dense projections", dense_sec, T, n, dense_gf);
    pf_row("routed experts (FP4)", s->expert_mm, T, n, s->f_expert);
    pf_row("sparse attention core", s->attn_core, T, n, s->f_attn);
    pf_row("compressor projections", s->comp_proj, T, n, s->f_comp);
    pf_row("indexer projections", s->idx_proj, T, n, s->f_idx);

    /* Top-level buckets. These do not overlap: t_hc is the two hc_pre/norm
     * passes, t_attn the whole attention half, t_gate/t_shared/t_moe the MoE
     * halves, and the two I/O buckets are the blocking calls. The remainder is
     * progress callbacks, the chunk loop and clock overhead. */
    const double covered = s->embed + p->t_hc + p->t_attn + p->t_gate +
                           p->t_shared + p->t_moe + s->expert_io +
                           s->trunk_io + s->bind + s->hc_post_moe;
    fprintf(stderr,
            "    top-level: hc %.3f  attn %.3f  router %.3f  shared %.3f  "
            "routed %.3f  exp-io %.3f  trunk %.3f  bind %.3f  other %.3f s\n",
            p->t_hc, p->t_attn, p->t_gate, p->t_shared, p->t_moe,
            s->expert_io, s->trunk_io, s->bind,
            T - covered);

    if (s->have_estats) {
        const uint64_t req = s->e_hits + s->e_misses;
        const double gb = (double)s->e_bytes / 1073741824.0;
        fprintf(stderr,
                "    expert cache: %llu requests, %llu hits (%.1f%%), "
                "%llu misses, %.2f GB read, %.2f GB/s over prefill\n",
                (unsigned long long)req, (unsigned long long)s->e_hits,
                req ? 100.0 * (double)s->e_hits / (double)req : 0.0,
                (unsigned long long)s->e_misses, gb,
                T > 0.0 ? gb / T : 0.0);
    }
    if (s->trunk_bytes) {
        const double gb = (double)s->trunk_bytes / 1073741824.0;
        fprintf(stderr, "    trunk data: %.2f GB read over prefill\n", gb);
    }
}

int dsv4_prefill_range(DSV4Prefill *p, const int32_t *ids, int n, int pos0,
                       const float *cs_d, const float *sn_d,
                       const float *cs_c, const float *sn_c,
                       float *h_out)
{
    if (!p || !ids || n <= 0 || !h_out || pos0 < 0) return -1;
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const double t0 = pnow();

    /* A resident engine calls this once per turn. Reset the accumulators so
     * this turn's profile is not the previous turn's plus this one. */
    p->t_total = p->t_attn = p->t_idx = p->t_gate = p->t_shared = 0.0;
    p->t_moe = p->t_io = p->t_trunk = 0.0;
    p->t_bind = p->t_hc = 0.0;
    memset(&p->st, 0, sizeof p->st);

    DSV4ExpertStats es0, es1;
    memset(&es0, 0, sizeof es0);
    memset(&es1, 0, sizeof es1);
    if (p->src->stats &&
        p->src->stats(p->src->ctx, &es0) == 0)
        p->st.have_estats = 1;
    const uint64_t trunk0 = p->tr ? p->tr->bytes_read : 0;

    const int nchunks = (n + p->B - 1) / p->B;
    const int first_m = n < p->B ? n : p->B;
    int chunk_idx = 0;
    report_progress(p, "start", 0, nchunks, 0, c->n_layers,
                    0, n, first_m, 0.0, 0.0);

    int off = 0;
    int last_m = 1;
    while (off < n) {
        int m = n - off;
        if (m > p->B) m = p->B;
        last_m = m;
        chunk_idx++;
        report_progress(p, "chunk", chunk_idx, nchunks, 0, c->n_layers,
                        off, n, m, pnow() - t0, (double)off / (double)n);

        {
            const double te = pnow();
            for (int t = 0; t < m; t++) {
                dsv4_embed_row(p->xh, p->mw->embed, p->mw->wdt, ids[off + t], d);
                for (int j = 0; j < hc; j++)
                    memcpy(p->H + (size_t)t * hcd + (size_t)j * d, p->xh,
                           (size_t)d * sizeof(float));
            }
            p->st.embed += pnow() - te;
        }

        for (int L = 0; L < c->n_layers; L++) {
            DSV4LayerBind lb;
            const double tb = pnow();
            const double io0 = p->tr->load_seconds;
            if (dsv4_trunk_bind(p->tr, c, L, &lb) != 0) return -1;
            const double io = p->tr->load_seconds - io0;
            const double call = pnow() - tb;
            p->t_io += io;
            p->t_trunk += io;
            p->t_bind += call > io ? call - io : 0.0;
            p->st.trunk_io += io;
            p->st.bind += call > io ? call - io : 0.0;

            const int comp = dsv4_compress_ratio(c, L) != 0;
            const float *cs = comp ? cs_c : cs_d;
            const float *sn = comp ? sn_c : sn_d;
            if (prefill_layer(p, L, &lb.w, m, pos0 + off, ids + off, cs, sn))
                return -1;

            const double frac = (double)off +
                (double)m * (double)(L + 1) / (double)c->n_layers;
            report_progress(p, "layer", chunk_idx, nchunks, L + 1,
                            c->n_layers, (int)frac, n, m, pnow() - t0,
                            frac / (double)n);
        }
        report_progress(p, "chunk_done", chunk_idx, nchunks, c->n_layers,
                        c->n_layers, off + m, n, m, pnow() - t0,
                        (double)(off + m) / (double)n);
        off += m;
    }
    report_progress(p, "done", nchunks, nchunks, c->n_layers, c->n_layers,
                    n, n, last_m, pnow() - t0, 1.0);

    memcpy(h_out, p->H + (size_t)(last_m - 1) * hcd, hcd * sizeof(float));
    p->t_total = pnow() - t0;

    if (p->st.have_estats) {
        if (p->src->stats(p->src->ctx, &es1) == 0) {
            p->st.e_hits = es1.hits - es0.hits;
            p->st.e_misses = es1.misses - es0.misses;
            p->st.e_bytes = es1.bytes_read - es0.bytes_read;
        }
    }
    if (p->tr) p->st.trunk_bytes = p->tr->bytes_read - trunk0;

    /* Fold the prompt's timers into the same report decode uses, so the
     * end-of-run "where the time went" is not 90% unaccounted. `t_io` carries
     * both expert reads and the chunk's trunk reads; split them so the report
     * keeps its two different meanings. */
    dsv4_prof.attn       += p->t_attn;
    dsv4_prof.idxer      += p->t_idx;
    dsv4_prof.gate       += p->t_gate;
    dsv4_prof.expert_mm  += p->t_moe;
    dsv4_prof.shared     += p->t_shared;
    dsv4_prof.hc         += p->t_hc;
    dsv4_prof.trunk_io   += p->t_trunk;
    dsv4_prof.bind       += p->t_bind;
    dsv4_prof.expert_io  += p->t_io - p->t_trunk;

    fprintf(stderr,
            "prefill: %d tokens at pos %d, %d chunk(s) of <=%d, %.2f s "
            "(%.3f s/token); attn %.2f moe %.2f io %.2f bind %.2f hc %.2f\n",
            n, pos0, nchunks, first_m, p->t_total,
            p->t_total / (double)n, p->t_attn, p->t_moe, p->t_io,
            p->t_bind, p->t_hc);
    pf_report(p, n, nchunks, first_m);
    return 0;
}

int dsv4_prefill_prompt(DSV4Prefill *p, const int32_t *ids, int n,
                        const float *cs_d, const float *sn_d,
                        const float *cs_c, const float *sn_c,
                        float *h_out)
{
    return dsv4_prefill_range(p, ids, n, 0,
                              cs_d, sn_d, cs_c, sn_c, h_out);
}

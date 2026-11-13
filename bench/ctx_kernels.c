/* SPDX-License-Identifier: Apache-2.0 */
/* bench/ctx_kernels.c - the context-dependent kernels, old versus new.
 *
 * WHY THIS EXISTS
 *   The decode benchmark in BENCHMARK.md uses a three-token prompt. That leaves
 *   the context at ~23 positions, where the indexer has nothing to select from
 *   and sparse attention looks at a nearly empty window. The costs that grow
 *   with context - top-k selection, indexer scoring, the attention dot and the
 *   mHC mixes - are invisible there.
 *
 *   This bench builds a SYNTHETIC KV cache of the real shapes and runs only the
 *   kernels. No checkpoint, no trunk, no disk. It is safe to run while the
 *   machine is busy with I/O, which is the point.
 *
 * WHAT IS COMPARED
 *   The reference copies below are the pre-optimisation algorithms, kept here
 *   so every new variant is checked against the implementation it replaced
 *   before it is timed. The production functions are linked from src/core.
 *
 * HOW TO READ THE SUMMARY
 *   The per-token estimate is anchored on the published short-context decode
 *   rate (BASE_DECODE_S, default 0.965 s/token) and adds the measured
 *   context-dependent delta from that anchor. It is a model, not an end-to-end
 *   measurement: it assumes every context-independent component stays put.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dsv4.h"

/* Production kernels. */
int  dsv4_topk(int *, const float *, int, int);
void dsv4_indexer_score(float *, const float *, const float *, const float *,
                        int, int, int);
void dsv4_sparse_attn(float *, const float *, const float *, const float *,
                      const int *, int, int, int, float, float *);
void dsv4_hc_pre(float *, float *, float *, float *, const float *,
                 const DSV4HcW *, int, int, int, float, float);
void dsv4_hc_post(float *, const float *, const float *, const float *,
                  const float *, int, int);
void dsv4_hc_split_sinkhorn(float *, float *, float *, const float *,
                            const float *, const float *, int, int, float);

#define MAX_REPS 32
#define H_DIM 64
#define HEAD_DIM 512
#define I_HEADS 64
#define I_DIM 128
#define WINDOW 128
#define INDEX_TOPK 512
#define HC_MULT 4
#define HC_HIDDEN 4096
#define HC_MIX ((2 + HC_MULT) * HC_MULT)
#define BASE_DECODE_S 0.965

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return v[n / 2];
}

static unsigned g_rng = 0x9e3779b9u;
static float frand(float scale)
{
    g_rng = g_rng * 1103515245u + 12345u;
    return (float)((int)((g_rng >> 16) & 0xffff) - 32768) * scale / 32768.0f;
}

/* --------------------------------------------------------- reference copies ---
 * Exactly the code these kernels had before the context work. They are the
 * definition the production versions are checked against, and they never enter
 * the engine. */

static int topk_ref(int *idxs, const float *scores, int n, int k)
{
    if (k > n) k = n;
    for (int i = 0; i < k; i++) {
        int best = -1;
        for (int c = 0; c < n; c++) {
            int taken = 0;
            for (int j = 0; j < i; j++) if (idxs[j] == c) { taken = 1; break; }
            if (taken) continue;
            if (best < 0 || scores[c] > scores[best]) best = c;
        }
        idxs[i] = best;
    }
    return k;
}

static void indexer_score_ref(float *out, const float *q, const float *kv,
                              const float *weights, int n_heads, int head_dim,
                              int n_pos)
{
    const float scale = (1.0f / sqrtf((float)head_dim))
                      * (1.0f / sqrtf((float)n_heads));
    for (int t = 0; t < n_pos; t++) {
        const float *kt = kv + (size_t)t * head_dim;
        double total = 0.0;
        for (int h = 0; h < n_heads; h++) {
            const float *qh = q + (size_t)h * head_dim;
            double dot = 0.0;
            for (int c = 0; c < head_dim; c++)
                dot = fma((double)qh[c], (double)kt[c], dot);
            if (dot > 0.0) total += dot * (double)weights[h];
        }
        out[t] = (float)total * scale;
    }
}

static void sparse_attn_ref(float *o, const float *q, const float *kv,
                            const float *sink, const int *idxs,
                            int h, int d, int topk, float scale, float *scratch)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (h > 4)
#endif
    for (int i = 0; i < h; i++) {
        float *s = scratch + (size_t)i * topk;
        const float *qi = q + (size_t)i * d;
        float mx = -INFINITY;
        for (int j = 0; j < topk; j++) {
            const int idx = idxs[j];
            if (idx < 0) { s[j] = -INFINITY; continue; }
            const float *kvj = kv + (size_t)idx * d;
            double acc = 0.0;
            for (int c = 0; c < d; c++)
                acc = fma((double)qi[c], (double)kvj[c], acc);
            s[j] = (float)acc * scale;
            if (s[j] > mx) mx = s[j];
        }
        double denom = 0.0;
        for (int j = 0; j < topk; j++) {
            if (idxs[j] < 0) { s[j] = 0.0f; continue; }
            s[j] = expf(s[j] - mx);
            denom += (double)s[j];
        }
        denom += (double)expf(sink[i] - mx);
        float *oi = o + (size_t)i * d;
        for (int c = 0; c < d; c++) oi[c] = 0.0f;
        for (int j = 0; j < topk; j++) {
            const int idx = idxs[j];
            if (idx < 0) continue;
            const float w = s[j];
            if (w == 0.0f) continue;
            const float *kvj = kv + (size_t)idx * d;
            for (int c = 0; c < d; c++) oi[c] += w * kvj[c];
        }
        const float inv = (float)(1.0 / denom);
        for (int c = 0; c < d; c++) oi[c] *= inv;
    }
}

static void hc_pre_ref(float *y, float *post, float *comb, float *mixes,
                       const float *x, const DSV4HcW *w, int hc, int hidden,
                       int iters, float norm_eps, float hc_eps)
{
    const int hcdim = hc * hidden;
    const int mix_hc = (2 + hc) * hc;
    double sq = 0.0;
    for (int i = 0; i < hcdim; i++) sq += (double)x[i] * (double)x[i];
    const float rsqrt = (float)(1.0 / sqrt(sq / (double)hcdim + (double)norm_eps));
    for (int m = 0; m < mix_hc; m++) {
        const float *row = w->fn + (size_t)m * hcdim;
        double acc = 0.0;
        for (int i = 0; i < hcdim; i++) acc = fma((double)row[i], (double)x[i], acc);
        mixes[m] = (float)acc * rsqrt;
    }
    float pre[DSV4_MAX_HC_MULT];
    dsv4_hc_split_sinkhorn(pre, post, comb, mixes, w->scale, w->base,
                           hc, iters, hc_eps);
    for (int d = 0; d < hidden; d++) {
        double acc = 0.0;
        for (int j = 0; j < hc; j++)
            acc += (double)pre[j] * (double)x[(size_t)j * hidden + d];
        y[d] = (float)acc;
    }
}

static void hc_post_ref(float *y, const float *x, const float *residual,
                        const float *post, const float *comb, int hc, int hidden)
{
    for (int k = 0; k < hc; k++) {
        float *out = y + (size_t)k * hidden;
        for (int d = 0; d < hidden; d++) {
            double acc = (double)post[k] * (double)x[d];
            for (int j = 0; j < hc; j++)
                acc += (double)comb[j * hc + k]
                     * (double)residual[(size_t)j * hidden + d];
            out[d] = (float)acc;
        }
    }
}

/* ------------------------------------------------------------- correctness --- */

static double l2_rel(const float *a, const float *b, int n)
{
    double e = 0.0, r = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        e += d * d;
        r += (double)a[i] * (double)a[i];
    }
    return sqrt(e / (r > 0.0 ? r : 1.0));
}

static int same_indices(const int *a, const int *b, int n)
{
    return memcmp(a, b, (size_t)n * sizeof *a) == 0;
}

/* ------------------------------------------------------------------- data --- */

typedef struct {
    int ctx;
    int avail4, avail128;
    int n4, n128, n0;
    int take4;
    int rows;                 /* kv rows allocated */
    float *q, *kv, *sink, *o_ref, *o_new, *scratch;
    float *iq, *ikv, *iw, *sc_ref, *sc_new;
    int *idx4, *idx128, *idx0;
    int *tk_ref, *tk_new;
} Case;

static void case_free(Case *c)
{
    free(c->q); free(c->kv); free(c->sink); free(c->o_ref); free(c->o_new);
    free(c->scratch); free(c->iq); free(c->ikv); free(c->iw);
    free(c->sc_ref); free(c->sc_new);
    free(c->idx4); free(c->idx128); free(c->idx0);
    free(c->tk_ref); free(c->tk_new);
    memset(c, 0, sizeof *c);
}

static int case_init(Case *c, int ctx, int max_avail)
{
    memset(c, 0, sizeof *c);
    c->ctx = ctx;
    if (ctx <= 0) {
        c->avail4 = c->avail128 = 0;
        c->n4 = c->n128 = c->n0 = 24;
        c->take4 = 0;
    } else {
        c->avail4   = (ctx + 1) / 4;
        c->avail128 = (ctx + 1) / 128;
        c->take4    = c->avail4 < INDEX_TOPK ? c->avail4 : INDEX_TOPK;
        c->n4       = WINDOW + c->take4;
        c->n128     = WINDOW + c->avail128;
        c->n0       = ctx < WINDOW ? ctx + 1 : WINDOW;
    }
    c->rows = WINDOW + (max_avail > c->avail128 ? max_avail : c->avail128);

    const size_t nq = (size_t)H_DIM * HEAD_DIM;
    const size_t nkv = (size_t)c->rows * HEAD_DIM;
    c->q = malloc(nq * sizeof *c->q);
    c->kv = malloc(nkv * sizeof *c->kv);
    c->sink = malloc(H_DIM * sizeof *c->sink);
    c->o_ref = malloc(nq * sizeof *c->o_ref);
    c->o_new = malloc(nq * sizeof *c->o_new);
    c->scratch = malloc((size_t)H_DIM * (size_t)(WINDOW + INDEX_TOPK + 8)
                        * sizeof *c->scratch);
    c->idx4 = malloc((size_t)(WINDOW + INDEX_TOPK) * sizeof *c->idx4);
    c->idx128 = malloc((size_t)(WINDOW + 4096) * sizeof *c->idx128);
    c->idx0 = malloc((size_t)WINDOW * sizeof *c->idx0);
    c->sc_ref = malloc((size_t)(max_avail ? max_avail : 1) * sizeof *c->sc_ref);
    c->sc_new = malloc((size_t)(max_avail ? max_avail : 1) * sizeof *c->sc_new);
    c->tk_ref = malloc((size_t)INDEX_TOPK * sizeof *c->tk_ref);
    c->tk_new = malloc((size_t)INDEX_TOPK * sizeof *c->tk_new);
    c->iq = malloc((size_t)I_HEADS * I_DIM * sizeof *c->iq);
    c->ikv = malloc((size_t)(max_avail ? max_avail : 1) * I_DIM * sizeof *c->ikv);
    c->iw = malloc(I_HEADS * sizeof *c->iw);

    if (!c->q || !c->kv || !c->sink || !c->o_ref || !c->o_new || !c->scratch
        || !c->idx4 || !c->idx128 || !c->idx0 || !c->sc_ref || !c->sc_new
        || !c->tk_ref || !c->tk_new || !c->iq || !c->ikv || !c->iw)
        return -1;

    for (size_t i = 0; i < nq; i++) { c->q[i] = frand(0.03f); c->o_ref[i] = 0.0f; c->o_new[i] = 0.0f; }
    for (size_t i = 0; i < nkv; i++) c->kv[i] = frand(0.03f);
    for (int i = 0; i < H_DIM; i++) c->sink[i] = frand(0.1f);
    for (size_t i = 0; i < (size_t)I_HEADS * I_DIM; i++) c->iq[i] = frand(0.03f);
    for (size_t i = 0; i < (size_t)(max_avail ? max_avail : 1) * I_DIM; i++)
        c->ikv[i] = frand(0.03f);
    for (int i = 0; i < I_HEADS; i++) c->iw[i] = frand(0.5f) + 0.5f;

    for (int j = 0; j < WINDOW; j++) {
        /* Ring slot, made non-negative for ctx < j. All 128 window rows are
         * valid once the window is full, which is the regime this bench is
         * about; ctx0 uses only the first 24 as the short-context anchor. */
        const int slot = ((ctx - j) % WINDOW + WINDOW) % WINDOW;
        c->idx0[j] = slot;
        c->idx4[j] = slot;
        c->idx128[j] = slot;
    }
    for (int i = 0; i < c->take4; i++) {
        if (c->avail4 <= INDEX_TOPK)
            c->idx4[WINDOW + i] = WINDOW + i;
        else
            c->idx4[WINDOW + i] = WINDOW + (int)((long long)i * c->avail4 / c->take4);
    }
    for (int i = 0; i < c->avail128; i++)
        c->idx128[WINDOW + i] = WINDOW + i;

    /* A non-degenerate score field with forced ties, so the tie rule is
     * exercised by the correctness comparison as well as by the unit gate. */
    for (int i = 0; i < max_avail; i++) {
        c->sc_ref[i] = frand(0.02f);
        if (i % 17 == 0) c->sc_ref[i] = 0.25f;
    }
    return 0;
}

/* ----------------------------------------------------------------- timing --- */

typedef void (*attn_fn)(float *, const float *, const float *, const float *,
                        const int *, int, int, int, float, float *);

static double time_topk(Case *c, int ref, int reps)
{
    double t[MAX_REPS];
    if (c->avail4 <= 0) return 0.0;
    for (int r = 0; r < reps; r++) {
        const double a = now_s();
        if (ref) topk_ref(c->tk_ref, c->sc_ref, c->avail4, c->take4);
        else     dsv4_topk(c->tk_new, c->sc_ref, c->avail4, c->take4);
        t[r] = now_s() - a;
    }
    return median(t, reps);
}

static double time_indexer(Case *c, int ref, int reps)
{
    double t[MAX_REPS];
    if (c->avail4 <= 0) return 0.0;
    for (int r = 0; r < reps; r++) {
        const double a = now_s();
        if (ref)
            indexer_score_ref(c->sc_ref, c->iq, c->ikv, c->iw,
                              I_HEADS, I_DIM, c->avail4);
        else
            dsv4_indexer_score(c->sc_new, c->iq, c->ikv, c->iw,
                               I_HEADS, I_DIM, c->avail4);
        t[r] = now_s() - a;
    }
    return median(t, reps);
}

static double time_attn(Case *c, attn_fn fn, const int *idxs, int n_idx,
                        float *o, int reps)
{
    double t[MAX_REPS];
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int r = 0; r < reps; r++) {
        const double a = now_s();
        fn(o, c->q, c->kv, c->sink, idxs, H_DIM, HEAD_DIM, n_idx, scale,
           c->scratch);
        t[r] = now_s() - a;
    }
    return median(t, reps);
}

static double time_hc_pre(const DSV4HcW *w, float *y, float *post, float *comb,
                          float *mixes, const float *x, int ref, int reps)
{
    double t[MAX_REPS];
    for (int r = 0; r < reps; r++) {
        const double a = now_s();
        if (ref) hc_pre_ref(y, post, comb, mixes, x, w, HC_MULT, HC_HIDDEN,
                            20, 1e-6f, 1e-6f);
        else     dsv4_hc_pre(y, post, comb, mixes, x, w, HC_MULT, HC_HIDDEN,
                             20, 1e-6f, 1e-6f);
        t[r] = now_s() - a;
    }
    return median(t, reps);
}

static double time_hc_post(const float *x, const float *res, const float *post,
                           const float *comb, float *y, int ref, int reps)
{
    double t[MAX_REPS];
    for (int r = 0; r < reps; r++) {
        const double a = now_s();
        if (ref) hc_post_ref(y, x, res, post, comb, HC_MULT, HC_HIDDEN);
        else     dsv4_hc_post(y, x, res, post, comb, HC_MULT, HC_HIDDEN);
        t[r] = now_s() - a;
    }
    return median(t, reps);
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 9;
    if (reps < 1) reps = 1;
    if (reps > MAX_REPS) reps = MAX_REPS;
#ifdef _OPENMP
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(4);
    const int nth = omp_get_max_threads();
#else
    const int nth = 1;
#endif

    static const int def_ctx[] = { 0, 1024, 2048, 3072, 4096, 5120, 6144 };
    int nctx = (int)(sizeof def_ctx / sizeof def_ctx[0]);
    int ctxs[32];
    if (argc > 2) {
        nctx = argc - 2;
        if (nctx > 32) nctx = 32;
        for (int i = 0; i < nctx; i++) ctxs[i] = atoi(argv[i + 2]);
    } else {
        for (int i = 0; i < nctx; i++) ctxs[i] = def_ctx[i];
    }

    int max_avail = 0;
    for (int i = 0; i < nctx; i++)
        if (ctxs[i] > 0) {
            const int a = (ctxs[i] + 1) / 4;
            if (a > max_avail) max_avail = a;
        }
    if (max_avail < 1) max_avail = 1;

    printf("ctx kernel bench: %d thread(s), %d rep(s), %d bit%s\n",
           nth, reps, 64, " (no checkpoint, no I/O)");
    printf("reference: short-context decode = %.3f s/token\n\n", BASE_DECODE_S);

    /* Fixed mHC workload. */
    float *hx = malloc((size_t)HC_MULT * HC_HIDDEN * sizeof *hx);
    float *hres = malloc((size_t)HC_MULT * HC_HIDDEN * sizeof *hres);
    float *hfn = malloc((size_t)HC_MIX * HC_MULT * HC_HIDDEN * sizeof *hfn);
    float *hbase = malloc(HC_MIX * sizeof *hbase);
    float *hscale = malloc(3 * sizeof *hscale);
    /* hc_pre writes `hidden`; hc_post writes hc*hidden into the same buffer. */
    float *hy_ref = malloc((size_t)HC_MULT * HC_HIDDEN * sizeof *hy_ref);
    float *hy_new = malloc((size_t)HC_MULT * HC_HIDDEN * sizeof *hy_new);
    float *hpost_ref = malloc(HC_MULT * sizeof *hpost_ref);
    float *hpost_new = malloc(HC_MULT * sizeof *hpost_new);
    float *hcomb_ref = malloc(HC_MULT * HC_MULT * sizeof *hcomb_ref);
    float *hcomb_new = malloc(HC_MULT * HC_MULT * sizeof *hcomb_new);
    float *hmix_ref = malloc(HC_MIX * sizeof *hmix_ref);
    float *hmix_new = malloc(HC_MIX * sizeof *hmix_new);
    DSV4HcW hw;
    for (int i = 0; i < HC_MULT * HC_HIDDEN; i++) {
        hx[i] = frand(0.02f);
        hres[i] = frand(0.02f);
    }
    for (int i = 0; i < HC_MIX * HC_MULT * HC_HIDDEN; i++) hfn[i] = frand(0.001f);
    for (int i = 0; i < HC_MIX; i++) hbase[i] = frand(0.01f);
    for (int i = 0; i < 3; i++) hscale[i] = 1.0f;
    hw.fn = hfn; hw.base = hbase; hw.scale = hscale;

    const double hpre_ref = time_hc_pre(&hw, hy_ref, hpost_ref, hcomb_ref,
                                        hmix_ref, hx, 1, reps);
    const double hpre_new = time_hc_pre(&hw, hy_new, hpost_new, hcomb_new,
                                        hmix_new, hx, 0, reps);

    double hc_yerr = 0.0, hc_mixerr = 0.0, hc_pcerr = 0.0;
    for (int i = 0; i < HC_HIDDEN; i++) {
        const double d1 = fabs((double)hy_ref[i] - hy_new[i]);
        if (d1 > hc_yerr) hc_yerr = d1;
    }
    for (int i = 0; i < HC_MIX; i++) {
        const double d1 = fabs((double)hmix_ref[i] - hmix_new[i]);
        if (d1 > hc_mixerr) hc_mixerr = d1;
    }
    for (int i = 0; i < HC_MULT; i++) {
        const double d1 = fabs((double)hpost_ref[i] - hpost_new[i]);
        if (d1 > hc_pcerr) hc_pcerr = d1;
    }

    const double hpost_ref_t = time_hc_post(hx, hres, hpost_ref, hcomb_ref,
                                            hy_ref, 1, reps);
    const double hpost_new_t = time_hc_post(hx, hres, hpost_ref, hcomb_ref,
                                            hy_new, 0, reps);
    double hc_posterr = 0.0;
    for (int i = 0; i < HC_MULT * HC_HIDDEN; i++) {
        const double d1 = fabs((double)hy_ref[i] - hy_new[i]);
        if (d1 > hc_posterr) hc_posterr = d1;
    }

    printf("mHC (hidden=%d, hc=%d):\n", HC_HIDDEN, HC_MULT);
    printf("  hc_pre   ref %7.3f ms  new %7.3f ms  %5.2fx  "
           "verify max|dy|=%.2e |dmix|=%.2e |dpost|=%.2e\n",
           hpre_ref * 1e3, hpre_new * 1e3,
           hpre_new > 0 ? hpre_ref / hpre_new : 0.0,
           hc_yerr, hc_mixerr, hc_pcerr);
    printf("  hc_post  ref %7.3f ms  new %7.3f ms  %5.2fx  "
           "verify max|dy|=%.2e\n",
           hpost_ref_t * 1e3, hpost_new_t * 1e3,
           hpost_new_t > 0 ? hpost_ref_t / hpost_new_t : 0.0, hc_posterr);

    /* Anchor: the short-context component cost, which BASE_DECODE_S already
     * contains. The summary adds only the context-dependent delta. */
    Case c0;
    if (case_init(&c0, 0, 1) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }
    const double a0_ref = time_attn(&c0, sparse_attn_ref, c0.idx0, c0.n0, c0.o_ref, reps);
    const double a0_new = time_attn(&c0, dsv4_sparse_attn, c0.idx0, c0.n0, c0.o_new, reps);
    const double comp0_ref = 43.0 * a0_ref * 1e3
                           + 86.0 * (hpre_ref + hpost_ref_t) * 1e3;
    const double comp0_new = 43.0 * a0_new * 1e3
                           + 86.0 * (hpre_new + hpost_new_t) * 1e3;
    printf("  anchor ctx0: attn ref %.3f ms / new %.3f ms; "
           "components ref %.1f / new %.1f ms/token\n\n",
           a0_ref * 1e3, a0_new * 1e3, comp0_ref, comp0_new);
    case_free(&c0);

    printf("%5s %7s %7s | %8s %8s | %8s %8s | %8s %8s | %8s %8s | %8s %8s\n",
           "ctx", "avail4", "n_idx4", "topk ref", "topk new",
           "idx ref", "idx new", "attn4 ref", "attn4 new",
           "attn128 r", "attn128 n", "tok/s old", "tok/s new");

    for (int ci = 0; ci < nctx; ci++) {
        Case c;
        if (case_init(&c, ctxs[ci], max_avail) != 0) {
            fprintf(stderr, "alloc failed at ctx %d\n", ctxs[ci]);
            return 1;
        }

        const double tk_ref = time_topk(&c, 1, reps);
        const double tk_new = time_topk(&c, 0, reps);
        const int tk_same = c.avail4 > 0
            ? same_indices(c.tk_ref, c.tk_new, c.take4) : 1;

        const double ix_ref = time_indexer(&c, 1, reps);
        const double ix_new = time_indexer(&c, 0, reps);
        const double ix_l2 = c.avail4 > 0 ? l2_rel(c.sc_ref, c.sc_new, c.avail4) : 0.0;

        const double a4_ref = time_attn(&c, sparse_attn_ref, c.idx4, c.n4,
                                        c.o_ref, reps);
        const double a4_new = time_attn(&c, dsv4_sparse_attn, c.idx4, c.n4,
                                        c.o_new, reps);
        const double a4_l2 = l2_rel(c.o_ref, c.o_new, H_DIM * HEAD_DIM);

        const double a128_ref = time_attn(&c, sparse_attn_ref, c.idx128, c.n128,
                                          c.o_ref, reps);
        const double a128_new = time_attn(&c, dsv4_sparse_attn, c.idx128, c.n128,
                                          c.o_new, reps);

        const double attn0_ref = time_attn(&c, sparse_attn_ref, c.idx0, c.n0,
                                           c.o_ref, reps);
        const double attn0_new = time_attn(&c, dsv4_sparse_attn, c.idx0, c.n0,
                                           c.o_new, reps);

        /* Per-token component sums in ms, using the real 43-layer mix of the
         * Flash decoder (the trailing MTP entries are not run):
         * 21 ratio-4/indexer, 20 ratio-128, 2 dense. */
        double comp_ref = 21.0 * (tk_ref + ix_ref + a4_ref) * 1e3
                        + 20.0 * a128_ref * 1e3
                        + 2.0 * attn0_ref * 1e3
                        + 86.0 * (hpre_ref + hpost_ref_t) * 1e3;
        double comp_new = 21.0 * (tk_new + ix_new + a4_new) * 1e3
                        + 20.0 * a128_new * 1e3
                        + 2.0 * attn0_new * 1e3
                        + 86.0 * (hpre_new + hpost_new_t) * 1e3;
        /* ctx0 IS the anchor; use the anchor's own measurement so that row
         * reads exactly BASE_DECODE_S rather than a second timing of the same
         * work. */
        if (ctxs[ci] <= 0) { comp_ref = comp0_ref; comp_new = comp0_new; }
        /* BASE_DECODE_S already contains the OLD ctx0 components, so both arms
         * are anchored on comp0_ref. Anchoring the new arm on comp0_new would
         * hide exactly the short-context gain this work also produces. */
        const double s_old = BASE_DECODE_S + (comp_ref - comp0_ref) / 1e3;
        const double s_new = BASE_DECODE_S + (comp_new - comp0_ref) / 1e3;

        printf("%5d %7d %7d | %8.3f %8.3f | %8.3f %8.3f | %8.3f %8.3f | "
               "%8.3f %8.3f | %8.3f %8.3f\n",
               ctxs[ci], c.avail4, c.n4,
               tk_ref * 1e3, tk_new * 1e3,
               ix_ref * 1e3, ix_new * 1e3,
               a4_ref * 1e3, a4_new * 1e3,
               a128_ref * 1e3, a128_new * 1e3,
               1.0 / s_old, 1.0 / s_new);
        printf("      checks: topk %s, indexer L2 %.1e, attn L2 %.1e, "
               "s/token old %.3f new %.3f\n",
               tk_same ? "exact" : "DIFFER", ix_l2, a4_l2, s_old, s_new);
        fflush(stdout);

        case_free(&c);
    }

    free(hx); free(hres); free(hfn); free(hbase); free(hscale);
    free(hy_ref); free(hy_new); free(hpost_ref); free(hpost_new);
    free(hcomb_ref); free(hcomb_new); free(hmix_ref); free(hmix_new);
    return 0;
}

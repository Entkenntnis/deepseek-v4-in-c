/* SPDX-License-Identifier: Apache-2.0 */
/* bench.c - compute-only microbenchmarks for DeepSeek-V4-Flash kernels.
 *
 * WHAT THIS IS
 *   One process, one table, no disk. Every weight matrix is allocated in RAM
 *   and filled with a deterministic, realistic fixture before it is timed.
 *   The benchmark links against src/core only; it never opens a checkpoint,
 *   a trunk or the expert cache. A regression here is a CPU/kernel regression,
 *   not an I/O one.
 *
 * WHY THE WORKING SET IS ROTATED
 *   A repeated GEMV over one 4 MB FP4 matrix would sit in L3 and report an
 *   LLC-bandwidth number. Real decode reads ~80 MB of routed experts per
 *   layer, so the microbench rotates several copies of each matrix and keeps
 *   the working set above the 12 MB L3 of the target CPU. That is why a FP8
 *   matrix gets two copies and a small one gets four or eight.
 *
 * FIXTURES
 *   FP4 uses e2m1 magnitudes with a non-uniform distribution and E8M0 scales
 *   in a realistic exponent range. FP8 e4m3 codes are normal-exponent values
 *   except for a subnormal in ~1.8% of 128-byte blocks, matching the measured
 *   0.0145% per byte in the real checkpoint and therefore the fast/slow-path
 *   mix of the production kernel. BF16 values are finite and in a sane range.
 *
 * USAGE
 *   ./bin/bench [--filter SUBSTR] [--threads N] [--pin off|all|p|e]
 *
 * The default is the engine's own default: four threads pinned to the P-cores.
 */
#define _GNU_SOURCE

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_threads.h"

#define PAT_BYTES 65536

static const char *g_filter;
static uint64_t g_rng = 0x243F6A8885A308D3ULL;

static void die(const char *what)
{
    fprintf(stderr, "bench: %s\n", what);
    exit(1);
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint32_t rnd32(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 32);
}

static float frnd(void)
{
    return (float)(rnd32() >> 8) * (1.0f / 16777216.0f);
}

static void *xalloc(size_t n)
{
    void *p = NULL;
    if (n == 0) n = 64;
    if (posix_memalign(&p, 64, n) != 0 || !p) die("allocation failed");
    return p;
}

static void fill_f32(float *p, size_t n, float amp)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (frnd() * 2.0f - 1.0f) * amp;
}

/* ------------------------------------------------------------- fixtures --- */

static void fill_bf16_pattern(uint8_t *pat)
{
    uint16_t *h = (uint16_t *)pat;
    for (int i = 0; i < PAT_BYTES / 2; i++) {
        uint16_t v = (uint16_t)(0x3F00u + (rnd32() % 0x0200u)); /* 0.5 .. 8 */
        if (rnd32() & 1u) v |= 0x8000u;
        h[i] = v;
    }
}

static void fill_fp8_pattern(uint8_t *pat)
{
    for (int i = 0; i < PAT_BYTES; i++) {
        /* One subnormal per ~6800 bytes is the measured rate, and makes about
         * 1.8% of 128-byte blocks take the exact slow path. */
        if ((rnd32() % 6800u) == 0u) {
            pat[i] = 0x01u;
            continue;
        }
        const uint32_t sign = rnd32() & 1u;
        uint32_t exp = 1u + (rnd32() % 15u);
        uint32_t man = rnd32() & 7u;
        if (exp == 15u && man == 7u) man = 6u;      /* 0x7F is NaN */
        pat[i] = (uint8_t)((sign << 7) | (exp << 3) | man);
    }
}

static uint8_t fp4_code(void)
{
    static const int cum[8] = { 8, 32, 58, 78, 90, 96, 99, 100 };
    const int r = (int)(rnd32() % 100u);
    int mag = 0;
    while (mag < 7 && r >= cum[mag]) mag++;
    uint8_t c = (uint8_t)mag;
    if (rnd32() & 1u) c |= 8u;
    return c;
}

static void fill_fp4_pattern(uint8_t *pat)
{
    for (int i = 0; i < PAT_BYTES; i++)
        pat[i] = (uint8_t)(fp4_code() | (fp4_code() << 4));
}

static void fill_bytes(void *dst, size_t n, int wdt)
{
    uint8_t *pat = (uint8_t *)xalloc(PAT_BYTES);
    if (wdt == DSV4_WBF16)      fill_bf16_pattern(pat);
    else if (wdt == DSV4_WFP8)  fill_fp8_pattern(pat);
    else                        fill_fp4_pattern(pat);

    uint8_t *d = (uint8_t *)dst;
    for (size_t off = 0; off < n; off += PAT_BYTES) {
        const size_t k = (n - off < PAT_BYTES) ? (n - off) : PAT_BYTES;
        memcpy(d + off, pat, k);
    }
    free(pat);
}

static void fill_scales(uint8_t *p, size_t n, int fp4)
{
    const unsigned lo = fp4 ? 118u : 120u;
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)(lo + (rnd32() % 7u));
}

typedef struct {
    DSV4QMat m;
    void    *w;
    void    *s;
    size_t   wb, sb;
} QFix;

static void qfix_init(QFix *q, int wdt, int rows, int cols)
{
    memset(q, 0, sizeof *q);
    q->m.wdt  = wdt;
    q->m.rows = rows;
    q->m.cols = cols;

    if (wdt == DSV4_WBF16) {
        q->wb = (size_t)rows * (size_t)cols * 2u;
        q->w  = xalloc(q->wb);
        fill_bytes(q->w, q->wb, wdt);
        q->m.blk_r = q->m.blk_c = 0;
    } else if (wdt == DSV4_WFP8) {
        q->wb = (size_t)rows * (size_t)cols;
        q->w  = xalloc(q->wb);
        fill_bytes(q->w, q->wb, wdt);
        const int sr = (rows + 127) / 128;
        const int sc = (cols + 127) / 128;
        q->sb = (size_t)sr * (size_t)sc;
        q->s  = xalloc(q->sb);
        fill_scales((uint8_t *)q->s, q->sb, 0);
        q->m.blk_r = 128;
        q->m.blk_c = 128;
    } else {
        q->wb = (size_t)rows * ((size_t)cols / 2u);
        q->w  = xalloc(q->wb);
        fill_bytes(q->w, q->wb, wdt);
        const int sr = rows;
        const int sc = (cols + 31) / 32;
        q->sb = (size_t)sr * (size_t)sc;
        q->s  = xalloc(q->sb);
        fill_scales((uint8_t *)q->s, q->sb, 1);
        q->m.blk_r = 1;
        q->m.blk_c = 32;
    }
    q->m.w = q->w;
    q->m.s = q->s;
}

static void qfix_free(QFix *q)
{
    free(q->w);
    free(q->s);
    memset(q, 0, sizeof *q);
}

static double qfix_bytes(const QFix *q, int nt)
{
    return (double)q->wb + (double)q->sb
         + (double)nt * (double)q->m.cols * 4.0
         + (double)nt * (double)q->m.rows * 4.0;
}

static const char *fmt_name(int wdt)
{
    switch (wdt) {
    case DSV4_WBF16: return "bf16";
    case DSV4_WFP8:  return "fp8";
    case DSV4_WFP4:  return "fp4";
    default:         return "f32";
    }
}

/* ------------------------------------------------------------- timing ----- */

typedef void (*RunFn)(void *);

static double measure(RunFn fn, void *ctx, int target_ms, int *out_reps)
{
    fn(ctx);
    double t0 = now_s();
    int reps = 0;
    double elapsed;
    do {
        fn(ctx);
        reps++;
        elapsed = now_s() - t0;
    } while ((elapsed * 1000.0 < (double)target_ms || reps < 3)
             && reps < 200000);
    *out_reps = reps;
    return elapsed;
}

static void report(const char *name, const char *shape, const char *fmt,
                   double sec, int reps, int calls, double flops, double bytes)
{
    const double per_ms = sec / ((double)reps * (double)calls) * 1e3;
    const double gfs = flops > 0.0 ? flops * (double)reps / sec / 1e9 : 0.0;
    const double gbs = bytes > 0.0 ? bytes * (double)reps / sec / 1e9 : 0.0;

    printf("%-30s %-20s %-5s %10.4f", name, shape, fmt, per_ms);
    if (flops > 0.0) printf(" %9.1f", gfs); else printf(" %9s", "-");
    if (bytes > 0.0) printf(" %9.2f", gbs); else printf(" %9s", "-");
    printf("\n");
    fflush(stdout);
}

static int skip(const char *name)
{
    return g_filter && !strstr(name, g_filter);
}

/* ---------------------------------------------------------- qmat cases ---- */

typedef struct {
    QFix *f;
    int   ncalls, ncopies, nt, cur;
    float *x, *y;
} QCase;

static void qcase_run(void *p)
{
    QCase *q = (QCase *)p;
    const int base = (q->cur % q->ncopies) * q->ncalls;
    q->cur++;
    for (int c = 0; c < q->ncalls; c++)
        dsv4_mmq_n(q->y, q->x, &q->f[base + c].m, q->nt);
}

static void bench_qmat(const char *name, const char *shape, int wdt,
                       int rows, int cols, int nt, int ncalls, int ncopies,
                       int target_ms)
{
    if (skip(name)) return;

    QCase q;
    memset(&q, 0, sizeof q);
    q.ncalls  = ncalls;
    q.ncopies = ncopies;
    q.nt      = nt;
    q.f = (QFix *)calloc((size_t)ncalls * (size_t)ncopies, sizeof(QFix));
    if (!q.f) die("allocation failed");
    for (int i = 0; i < ncalls * ncopies; i++)
        qfix_init(&q.f[i], wdt, rows, cols);

    q.x = (float *)xalloc((size_t)nt * (size_t)cols * sizeof(float));
    q.y = (float *)xalloc((size_t)nt * (size_t)rows * sizeof(float));
    fill_f32(q.x, (size_t)nt * (size_t)cols, 1.0f);

    /* Touch every rotated copy once. Without this the first timed pass over a
     * copy pays its page faults and the average depends on how many copies
     * fit in the measurement window. */
    for (int i = 0; i < ncopies; i++) qcase_run(&q);

    int reps = 0;
    const double sec = measure(qcase_run, &q, target_ms, &reps);
    const double flops = 2.0 * (double)nt * (double)rows * (double)cols
                       * (double)ncalls;
    const double bytes = qfix_bytes(&q.f[0], nt) * (double)ncalls;
    report(name, shape, fmt_name(wdt), sec, reps, ncalls, flops, bytes);

    for (int i = 0; i < ncalls * ncopies; i++) qfix_free(&q.f[i]);
    free(q.f);
    free(q.x);
    free(q.y);
}

/* One routed expert: gate/up matmuls, clamped SwiGLU, down matmul. */
typedef struct {
    QFix   w1, w3, w2;
    float *x, *g, *u, *mid, *y;
} TrioCase;

static void trio_run(void *p)
{
    TrioCase *t = (TrioCase *)p;
    dsv4_mmq(t->g, t->x, &t->w1.m);
    dsv4_mmq(t->u, t->x, &t->w3.m);
    dsv4_swiglu(t->mid, t->g, t->u, 2048, 10.0f);
    dsv4_mmq(t->y, t->mid, &t->w2.m);
}

static void bench_expert_trio(void)
{
    if (skip("expert_trio")) return;

    TrioCase t;
    memset(&t, 0, sizeof t);
    qfix_init(&t.w1, DSV4_WFP4, 2048, 4096);
    qfix_init(&t.w3, DSV4_WFP4, 2048, 4096);
    qfix_init(&t.w2, DSV4_WFP4, 4096, 2048);
    t.x   = (float *)xalloc(4096u * sizeof(float));
    t.g   = (float *)xalloc(2048u * sizeof(float));
    t.u   = (float *)xalloc(2048u * sizeof(float));
    t.mid = (float *)xalloc(2048u * sizeof(float));
    t.y   = (float *)xalloc(4096u * sizeof(float));
    fill_f32(t.x, 4096, 1.0f);

    int reps = 0;
    const double sec = measure(trio_run, &t, 40, &reps);
    const double flops = 2.0 * 2048.0 * 4096.0 * 3.0 + 6.0 * 2048.0;
    const double bytes = (double)(t.w1.wb + t.w1.sb + t.w3.wb + t.w3.sb
                                + t.w2.wb + t.w2.sb)
                       + (double)(4096 + 3 * 2048 + 4096) * 4.0;
    report("expert_trio", "3 x fp4", "fp4", sec, reps, 1, flops, bytes);

    qfix_free(&t.w1);
    qfix_free(&t.w3);
    qfix_free(&t.w2);
    free(t.x); free(t.g); free(t.u); free(t.mid); free(t.y);
}

/* ------------------------------------------------------- elementwise ------ */

typedef struct { float *y, *x, *w; int n; } TernCase;

static void rms_run(void *p)
{
    TernCase *c = (TernCase *)p;
    dsv4_rmsnorm(c->y, c->x, c->w, c->n, 1e-6f);
}

static void swi_run(void *p)
{
    TernCase *c = (TernCase *)p;
    dsv4_swiglu(c->y, c->x, c->w, c->n, 10.0f);
}

static void bench_simple(const char *name, const char *shape, int n,
                         RunFn fn, int kind)
{
    if (skip(name)) return;

    TernCase c;
    memset(&c, 0, sizeof c);
    c.n = n;
    c.y = (float *)xalloc((size_t)n * sizeof(float));
    c.x = (float *)xalloc((size_t)n * sizeof(float));
    c.w = (float *)xalloc((size_t)n * sizeof(float));
    fill_f32(c.x, (size_t)n, 1.0f);
    fill_f32(c.w, (size_t)n, 1.0f);

    int reps = 0;
    const double sec = measure(fn, &c, 40, &reps);
    const double flops = (kind == 0 ? 3.0 : 6.0) * (double)n;
    const double bytes = 3.0 * (double)n * 4.0;
    report(name, shape, "f32", sec, reps, 1, flops, bytes);

    free(c.y); free(c.x); free(c.w);
}

typedef struct {
    int   *idx;
    float *wts, *scores, *orig, *bias, *base;
    int    n, k;
} RouteCase;

static void route_run(void *p)
{
    RouteCase *r = (RouteCase *)p;
    memcpy(r->scores, r->base, (size_t)r->n * sizeof(float));
    dsv4_route(r->idx, r->wts, r->scores, r->orig, r->bias, NULL, 0,
               r->n, r->k, 2.5f);
}

static void bench_route(void)
{
    if (skip("route")) return;

    RouteCase r;
    memset(&r, 0, sizeof r);
    r.n = 256;
    r.k = 6;
    r.idx    = (int *)xalloc(8u * sizeof(int));
    r.wts    = (float *)xalloc(8u * sizeof(float));
    r.scores = (float *)xalloc(256u * sizeof(float));
    r.orig   = (float *)xalloc(256u * sizeof(float));
    r.bias   = (float *)xalloc(256u * sizeof(float));
    r.base   = (float *)xalloc(256u * sizeof(float));
    fill_f32(r.base, 256, 4.0f);
    fill_f32(r.bias, 256, 0.1f);

    int reps = 0;
    const double sec = measure(route_run, &r, 40, &reps);
    const double flops = 8.0 * 256.0 + 6.0 * 256.0;
    const double bytes = (4.0 * 256.0 + 2.0 * 8.0 + 2.0 * 256.0) * 4.0;
    report("route", "256 -> 6", "f32", sec, reps, 1, flops, bytes);

    free(r.idx); free(r.wts); free(r.scores); free(r.orig);
    free(r.bias); free(r.base);
}

/* ------------------------------------------------------------------ mHC --- */

typedef struct {
    float *y, *post, *comb, *mixes, *x;
    float *fn, *base, *scale;
    DSV4HcW w;
    int    hc, hidden, iters;
} HcPreCase;

static void hcpre_run(void *p)
{
    HcPreCase *c = (HcPreCase *)p;
    dsv4_hc_pre(c->y, c->post, c->comb, c->mixes, c->x, &c->w,
                c->hc, c->hidden, c->iters, 1e-6f, 1e-6f);
}

static void bench_hc_pre(void)
{
    if (skip("hc_pre")) return;

    const int hc = 4, hidden = 4096, mix = (2 + hc) * hc;
    const int hcdim = hc * hidden;

    HcPreCase c;
    memset(&c, 0, sizeof c);
    c.hc = hc; c.hidden = hidden; c.iters = 20;
    c.y      = (float *)xalloc((size_t)hidden * sizeof(float));
    c.post   = (float *)xalloc((size_t)hc * sizeof(float));
    c.comb   = (float *)xalloc((size_t)hc * hc * sizeof(float));
    c.mixes  = (float *)xalloc((size_t)mix * sizeof(float));
    c.x      = (float *)xalloc((size_t)hcdim * sizeof(float));
    c.fn     = (float *)xalloc((size_t)mix * hcdim * sizeof(float));
    c.base   = (float *)xalloc((size_t)mix * sizeof(float));
    c.scale  = (float *)xalloc(3u * sizeof(float));
    fill_f32(c.x, (size_t)hcdim, 1.0f);
    fill_f32(c.fn, (size_t)mix * hcdim, 0.02f);
    fill_f32(c.base, (size_t)mix, 0.1f);
    c.scale[0] = c.scale[1] = c.scale[2] = 1.0f;
    c.w.fn = c.fn; c.w.base = c.base; c.w.scale = c.scale;

    int reps = 0;
    const double sec = measure(hcpre_run, &c, 40, &reps);
    const double flops = 2.0 * (double)mix * (double)hcdim
                       + 2.0 * (double)hc * (double)hidden;
    const double bytes = ((double)mix * hcdim + hcdim + hidden
                        + mix + hc * hc) * 4.0;
    report("hc_pre", "hc4 hidden4096", "f32/f64", sec, reps, 1, flops, bytes);

    free(c.y); free(c.post); free(c.comb); free(c.mixes); free(c.x);
    free(c.fn); free(c.base); free(c.scale);
}

typedef struct {
    float *y, *x, *resid, *post, *comb;
    int hc, hidden;
} HcPostCase;

static void hcpost_run(void *p)
{
    HcPostCase *c = (HcPostCase *)p;
    dsv4_hc_post(c->y, c->x, c->resid, c->post, c->comb, c->hc, c->hidden);
}

static void bench_hc_post(void)
{
    if (skip("hc_post")) return;

    const int hc = 4, hidden = 4096;
    HcPostCase c;
    memset(&c, 0, sizeof c);
    c.hc = hc; c.hidden = hidden;
    c.y     = (float *)xalloc((size_t)hc * hidden * sizeof(float));
    c.x     = (float *)xalloc((size_t)hidden * sizeof(float));
    c.resid = (float *)xalloc((size_t)hc * hidden * sizeof(float));
    c.post  = (float *)xalloc((size_t)hc * sizeof(float));
    c.comb  = (float *)xalloc((size_t)hc * hc * sizeof(float));
    fill_f32(c.x, (size_t)hidden, 1.0f);
    fill_f32(c.resid, (size_t)hc * hidden, 1.0f);
    fill_f32(c.post, (size_t)hc, 1.0f);
    fill_f32(c.comb, (size_t)hc * hc, 0.25f);

    int reps = 0;
    const double sec = measure(hcpost_run, &c, 40, &reps);
    const double flops = 2.0 * (double)hc * (double)hidden * (double)(hc + 1);
    const double bytes = ((double)hc * hidden * 2 + hidden
                        + hc + hc * hc) * 4.0;
    report("hc_post", "hc4 hidden4096", "f32/f64", sec, reps, 1, flops, bytes);

    free(c.y); free(c.x); free(c.resid); free(c.post); free(c.comb);
}

typedef struct {
    float *pre, *post, *comb, *mixes, *scale, *base;
    int hc, iters;
} SinkCase;

static void sink_run(void *p)
{
    SinkCase *c = (SinkCase *)p;
    dsv4_hc_split_sinkhorn(c->pre, c->post, c->comb, c->mixes,
                           c->scale, c->base, c->hc, c->iters, 1e-6f);
}

static void bench_sinkhorn(void)
{
    if (skip("hc_sinkhorn")) return;

    const int hc = 4, mix = (2 + hc) * hc;
    SinkCase c;
    memset(&c, 0, sizeof c);
    c.hc = hc; c.iters = 20;
    c.pre   = (float *)xalloc((size_t)hc * sizeof(float));
    c.post  = (float *)xalloc((size_t)hc * sizeof(float));
    c.comb  = (float *)xalloc((size_t)hc * hc * sizeof(float));
    c.mixes = (float *)xalloc((size_t)mix * sizeof(float));
    c.scale = (float *)xalloc(3u * sizeof(float));
    c.base  = (float *)xalloc((size_t)mix * sizeof(float));
    fill_f32(c.mixes, (size_t)mix, 0.1f);
    fill_f32(c.base, (size_t)mix, 0.1f);
    c.scale[0] = c.scale[1] = c.scale[2] = 1.0f;

    int reps = 0;
    const double sec = measure(sink_run, &c, 40, &reps);
    report("hc_sinkhorn", "hc4 iters20", "f32/f64", sec, reps, 1, 0.0, 0.0);

    free(c.pre); free(c.post); free(c.comb); free(c.mixes);
    free(c.scale); free(c.base);
}

/* ---------------------------------------------------------------- RoPE ---- */

typedef struct {
    float *v, *cs, *sn;
    int hd, rd, h, pos, half;
} RopeCase;

static void rope_run(void *p)
{
    RopeCase *r = (RopeCase *)p;
    for (int i = 0; i < r->h; i++)
        dsv4_rope_apply(r->v + (size_t)i * (size_t)r->hd, r->hd, r->rd,
                        r->cs, r->sn, r->pos, r->half, 0);
}

static void bench_rope(void)
{
    if (skip("rope_q")) return;

    const int hd = 512, rd = 64, h = 64, seq = 2048, half = rd / 2;
    RopeCase c;
    memset(&c, 0, sizeof c);
    c.hd = hd; c.rd = rd; c.h = h; c.pos = 1000; c.half = half;
    c.v  = (float *)xalloc((size_t)h * hd * sizeof(float));
    c.cs = (float *)xalloc((size_t)seq * half * sizeof(float));
    c.sn = (float *)xalloc((size_t)seq * half * sizeof(float));
    fill_f32(c.v, (size_t)h * hd, 1.0f);
    dsv4_rope_table(c.cs, c.sn, rd, seq, 0, 10000.0, 16.0, 32.0, 1.0);

    int reps = 0;
    const double sec = measure(rope_run, &c, 40, &reps);
    const double flops = 3.0 * (double)h * (double)rd;
    const double bytes = 2.0 * (double)h * (double)rd * 4.0;
    report("rope_q", "64x512 rd64", "f32", sec, reps, 1, flops, bytes);

    free(c.v); free(c.cs); free(c.sn);
}

/* ------------------------------------------------------------- attention -- */

typedef struct {
    float *o, *q, *kv, *sink, *scratch;
    int   *idxs;
    int    h, d, n;
} AttnCase;

static void attn_run(void *p)
{
    AttnCase *a = (AttnCase *)p;
    dsv4_sparse_attn(a->o, a->q, a->kv, a->sink, a->idxs, a->h, a->d, a->n,
                     1.0f / sqrtf((float)a->d), a->scratch);
}

static void bench_attn(const char *name, int n)
{
    if (skip(name)) return;

    const int h = 64, d = 512;
    AttnCase a;
    memset(&a, 0, sizeof a);
    a.h = h; a.d = d; a.n = n;
    a.o       = (float *)xalloc((size_t)h * d * sizeof(float));
    a.q       = (float *)xalloc((size_t)h * d * sizeof(float));
    a.kv      = (float *)xalloc((size_t)n * d * sizeof(float));
    a.sink    = (float *)xalloc((size_t)h * sizeof(float));
    a.scratch = (float *)xalloc((size_t)h * n * sizeof(float));
    a.idxs    = (int *)xalloc((size_t)n * sizeof(int));
    fill_f32(a.q, (size_t)h * d, 1.0f);
    fill_f32(a.kv, (size_t)n * d, 1.0f);
    fill_f32(a.sink, (size_t)h, 1.0f);
    for (int i = 0; i < n; i++) a.idxs[i] = i;

    int reps = 0;
    const double sec = measure(attn_run, &a, 40, &reps);
    const double flops = 4.0 * (double)h * (double)n * (double)d;
    const double bytes = (double)h * (double)n * (double)d * 4.0;
    char shape[32];
    snprintf(shape, sizeof shape, "64x512 n=%d", n);
    report(name, shape, "f32/f64", sec, reps, 1, flops, bytes);

    free(a.o); free(a.q); free(a.kv); free(a.sink);
    free(a.scratch); free(a.idxs);
}

typedef struct {
    float *out, *q, *kv, *weights;
    int    nh, hd, n;
} IdxCase;

static void idx_run(void *p)
{
    IdxCase *c = (IdxCase *)p;
    dsv4_indexer_score(c->out, c->q, c->kv, c->weights, c->nh, c->hd, c->n);
}

static void bench_indexer(void)
{
    if (skip("indexer_score")) return;

    const int nh = 64, hd = 128, n = 2048;
    IdxCase c;
    memset(&c, 0, sizeof c);
    c.nh = nh; c.hd = hd; c.n = n;
    c.out     = (float *)xalloc((size_t)n * sizeof(float));
    c.q       = (float *)xalloc((size_t)nh * hd * sizeof(float));
    c.kv      = (float *)xalloc((size_t)n * hd * sizeof(float));
    c.weights = (float *)xalloc((size_t)nh * sizeof(float));
    fill_f32(c.q, (size_t)nh * hd, 1.0f);
    fill_f32(c.kv, (size_t)n * hd, 1.0f);
    fill_f32(c.weights, (size_t)nh, 1.0f);

    int reps = 0;
    const double sec = measure(idx_run, &c, 40, &reps);
    const double flops = 2.0 * (double)nh * (double)n * (double)hd;
    const double bytes = (double)n * hd * 4.0;
    report("indexer_score", "64x128 n=2048", "f32/f64", sec, reps, 1,
           flops, bytes);

    free(c.out); free(c.q); free(c.kv); free(c.weights);
}

typedef struct {
    int   *idx;
    float *scores;
    int    n, k;
} TopkCase;

static void topk_run(void *p)
{
    TopkCase *t = (TopkCase *)p;
    dsv4_topk(t->idx, t->scores, t->n, t->k);
}

static void bench_topk(void)
{
    if (skip("topk_2048")) return;

    TopkCase t;
    memset(&t, 0, sizeof t);
    t.n = 2048; t.k = 512;
    t.idx    = (int *)xalloc((size_t)t.k * sizeof(int));
    t.scores = (float *)xalloc((size_t)t.n * sizeof(float));
    fill_f32(t.scores, (size_t)t.n, 1.0f);

    int reps = 0;
    const double sec = measure(topk_run, &t, 40, &reps);
    report("topk_2048_512", "n=2048 k=512", "f32", sec, reps, 1, 0.0, 0.0);

    free(t.idx); free(t.scores);
}

/* ------------------------------------------------------------ compressor -- */

typedef struct {
    float *out, *kvin, *scin, *ape, *kvs, *scs;
    int    pos, ratio, hd;
} CompCase;

static void comp_run(void *p)
{
    CompCase *c = (CompCase *)p;
    dsv4_compress_step(c->out, c->kvin, c->scin, c->ape, c->kvs, c->scs,
                       c->pos, c->ratio, c->hd);
}

static void bench_compress(const char *name, int ratio)
{
    if (skip(name)) return;

    const int hd = 512;
    const int coff = (ratio == 4) ? 2 : 1;
    const int width = coff * hd;
    const int nslot = coff * ratio;

    CompCase c;
    memset(&c, 0, sizeof c);
    c.hd = hd; c.ratio = ratio; c.pos = ratio - 1;
    c.out  = (float *)xalloc((size_t)hd * sizeof(float));
    c.kvin = (float *)xalloc((size_t)width * sizeof(float));
    c.scin = (float *)xalloc((size_t)width * sizeof(float));
    c.ape  = (float *)xalloc((size_t)ratio * width * sizeof(float));
    c.kvs  = (float *)xalloc((size_t)nslot * width * sizeof(float));
    c.scs  = (float *)xalloc((size_t)nslot * width * sizeof(float));
    fill_f32(c.kvin, (size_t)width, 1.0f);
    fill_f32(c.scin, (size_t)width, 1.0f);
    fill_f32(c.ape, (size_t)ratio * width, 0.1f);
    fill_f32(c.kvs, (size_t)nslot * width, 1.0f);
    for (size_t i = 0; i < (size_t)nslot * width; i++) c.scs[i] = -INFINITY;

    int reps = 0;
    const double sec = measure(comp_run, &c, 40, &reps);
    const double flops = 3.0 * (double)nslot * (double)width;
    const double bytes = 2.0 * (double)nslot * (double)width * 4.0;
    char shape[32];
    snprintf(shape, sizeof shape, "hd512 ratio=%d", ratio);
    report(name, shape, "f32", sec, reps, 1, flops, bytes);

    free(c.out); free(c.kvin); free(c.scin); free(c.ape);
    free(c.kvs); free(c.scs);
}

/* ----------------------------------------------------------------- main --- */

static void usage(const char *argv0)
{
    fprintf(stderr, "usage: %s [--filter SUBSTR] [--threads N] "
                    "[--pin off|all|p|e]\n", argv0);
}

int main(int argc, char **argv)
{
    int pin = DSV4_PIN_P;
    int threads = 4;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--filter") && i + 1 < argc) {
            g_filter = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pin") && i + 1 < argc) {
            const int p = dsv4_threads_parse_pin(argv[++i]);
            if (p < 0) { usage(argv[0]); return 1; }
            pin = p;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    const DSV4ThreadOpts to = { pin, threads, 0, DSV4_PIN_OFF };
    dsv4_threads_apply(&to);

    printf("DeepSeek-V4 Flash microbenchmarks (compute only, weights resident)\n");
    printf("%-30s %-20s %-5s %10s %9s %9s\n",
           "kernel", "shape", "fmt", "ms/call", "GF/s", "GB/s");

    const double t0 = now_s();

    /* routed expert, decode */
    bench_qmat("expert_w1",   "2048x4096", DSV4_WFP4, 2048, 4096, 1, 1, 4, 40);
    bench_qmat("expert_w2",   "4096x2048", DSV4_WFP4, 4096, 2048, 1, 1, 4, 40);
    bench_expert_trio();

    /* attention projections, decode */
    bench_qmat("wq_a",        "1024x4096", DSV4_WFP8, 1024, 4096, 1, 1, 4, 40);
    bench_qmat("wq_b",        "32768x1024", DSV4_WFP8, 32768, 1024, 1, 1, 2, 40);
    bench_qmat("wkv",         "512x4096",  DSV4_WFP8, 512, 4096, 1, 1, 8, 40);
    bench_qmat("wo_a_group",  "1024x4096", DSV4_WFP8, 1024, 4096, 1, 1, 4, 40);
    bench_qmat("wo_a_8groups", "8x1024x4096", DSV4_WFP8, 1024, 4096, 1, 8, 1, 40);
    bench_qmat("wo_b",        "4096x8192", DSV4_WFP8, 4096, 8192, 1, 1, 2, 40);

    /* shared expert, decode */
    bench_qmat("shared_gate_up", "2048x4096", DSV4_WFP8, 2048, 4096, 1, 1, 2, 40);
    bench_qmat("shared_down", "4096x2048", DSV4_WFP8, 4096, 2048, 1, 1, 2, 40);

    /* indexer projections + router + head */
    bench_qmat("indexer_wq_b", "8192x1024", DSV4_WFP8, 8192, 1024, 1, 1, 2, 40);
    bench_qmat("indexer_weights_proj", "64x4096", DSV4_WBF16, 64, 4096, 1, 1, 8, 40);
    bench_qmat("gate",        "256x4096", DSV4_WBF16, 256, 4096, 1, 1, 8, 40);
    bench_qmat("head",        "129280x4096", DSV4_WBF16, 129280, 4096, 1, 1, 1, 40);

    /* prefill GEMM */
    bench_qmat("expert_w1_nt16", "2048x4096 nt16", DSV4_WFP4, 2048, 4096, 16, 1, 4, 40);
    bench_qmat("expert_w1_nt128", "2048x4096 nt128", DSV4_WFP4, 2048, 4096, 128, 1, 4, 40);
    bench_qmat("wq_b_nt16",   "32768x1024 nt16", DSV4_WFP8, 32768, 1024, 16, 1, 2, 40);
    bench_qmat("wo_a_nt16",   "1024x4096 nt16", DSV4_WFP8, 1024, 4096, 16, 1, 4, 40);

    /* attention / indexer / compressor */
    bench_attn("sparse_attn_idx", 640);
    bench_attn("sparse_attn_win", 128);
    bench_indexer();
    bench_topk();
    bench_compress("compress_r4", 4);
    bench_compress("compress_r128", 128);

    /* elementwise */
    bench_simple("rmsnorm_4096", "n=4096", 4096, rms_run, 0);
    bench_simple("rmsnorm_1024", "n=1024", 1024, rms_run, 0);
    bench_simple("swiglu_2048", "n=2048", 2048, swi_run, 1);
    bench_route();
    bench_hc_pre();
    bench_hc_post();
    bench_sinkhorn();
    bench_rope();

    printf("total: %.3f s\n", now_s() - t0);
    return 0;
}

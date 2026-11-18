/* SPDX-License-Identifier: Apache-2.0 */
/* fixtures.c - deterministic, realistic inputs for the microbenchmarks. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bench.h"

#define PAT_BYTES 65536

static uint64_t g_rng = 0x243F6A8885A308D3ULL;

void bench_die(const char *what)
{
    fprintf(stderr, "bench: %s\n", what);
    exit(1);
}

double bench_now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

uint32_t bench_rnd32(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 32);
}

float bench_frnd(void)
{
    return (float)(bench_rnd32() >> 8) * (1.0f / 16777216.0f);
}

void *bench_xalloc(size_t n)
{
    void *p = NULL;
    if (n == 0) n = 64;
    if (posix_memalign(&p, 64, n) != 0 || !p) bench_die("allocation failed");
    return p;
}

static void *xalloc_align(size_t n, size_t align)
{
    void *p = NULL;
    if (align < sizeof(void *)) align = sizeof(void *);
    if (n == 0) n = 64;
    if (posix_memalign(&p, align, n) != 0 || !p) bench_die("allocation failed");
    return p;
}

static size_t round_up(size_t n, size_t align)
{
    return (n + align - 1) & ~(align - 1);
}

/* ------------------------------------------------------------- timing ------ */

double bench_measure(RunFn fn, void *ctx, int target_ms, int *out_reps)
{
    fn(ctx);
    const double t0 = bench_now_s();
    int reps = 0;
    double elapsed;
    do {
        fn(ctx);
        reps++;
        elapsed = bench_now_s() - t0;
    } while ((elapsed * 1000.0 < (double)target_ms || reps < 3)
             && reps < 200000);
    *out_reps = reps;
    return elapsed;
}

void bench_report(const char *name, const char *shape, const char *fmt,
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

void bench_fill_f32(float *p, size_t n, float amp)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (bench_frnd() * 2.0f - 1.0f) * amp;
}

/* ------------------------------------------------------------- fixtures --- */

static void fill_bf16_pattern(uint8_t *pat)
{
    uint16_t *h = (uint16_t *)pat;
    for (int i = 0; i < PAT_BYTES / 2; i++) {
        uint16_t v = (uint16_t)(0x3F00u + (bench_rnd32() % 0x0200u)); /* 0.5 .. 8 */
        if (bench_rnd32() & 1u) v |= 0x8000u;
        h[i] = v;
    }
}

static void fill_fp8_pattern(uint8_t *pat)
{
    for (int i = 0; i < PAT_BYTES; i++) {
        /* A sparse subnormal rate matching a sampled trunk tensor, so only a
         * small fraction of 128-byte blocks take the exact slow path. */
        if ((bench_rnd32() % 6800u) == 0u) {
            pat[i] = 0x01u;
            continue;
        }
        const uint32_t sign = bench_rnd32() & 1u;
        uint32_t exp = 1u + (bench_rnd32() % 15u);
        uint32_t man = bench_rnd32() & 7u;
        if (exp == 15u && man == 7u) man = 6u;      /* 0x7F is NaN */
        pat[i] = (uint8_t)((sign << 7) | (exp << 3) | man);
    }
}

static uint8_t fp4_code(void)
{
    static const int cum[8] = { 8, 32, 58, 78, 90, 96, 99, 100 };
    const int r = (int)(bench_rnd32() % 100u);
    int mag = 0;
    while (mag < 7 && r >= cum[mag]) mag++;
    uint8_t c = (uint8_t)mag;
    if (bench_rnd32() & 1u) c |= 8u;
    return c;
}

static void fill_fp4_pattern(uint8_t *pat)
{
    for (int i = 0; i < PAT_BYTES; i++)
        pat[i] = (uint8_t)(fp4_code() | (fp4_code() << 4));
}

static void fill_bytes(void *dst, size_t n, int wdt)
{
    uint8_t *pat = (uint8_t *)bench_xalloc(PAT_BYTES);
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
        p[i] = (uint8_t)(lo + (bench_rnd32() % 7u));
}

void qfix_init_aligned(QFix *q, int wdt, int rows, int cols, size_t align)
{
    memset(q, 0, sizeof *q);
    if (align < 64) align = 64;
    q->m.wdt  = wdt;
    q->m.rows = rows;
    q->m.cols = cols;

    if (wdt == DSV4_WBF16) {
        q->wb = (size_t)rows * (size_t)cols * 2u;
        q->walloc = round_up(q->wb, align);
        q->w  = xalloc_align(q->walloc, align);
        fill_bytes(q->w, q->wb, wdt);
        q->m.blk_r = q->m.blk_c = 0;
    } else if (wdt == DSV4_WFP8) {
        q->wb = (size_t)rows * (size_t)cols;
        q->walloc = round_up(q->wb, align);
        q->w  = xalloc_align(q->walloc, align);
        fill_bytes(q->w, q->wb, wdt);
        const int sr = (rows + 127) / 128;
        const int sc = (cols + 127) / 128;
        q->sb = (size_t)sr * (size_t)sc;
        q->salloc = round_up(q->sb, align);
        q->s  = xalloc_align(q->salloc, align);
        fill_scales((uint8_t *)q->s, q->sb, 0);
        q->m.blk_r = 128;
        q->m.blk_c = 128;
    } else {
        q->wb = (size_t)rows * ((size_t)cols / 2u);
        q->walloc = round_up(q->wb, align);
        q->w  = xalloc_align(q->walloc, align);
        fill_bytes(q->w, q->wb, wdt);
        const int sr = rows;
        const int sc = (cols + 31) / 32;
        q->sb = (size_t)sr * (size_t)sc;
        q->salloc = round_up(q->sb, align);
        q->s  = xalloc_align(q->salloc, align);
        fill_scales((uint8_t *)q->s, q->sb, 1);
        q->m.blk_r = 1;
        q->m.blk_c = 32;
    }
    q->m.w = q->w;
    q->m.s = q->s;
}

void qfix_init(QFix *q, int wdt, int rows, int cols)
{
    qfix_init_aligned(q, wdt, rows, cols, 64);
}

void qfix_free(QFix *q)
{
    free(q->w);
    free(q->s);
    memset(q, 0, sizeof *q);
}

double qfix_bytes(const QFix *q, int nt)
{
    return (double)q->wb + (double)q->sb
         + (double)nt * (double)q->m.cols * 4.0
         + (double)nt * (double)q->m.rows * 4.0;
}

const char *bench_fmt_name(int wdt)
{
    switch (wdt) {
    case DSV4_WBF16: return "bf16";
    case DSV4_WFP8:  return "fp8";
    case DSV4_WFP4:  return "fp4";
    default:         return "f32";
    }
}

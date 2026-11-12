/* SPDX-License-Identifier: Apache-2.0 */
/* bench/fp8_decode.c - three ways to turn packed E4M3 bytes into the doubles
 * the GEMV consumes, timed on Flash's real FP8 geometries.
 *
 *   bit      the current engine path: dsv4_e4m3_to_f32() per weight, four
 *            scalar conversions + _mm_set_ps per 4 weights. GCC inlines the
 *            conversion into ~20 ALU ops per weight.
 *   lut4     the same loop, but the decode is a 256-entry L1 table lookup.
 *   gather8  one vpgatherdd per 8 weights from the same table, then two
 *            cvtps2pd and two FMAs.
 *
 * All three use the SAME double accumulation tree as dsv4_matmul.c
 * (v0..v3 over columns i..i+15, reduced as (v0+v1)+(v2+v3)), so a faster
 * variant must still be bitwise identical to the scalar reference. This tool
 * checks that rather than assuming it, and prints GF/s next to the answer.
 *
 * This does NOT change the engine. It is the measurement that decides whether
 * the LUT/gather path is worth wiring into dsv4_matmul.c.
 *
 * usage: fp8_decode [reps] [--only bit|lut4|gather8]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4.h"
#include "dsv4_quant.h"

void dsv4_matmul_fp8(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_matmul_fp8_scalar(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static unsigned rng = 987654321u;
static unsigned nextr(void) { rng = rng * 1103515245u + 12345u; return rng; }

/* ------------------------------------------------------------------ table ---
 * Built from the scalar definition itself, so it cannot disagree with it. The
 * all-256 check below then verifies the gather against the same values. */
static uint32_t e4m3_bits[256];
static float    e4m3_f[256];

static void lut_init(void)
{
    for (int b = 0; b < 256; b++) {
        float f = dsv4_e4m3_to_f32((uint8_t)b);
        e4m3_f[b] = f;
        memcpy(&e4m3_bits[b], &f, sizeof f);
    }
}

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

static inline __m256 load8_gather(const uint8_t *wb, int i)
{
    __m128i b8 = _mm_loadl_epi64((const __m128i *)(wb + i));
    __m256i idx = _mm256_cvtepu8_epi32(b8);
    __m256i bits = _mm256_i32gather_epi32((const int *)e4m3_bits, idx, 4);
    return _mm256_castsi256_ps(bits);
}

static inline __m256d load4_lut(const uint8_t *wb, int i)
{
    return _mm256_cvtps_pd(_mm_set_ps(e4m3_f[wb[i + 3]], e4m3_f[wb[i + 2]],
                                      e4m3_f[wb[i + 1]], e4m3_f[wb[i + 0]]));
}

static inline __m256d load4_bit(const uint8_t *wb, int i)
{
    return _mm256_cvtps_pd(_mm_set_ps(dsv4_e4m3_to_f32(wb[i + 3]),
                                      dsv4_e4m3_to_f32(wb[i + 2]),
                                      dsv4_e4m3_to_f32(wb[i + 1]),
                                      dsv4_e4m3_to_f32(wb[i + 0])));
}
#endif

#define XB4(i) _mm256_loadu_pd(xdb + (i))

/* --------------------------------------------------------------- kernels --- */
/* One GEMV body per variant. The outer loop is copied from GEN_MATMULS in
 * dsv4_matmul.c so the only difference measured is the decode. */

#define GEMV_LOOP(NAME, DOT)                                                    \
static void NAME(float *y, const float *x, const uint8_t *W, const uint8_t *S,  \
                 int in, int out, int blk_r, int blk_c)                         \
{                                                                               \
    double *xd = malloc((size_t)in * sizeof *xd);                               \
    if (!xd) return;                                                            \
    for (int i = 0; i < in; i++) xd[i] = (double)x[i];                          \
    const int sc = (in + blk_c - 1) / blk_c;                                    \
    _Pragma("omp parallel for schedule(dynamic, 4) if (out > 64)")              \
    for (int o = 0; o < out; o++) {                                            \
        const uint8_t *rw = W + (size_t)o * in;                                 \
        double total = 0.0;                                                    \
        for (int c0 = 0; c0 < in; c0 += blk_c) {                               \
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;               \
            const uint8_t *wb = rw + c0;                                       \
            const double *xdb = xd + c0;                                       \
            double acc;                                                        \
            DOT();                                                             \
            total += acc * (double)dsv4_e8m0_to_f32(                           \
                        S[(o / blk_r) * (size_t)sc + (c0 / blk_c)]);           \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
    free(xd);                                                                  \
}

/* The current engine path. */
#if defined(__AVX2__) && defined(__FMA__)
#define DOT_BIT()                                                              \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            v0 = _mm256_fmadd_pd(load4_bit(wb, i),      XB4(i),      v0);      \
            v1 = _mm256_fmadd_pd(load4_bit(wb, i + 4),  XB4(i + 4),  v1);      \
            v2 = _mm256_fmadd_pd(load4_bit(wb, i + 8),  XB4(i + 8),  v2);      \
            v3 = _mm256_fmadd_pd(load4_bit(wb, i + 12), XB4(i + 12), v3);      \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma((double)dsv4_e4m3_to_f32(wb[i]),          \
                                     xdb[i], acc);                             \
    } while (0)

#define DOT_LUT()                                                              \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            v0 = _mm256_fmadd_pd(load4_lut(wb, i),      XB4(i),      v0);      \
            v1 = _mm256_fmadd_pd(load4_lut(wb, i + 4),  XB4(i + 4),  v1);      \
            v2 = _mm256_fmadd_pd(load4_lut(wb, i + 8),  XB4(i + 8),  v2);      \
            v3 = _mm256_fmadd_pd(load4_lut(wb, i + 12), XB4(i + 12), v3);      \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) {                                                   \
            acc = fma((double)e4m3_f[wb[i]], xdb[i], acc);                     \
        }                                                                      \
    } while (0)

#define DOT_GATHER()                                                           \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            const __m256 f0 = load8_gather(wb, i);                             \
            const __m256 f1 = load8_gather(wb, i + 8);                         \
            v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(                              \
                     _mm256_castps256_ps128(f0)), XB4(i),      v0);            \
            v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(                              \
                     _mm256_extractf128_ps(f0, 1)), XB4(i + 4),  v1);          \
            v2 = _mm256_fmadd_pd(_mm256_cvtps_pd(                              \
                     _mm256_castps256_ps128(f1)), XB4(i + 8),  v2);            \
            v3 = _mm256_fmadd_pd(_mm256_cvtps_pd(                              \
                     _mm256_extractf128_ps(f1, 1)), XB4(i + 12), v3);          \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma((double)dsv4_e4m3_to_f32(wb[i]),          \
                                     xdb[i], acc);                             \
    } while (0)

GEMV_LOOP(gemv_bit,    DOT_BIT)
GEMV_LOOP(gemv_lut,    DOT_LUT)
GEMV_LOOP(gemv_gather, DOT_GATHER)
#else
static void gemv_bit(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int br, int bc)
{ (void)y;(void)x;(void)W;(void)S;(void)in;(void)out;(void)br;(void)bc; }
static void gemv_lut(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int br, int bc)
{ (void)y;(void)x;(void)W;(void)S;(void)in;(void)out;(void)br;(void)bc; }
static void gemv_gather(float *y, const float *x, const uint8_t *W,
                        const uint8_t *S, int in, int out, int br, int bc)
{ (void)y;(void)x;(void)W;(void)S;(void)in;(void)out;(void)br;(void)bc; }
#endif

/* ------------------------------------------------------------ verification -- */
static int check_lut(void)
{
    int bad = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t want;
        float f = dsv4_e4m3_to_f32((uint8_t)b);
        memcpy(&want, &f, 4);
        if (e4m3_bits[b] != want) {
            printf("  FAIL  lut[0x%02X] = 0x%08X, function = 0x%08X\n",
                   b, e4m3_bits[b], want);
            bad++;
        }
    }
#if defined(__AVX2__) && defined(__FMA__)
    /* Gather all 256 values, 8 at a time, and compare bit patterns. */
    uint8_t bytes[256];
    for (int b = 0; b < 256; b++) bytes[b] = (uint8_t)b;
    for (int i = 0; i < 256; i += 8) {
        __m256 f = load8_gather(bytes, i);
        float got[8];
        _mm256_storeu_ps(got, f);
        for (int j = 0; j < 8; j++) {
            uint32_t a, e;
            memcpy(&a, &got[j], 4);
            memcpy(&e, &e4m3_bits[bytes[i + j]], 4);
            if (a != e) {
                printf("  FAIL  gather[0x%02X] = 0x%08X, lut = 0x%08X\n",
                       bytes[i + j], a, e);
                bad++;
            }
        }
    }
#endif
    return bad;
}

int main(int argc, char **argv)
{
    int reps = 10;
    const char *only = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else reps = atoi(argv[i]);
    }

    lut_init();
    printf("fp8 decode microbenchmark (E4M3 -> f64 GEMV)\n");
#if defined(__AVX2__) && defined(__FMA__)
    printf("AVX2+FMA: yes   gather variant: compiled in\n");
#else
    printf("AVX2+FMA: no    this benchmark needs it; stopping\n");
    return 1;
#endif
    int bad = check_lut();
    printf("all-256 decode check: %s\n\n", bad ? "FAILED" : "bit-identical");

    struct { const char *name; int in, out, br, bc; } job[] = {
        { "fp8 attn wq_a 1024x4096",  4096,  1024, 128, 128 },
        { "fp8 shared w1 2048x4096",  4096,  2048, 128, 128 },
        { "fp8 attn wo_b 4096x8192",  8192,  4096, 128, 128 },
        { "fp8 attn wq_b 32768x1024", 1024, 32768, 128, 128 },
    };
    struct { const char *name; void (*fn)(float *, const float *, const uint8_t *,
                                          const uint8_t *, int, int, int, int); } var[] = {
        { "bit",     gemv_bit    },
        { "lut4",    gemv_lut    },
        { "gather8", gemv_gather },
    };

    for (unsigned j = 0; j < sizeof job / sizeof job[0]; j++) {
        const int in = job[j].in, out = job[j].out;
        const int sc = (in + job[j].bc - 1) / job[j].bc;
        const size_t sbytes = (size_t)((out + job[j].br - 1) / job[j].br) * sc;

        float *x  = malloc((size_t)in * sizeof *x);
        float *yr = malloc((size_t)out * sizeof *yr);
        float *y  = malloc((size_t)out * sizeof *y);
        uint8_t *W = malloc((size_t)out * in);
        uint8_t *S = malloc(sbytes);
        if (!x || !yr || !y || !W || !S) { printf("out of memory\n"); return 1; }

        for (int i = 0; i < in; i++)
            x[i] = (float)((int)(nextr() >> 16 & 0xffff) - 32768) * 1e-3f;
        for (size_t i = 0; i < (size_t)out * in; i++) {
            uint8_t b = (uint8_t)(nextr() >> 16);
            /* exclude NaN encodings: bit comparison, not arithmetic */
            if ((b & 0x7Fu) == 0x7Fu) b &= 0xFEu;
            W[i] = b;
        }
        memset(S, 127, sbytes);

        /* Reference: the engine's scalar path. */
        dsv4_matmul_fp8_scalar(yr, x, W, S, in, out, job[j].br, job[j].bc);

        printf("%s\n", job[j].name);
        printf("  %-8s %9s %9s %9s %8s\n",
               "variant", "ms", "GF/s", "GB/s", "bitwise");
        double tbit = 0.0, tgather = 0.0, tlut = 0.0;
        for (unsigned v = 0; v < sizeof var / sizeof var[0]; v++) {
            if (only && strcmp(only, var[v].name)) continue;
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                double t0 = now();
                var[v].fn(y, x, W, S, in, out, job[j].br, job[j].bc);
                double t1 = now();
                if (t1 - t0 < best) best = t1 - t0;
            }
            int same = memcmp(yr, y, (size_t)out * sizeof *yr) == 0;
            double flop = 2.0 * (double)in * (double)out;
            double bytes = (double)in * (double)out;
            printf("  %-8s %9.3f %9.1f %9.1f %8s\n",
                   var[v].name, best * 1e3, flop / best * 1e-9,
                   bytes / best * 1e-9, same ? "yes" : "NO");
            if (!strcmp(var[v].name, "bit"))    tbit = best;
            if (!strcmp(var[v].name, "lut4"))   tlut = best;
            if (!strcmp(var[v].name, "gather8")) tgather = best;
        }
        if (tbit > 0.0) {
            if (tlut > 0.0)    printf("  lut4    vs bit: %.2fx\n", tbit / tlut);
            if (tgather > 0.0) printf("  gather8 vs bit: %.2fx\n", tbit / tgather);
        }
        free(x); free(yr); free(y); free(W); free(S);
    }
    return bad ? 1 : 0;
}

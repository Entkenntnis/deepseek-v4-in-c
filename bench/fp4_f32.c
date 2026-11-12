/* SPDX-License-Identifier: Apache-2.0 */
/* bench/fp4_f32.c - FP32 inner accumulation on this CPU.
 *
 * Two independent questions:
 *
 *   1. What is the raw FP32 and FP64 FMA ceiling of this machine? That is a
 *      register-only loop with eight independent accumulators, so it measures
 *      execution ports and clock, not memory or dequant.
 *
 *   2. Does the production FP4 kernel hit its error bound and what throughput
 *      does it reach? dsv4_matmul.c accumulates one 32-weight scale block in
 *      float and folds blocks in double. This benchmark runs the shipped float
 *      kernel at the two real expert geometries and reports its error against
 *      the scalar double reference.
 *
 * The FP4 kernel is deliberately not bitwise-equal to the double reference;
 * test_matmul and this binary bound the difference in L2 norm instead.
 */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dsv4.h"

void dsv4_matmul_fp4(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_matmul_fp4_scalar(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
int  dsv4_matmul_has_avx2(void);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int nthreads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Keep the FMA loops observable; otherwise the compiler is entitled to delete
 * the entire register-only recurrence. */
static volatile double fma_sink;

static float hsum_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t);
}

static double hsum_pd(__m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d t = _mm_add_pd(lo, hi);
    t = _mm_hadd_pd(t, t);
    return _mm_cvtsd_f64(t);
}

/* Eight independent chains at both widths. Eight is enough to hide the
 * four-cycle FMA latency even if the core can issue two FMAs per cycle. */
static double peak_f32(long iters)
{
    double sum = 0.0, t0 = now();
#ifdef _OPENMP
#pragma omp parallel reduction(+:sum)
#endif
    {
        __m256 a0 = _mm256_set1_ps(1.0f), a1 = _mm256_set1_ps(1.0f);
        __m256 a2 = _mm256_set1_ps(1.0f), a3 = _mm256_set1_ps(1.0f);
        __m256 a4 = _mm256_set1_ps(1.0f), a5 = _mm256_set1_ps(1.0f);
        __m256 a6 = _mm256_set1_ps(1.0f), a7 = _mm256_set1_ps(1.0f);
        const __m256 b = _mm256_set1_ps(1.0000001f);
        const __m256 c = _mm256_set1_ps(1e-7f);
        for (long i = 0; i < iters; i++) {
            a0 = _mm256_fmadd_ps(b, c, a0);
            a1 = _mm256_fmadd_ps(b, c, a1);
            a2 = _mm256_fmadd_ps(b, c, a2);
            a3 = _mm256_fmadd_ps(b, c, a3);
            a4 = _mm256_fmadd_ps(b, c, a4);
            a5 = _mm256_fmadd_ps(b, c, a5);
            a6 = _mm256_fmadd_ps(b, c, a6);
            a7 = _mm256_fmadd_ps(b, c, a7);
        }
        sum += hsum_ps(_mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)),
            _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7))));
    }
    fma_sink = sum;
    double dt = now() - t0;
    double flops = (double)nthreads() * (double)iters * 8.0 * 8.0 * 2.0;
    return flops / dt * 1e-9;
}

static double peak_f64(long iters)
{
    double sum = 0.0, t0 = now();
#ifdef _OPENMP
#pragma omp parallel reduction(+:sum)
#endif
    {
        __m256d a0 = _mm256_set1_pd(1.0), a1 = _mm256_set1_pd(1.0);
        __m256d a2 = _mm256_set1_pd(1.0), a3 = _mm256_set1_pd(1.0);
        __m256d a4 = _mm256_set1_pd(1.0), a5 = _mm256_set1_pd(1.0);
        __m256d a6 = _mm256_set1_pd(1.0), a7 = _mm256_set1_pd(1.0);
        const __m256d b = _mm256_set1_pd(1.0000001);
        const __m256d c = _mm256_set1_pd(1e-7);
        for (long i = 0; i < iters; i++) {
            a0 = _mm256_fmadd_pd(b, c, a0);
            a1 = _mm256_fmadd_pd(b, c, a1);
            a2 = _mm256_fmadd_pd(b, c, a2);
            a3 = _mm256_fmadd_pd(b, c, a3);
            a4 = _mm256_fmadd_pd(b, c, a4);
            a5 = _mm256_fmadd_pd(b, c, a5);
            a6 = _mm256_fmadd_pd(b, c, a6);
            a7 = _mm256_fmadd_pd(b, c, a7);
        }
        sum += hsum_pd(_mm256_add_pd(
            _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)),
            _mm256_add_pd(_mm256_add_pd(a4, a5), _mm256_add_pd(a6, a7))));
    }
    fma_sink = sum;
    double dt = now() - t0;
    double flops = (double)nthreads() * (double)iters * 8.0 * 4.0 * 2.0;
    return flops / dt * 1e-9;
}

static unsigned rng = 987654321u;
static unsigned nextr(void) { rng = rng * 1103515245u + 12345u; return rng; }

#if defined(__AVX2__) && defined(__FMA__)
static void run_shape(const char *name, int in, int out, int reps)
{
    const int blk_r = 1, blk_c = 32;
    const int sc = (in + blk_c - 1) / blk_c;
    const size_t wbytes = (size_t)out * (size_t)in / 2;
    const size_t sbytes = (size_t)((out + blk_r - 1) / blk_r) * (size_t)sc;

    float *x = malloc((size_t)in * sizeof *x);
    float *yr = malloc((size_t)out * sizeof *yr);
    float *yf = malloc((size_t)out * sizeof *yf);
    uint8_t *W = malloc(wbytes), *S = malloc(sbytes);
    if (!x || !yr || !yf || !W || !S) {
        printf("%s: out of memory\n", name);
        exit(1);
    }

    for (int i = 0; i < in; i++)
        x[i] = (float)((int)(nextr() >> 16 & 0xffff) - 32768) * 3e-5f;
    for (size_t i = 0; i < wbytes; i++) W[i] = (uint8_t)(nextr() >> 16);
    for (size_t i = 0; i < sbytes; i++)
        S[i] = (uint8_t)(124 + (nextr() >> 16) % 7);   /* 2^-3 .. 2^3 */

    dsv4_matmul_fp4_scalar(yr, x, W, S, in, out, blk_r, blk_c);
    dsv4_matmul_fp4(yf, x, W, S, in, out, blk_r, blk_c);

    double max_abs = 0.0, max_rel = 0.0, nrm2_err = 0.0, nrm2_ref = 0.0;
    for (int i = 0; i < out; i++) {
        const double d = fabs((double)yf[i] - (double)yr[i]);
        const double r = fabs((double)yr[i]);
        if (d > max_abs) max_abs = d;
        if (r > 1e-30 && d / r > max_rel) max_rel = d / r;
        nrm2_err += d * d;
        nrm2_ref += r * r;
    }
    const double nrm_rel = nrm2_ref > 0.0 ? sqrt(nrm2_err / nrm2_ref) : 0.0;

    printf("  timing %s (%d reps)...\n", name, reps);
    fflush(stdout);
    double tf = 1e30;
    for (int r = 0; r < reps; r++) {
        const double t0 = now();
        dsv4_matmul_fp4(yf, x, W, S, in, out, blk_r, blk_c);
        const double t1 = now();
        if (t1 - t0 < tf) tf = t1 - t0;
    }

    const double flop = 2.0 * (double)in * (double)out;
    printf("%-22s  f32 %7.1f GF   error vs double: "
           "max abs %.3e   max rel %.3e   L2 rel %.3e\n",
           name, flop / tf * 1e-9, max_abs, max_rel, nrm_rel);

    free(x); free(yr); free(yf); free(W); free(S);
}
#endif

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 10;
    const long iters = 20000000L;

    printf("measuring raw FP32/FP64 FMA ceilings...\n");
    fflush(stdout);
    double p32 = 0.0, p64 = 0.0;
    for (int r = 0; r < 3; r++) {
        const double a = peak_f32(iters), b = peak_f64(iters);
        if (a > p32) p32 = a;
        if (b > p64) p64 = b;
    }
    printf("FP32 vs FP64 on this CPU, %d thread(s)\n", nthreads());
    printf("  raw FMA ceiling:  fp32 %7.1f GF/s   fp64 %7.1f GF/s   "
           "ratio %.2fx\n", p32, p64, p32 / p64);

#if defined(__AVX2__) && defined(__FMA__)
    if (!dsv4_matmul_has_avx2()) {
        printf("  AVX2 not available at runtime; cannot run the kernel test\n");
        return 0;
    }
    printf("\nFP4 expert matmul, production float-inner kernel\n");
    run_shape("fp4 expert w1 2048x4096", 4096, 2048, reps);
    run_shape("fp4 expert w2 4096x2048", 2048, 4096, reps);
#else
    printf("  AVX2+FMA not compiled in; float kernel not available\n");
#endif
    return 0;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_dot.h - the f32-input / double-accumulate dot product shared by the
 * kernels that are not the quantised GEMV.
 *
 * These loops used to be a single `acc = fma(a[i], b[i], acc)` chain. On the
 * P-cores that is a four-cycle dependency per element: the FPU is idle most of
 * the time even though the arithmetic looks free. Four independent 4-lane
 * double chains, folded in the same (a0+a1)+(a2+a3) tree the matmul uses, turn
 * that into throughput without dropping to f32 accuracy.
 *
 * The scalar path is the definition; the AVX2 path keeps f32 inputs exact
 * (float -> double is lossless) and only changes the order in which the double
 * partial sums are combined. */
#ifndef DSV4_DOT_H
#define DSV4_DOT_H

#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

static inline double dsv4_dot_f32d(const float *a, const float *b, int n)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
    __m256d a2 = _mm256_setzero_pd(), a3 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        a0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i)),
                             _mm256_cvtps_pd(_mm_loadu_ps(b + i)), a0);
        a1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 4)),
                             _mm256_cvtps_pd(_mm_loadu_ps(b + i + 4)), a1);
        a2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 8)),
                             _mm256_cvtps_pd(_mm_loadu_ps(b + i + 8)), a2);
        a3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 12)),
                             _mm256_cvtps_pd(_mm_loadu_ps(b + i + 12)), a3);
    }
    double bb[4];
    _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(a0, a1),
                                       _mm256_add_pd(a2, a3)));
    double acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);
    for (; i < n; i++)
        acc = fma((double)a[i], (double)b[i], acc);
    return acc;
#else
    double acc = 0.0;
    for (int i = 0; i < n; i++)
        acc = fma((double)a[i], (double)b[i], acc);
    return acc;
#endif
}

#endif /* DSV4_DOT_H */

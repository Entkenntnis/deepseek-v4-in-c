/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_dec_reduce.h - the four-lane double reduction tree.
 *
 * The 16-accumulator tree K3 uses, and the four-lane form that reproduces it
 * exactly. Put a[0..3] in v0, a[4..7] in v1, a[8..11] in v2, a[12..15] in v3,
 * and lane j of each vector holds a[j], a[4+j], a[8+j], a[12+j]. Then
 *
 *     b[j] = (a[j] + a[4+j]) + (a[8+j] + a[12+j])
 *
 * is exactly (v0 + v1) + (v2 + v3), elementwise, with the same grouping. The
 * final (b0 + b1) + (b2 + b3) is then done on the four extracted lanes in the
 * scalar order. _mm256_fmadd_pd is the same IEEE fused multiply-add as fma(),
 * and float -> double is exact, so every intermediate is bit-for-bit what the
 * scalar path computes. That is a claim the matmul gate checks at runtime
 * rather than a claim this comment makes.
 *
 * LOAD4(i) must return the four weights at logical columns i..i+3 as doubles. */
#ifndef DSV4_DEC_REDUCE_H
#define DSV4_DEC_REDUCE_H

#include <math.h>

#include <immintrin.h>

#define XB4(i) _mm256_loadu_pd(xdb + (i))
#define BLOCK_DOT_V(LOAD1, LOAD4)                                              \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            v0 = _mm256_fmadd_pd(LOAD4(i),      XB4(i),      v0);              \
            v1 = _mm256_fmadd_pd(LOAD4(i + 4),  XB4(i + 4),  v1);              \
            v2 = _mm256_fmadd_pd(LOAD4(i + 8),  XB4(i + 8),  v2);              \
            v3 = _mm256_fmadd_pd(LOAD4(i + 12), XB4(i + 12), v3);              \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma(LOAD1(i), (double)xb[i], acc);            \
    } while (0)

#endif /* DSV4_DEC_REDUCE_H */

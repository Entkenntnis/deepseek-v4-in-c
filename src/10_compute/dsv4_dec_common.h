/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_dec_common.h - horizontal reduction shared by the float-inner decoders. */
#ifndef DSV4_DEC_COMMON_H
#define DSV4_DEC_COMMON_H

#include <stdint.h>
#include <math.h>

#include <immintrin.h>

/* The horizontal sum used by the float-inner kernels. Returned as float; every
 * caller that folds it in double widens it, which is exact. */
static inline float dsv4_hsum256(__m256 s)
{
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t);
}


static inline double bf16d(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return (double)v.f;
}

#define LOAD1_BF16(k) bf16d(row[k])

/* BF16 widens without a table: the stored halfword IS the top half of the f32,
 * so shifting it up 16 bits and reinterpreting is the whole conversion. */
static inline __m256d load4_bf16(const uint16_t *row, int i)
{
    const __m128i h = _mm_loadl_epi64((const __m128i *)(row + i));
    const __m128i w = _mm_slli_epi32(_mm_cvtepu16_epi32(h), 16);
    return _mm256_cvtps_pd(_mm_castsi128_ps(w));
}

#define LOAD4_BF16(k) load4_bf16(row, (k))

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

#endif /* DSV4_DEC_COMMON_H */

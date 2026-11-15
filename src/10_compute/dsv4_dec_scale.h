/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_dec_scale.h - float E8M0 scale vectors and the 4-block fold.
 *
 * The quantised matmul bodies accumulate already-formed block sums across K.
 * Those block sums are float; the cross-block sum is float too, with four
 * independent partial accumulators and a pairwise final tree. This keeps the
 * rounding error below the weight quantisation error without a double lane.
 */
#ifndef DSV4_DEC_SCALE_H
#define DSV4_DEC_SCALE_H

#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "dsv4_quant.h"

/* Four E8M0 bytes -> four f32 scales. Exact for normal exponents; 0x00 and
 * 0xFF take the scalar definition (2^-127 subnormal and NaN respectively). */
static inline __m128 dsv4_e8m0x4_to_ps(const uint8_t *p)
{
    uint32_t b;
    memcpy(&b, p, 4);
    __m128i v = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)b));
    const __m128i bad = _mm_or_si128(
        _mm_cmpeq_epi32(v, _mm_setzero_si128()),
        _mm_cmpeq_epi32(v, _mm_set1_epi32(0xFF)));
    if (_mm_movemask_epi8(bad)) {
        float s[4];
        for (int i = 0; i < 4; i++) s[i] = dsv4_e8m0_to_f32(p[i]);
        return _mm_loadu_ps(s);
    }
    return _mm_castsi128_ps(_mm_slli_epi32(v, 23));
}

/* Four 32-wide (FP4) or 128-wide (FP8) block sums packed into four floats. */
static inline __m128 dsv4_fold4_ps(__m256 s0, __m256 s1,
                                   __m256 s2, __m256 s3)
{
    const __m256 h01 = _mm256_hadd_ps(s0, s1);
    const __m256 h23 = _mm256_hadd_ps(s2, s3);
    const __m256 h   = _mm256_hadd_ps(h01, h23);
    return _mm_add_ps(_mm256_castps256_ps128(h),
                      _mm256_extractf128_ps(h, 1));
}

static inline float dsv4_hsum128(__m128 s)
{
    __m128 t = _mm_hadd_ps(s, s);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t);
}

#endif /* DSV4_DEC_SCALE_H */

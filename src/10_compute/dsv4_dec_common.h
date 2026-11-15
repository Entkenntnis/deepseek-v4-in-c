/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_dec_common.h - horizontal reduction shared by the float-inner decoders. */
#ifndef DSV4_DEC_COMMON_H
#define DSV4_DEC_COMMON_H

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

#endif /* DSV4_DEC_COMMON_H */

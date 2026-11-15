/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_dec_bf16.h - BF16 decoder used by the reference and batch kernels. */
#ifndef DSV4_DEC_BF16_H
#define DSV4_DEC_BF16_H

#include <stdint.h>

#include <immintrin.h>

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

#endif /* DSV4_DEC_BF16_H */

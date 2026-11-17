/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_dec_fp8.h - E4M3 fast-path decoder and 128-wide block sum. */
#ifndef DSV4_DEC_FP8_H
#define DSV4_DEC_FP8_H

#include <stdint.h>

#include <immintrin.h>

/* FAST PATH FOR THE COMMON CASE, WITH A BLOCK-LEVEL SCALAR FALLBACK.
 *
 * A decode profile put most of this kernel's cycles on vpgatherdd and the
 * vpmovzxbd feeding it. A gather is several load uops per eight weights, and
 * the kernel was load-port bound, not DRAM-bound. The table lookup is only
 * needed for the subnormal E4M3 codes and for the two NaNs.
 *
 * The model's own weights almost never contain either: in a sampled trunk
 * tensor essentially no bytes are subnormal and none are NaN. A 128-byte block
 * is therefore almost always special-free. For those blocks the decode is the
 * closed form
 *
 *     0x3C000000 + ((byte & 0x7F) << 20) | sign
 *
 * -- five integer ops and one load per eight weights, against one eight-way
 * gather per eight weights. The rare special block takes the exact gather
 * path. The model-level contract is the L2 gate in `make test`, not bit
 * identity.
 *
 * The scan is 4 x 32 bytes and walks a block that is about to be read anyway;
 * the second read hits L1. A uniformly random matrix contains far more
 * subnormals, so almost every block takes the fallback and this path is slower
 * there. That is a property of the benchmark input, not of the model: random
 * bytes are a far worse approximation to E4M3 weights. Compare with
 * `make bench`. */
static inline __m256 fp8_8_normal(const uint8_t *wb, int i)
{
    const __m128i b8 = _mm_loadl_epi64((const __m128i *)(wb + i));
    const __m256i b = _mm256_cvtepu8_epi32(b8);
    const __m256i bits = _mm256_add_epi32(
        _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x7F)), 20),
        _mm256_set1_epi32(0x3C000000));
    const __m256i sign = _mm256_and_si256(_mm256_slli_epi32(b, 24),
                                          _mm256_set1_epi32((int)0x80000000u));
    return _mm256_castsi256_ps(_mm256_or_si256(bits, sign));
}

/* Any byte with exp==0 (subnormal or zero) or abs==0x7F (NaN). */
static inline int fp8_block_has_special(const uint8_t *wb)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i emask = _mm256_set1_epi8(0x78);
    const __m256i amask = _mm256_set1_epi8(0x7F);
    const __m256i nan = _mm256_set1_epi8(0x7F);
    __m256i any = zero;
    for (int i = 0; i < 128; i += 32) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(wb + i));
        const __m256i abs = _mm256_and_si256(v, amask);
        any = _mm256_or_si256(any, _mm256_cmpeq_epi8(
            _mm256_and_si256(abs, emask), zero));
        any = _mm256_or_si256(any, _mm256_cmpeq_epi8(abs, nan));
    }
    return !_mm256_testz_si256(any, any);
}

/* One 128-wide FP8 scale block, as the sum of its four 8-lane float
 * accumulators. LOAD8(k) must return the eight decoded weights at logical
 * columns k..k+7. The tree is the production one; the macro exists so the
 * register-decode (nt == 1) and decode-once (nt > 1) forms cannot diverge. */
#define FP8_BLOCK_SUM(LOAD8)                                                   \
    do {                                                                       \
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();             \
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();             \
        for (int i = 0; i < 128; i += 32) {                                    \
            a0 = _mm256_fmadd_ps(LOAD8(i),      _mm256_loadu_ps(xb + i), a0);  \
            a1 = _mm256_fmadd_ps(LOAD8(i + 8),  _mm256_loadu_ps(xb + i + 8),   \
                                 a1);                                          \
            a2 = _mm256_fmadd_ps(LOAD8(i + 16), _mm256_loadu_ps(xb + i + 16),  \
                                 a2);                                          \
            a3 = _mm256_fmadd_ps(LOAD8(i + 24), _mm256_loadu_ps(xb + i + 24),  \
                                 a3);                                          \
        }                                                                      \
        s = _mm256_add_ps(_mm256_add_ps(a0, a1),                               \
                          _mm256_add_ps(a2, a3));                              \
    } while (0)

#endif /* DSV4_DEC_FP8_H */

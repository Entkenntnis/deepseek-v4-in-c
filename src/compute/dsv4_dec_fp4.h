/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_dec_fp4.h - E2M1 vector decode and 32-wide block sums. */
#ifndef DSV4_DEC_FP4_H
#define DSV4_DEC_FP4_H

#include <stdint.h>
#include <string.h>

#include <immintrin.h>

/* EIGHT FP4 VALUES AT ONCE, without eight table lookups.
 *
 * e2m1 decomposes exactly: code & 7 selects a magnitude from
 * {0, .5, 1, 1.5, 2, 3, 4, 6} and bit 3 is the sign. So a magnitude is one
 * cross-lane permute over an 8-entry table, and the sign is that bit shifted
 * into the float's sign position and OR-ed in. That reproduces dsv4_e2m1_lut
 * entry for entry, including -0.0 for code 8, which is why it can replace the
 * lookup rather than approximate it.
 *
 * Four packed bytes carry the eight nibbles. Duplicate each byte, widen to
 * eight 32-bit lanes, then shift the odd lanes down by four so lane k holds the
 * nibble of logical element base+k.
 *
 * Measured honestly: this is worth almost nothing on its own (81.5 GF/s against
 * 80.9 for the four-lookup form). FP4 was never dequant-bound. It is kept
 * because it is not slower and it removes the table from the hot loop, but the
 * gain in dsv4_matmul.c came from elsewhere -- see the batch kernels.
 *
 * THE TWO MASKS THAT WERE NOT NEEDED. The shift result used to be masked with
 * 0x0F and then masked again with 7 for the magnitude permute. Neither mask
 * can change the result: vpermps indexes with the low three bits, and the sign
 * test reads bit 3, which the 0x0F mask left untouched. Dropping them removes
 * two vector ops per eight weights and leaves every output bitwise identical.
 * `sh` still selects the low or high nibble; an even lane's shifted value
 * contains the rest of the byte, but no instruction ever reads those bits.
 *
 * ONE BROADCAST INSTEAD OF TWO SHUFFLES. The four packed bytes used to be
 * expanded to eight byte lanes with `vpshufb` and then widened to eight dwords
 * with `vpmovzxbd`. Both are Port-5 uops, and Port 5 is also where `vpermps`
 * runs, so the decode spent three shuffle-port uops per eight weights. A
 * broadcast of the same dword followed by a variable shift produces the exact
 * same lane contents with one Port-5 uop:
 *
 *     lane k = (four >> (4*k))
 *
 * `vpermps` only reads the low three bits (the magnitude) and the sign test
 * only reads bit 3, so all the higher bits the shift leaves behind are
 * irrelevant. The nibble order is unchanged: lane 0 is the low nibble of the
 * first byte, lane 1 its high nibble, lane 2 the low nibble of the second
 * byte, and so on. That makes this a bit-for-bit replacement, not merely an
 * equivalent decode; test_matmul's FP4 gate checks the result. */
static inline __m256 fp4_8(const uint8_t *row, int64_t base)
{
    uint32_t four;
    memcpy(&four, row + (base >> 1), 4);
    __m256i idx = _mm256_broadcastd_epi32(_mm_cvtsi32_si128((int)four));
    /* lane k gets nibble k: element 0 is the LOW nibble of byte 0. */
    idx = _mm256_srlv_epi32(idx, _mm256_setr_epi32(0, 4, 8, 12,
                                                   16, 20, 24, 28));
    const __m256 mag = _mm256_permutevar8x32_ps(
        _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f), idx);
    const __m256i sgn = _mm256_slli_epi32(
        _mm256_and_si256(idx, _mm256_set1_epi32(8)), 28);
    return _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_castps_si256(mag), sgn));
}

/* 512 blocks covers every released geometry (Pro's widest reduce dimension,
 * 16384, is 512 blocks). Anything wider takes the generic block loop. */
#define FP4_SC_MAX 512

/* One 32-wide FP4 block, as eight lane sums before the horizontal reduction.
 * LOAD8(k) must return the eight decoded weights at logical columns k..k+7.
 * The four accumulators and their pair tree are the production kernel's; the
 * macro exists so the inline-decode and decode-once forms cannot diverge. */
#define FP4_BLOCK_SUM(LOAD8)                                                   \
    _mm256_add_ps(                                                             \
        _mm256_add_ps(                                                         \
            _mm256_fmadd_ps(LOAD8(0),  _mm256_loadu_ps(xb),                     \
                            _mm256_setzero_ps()),                              \
            _mm256_fmadd_ps(LOAD8(8),  _mm256_loadu_ps(xb + 8),                 \
                            _mm256_setzero_ps())),                             \
        _mm256_add_ps(                                                         \
            _mm256_fmadd_ps(LOAD8(16), _mm256_loadu_ps(xb + 16),                \
                            _mm256_setzero_ps()),                              \
            _mm256_fmadd_ps(LOAD8(24), _mm256_loadu_ps(xb + 24),                \
                            _mm256_setzero_ps())))

/* Decode-on-the-fly, for the single-token GEMV. */
static inline __m256 fp4_block_sum32(const uint8_t *row, int64_t base,
                                     const float *xb)
{
#define FP4_LOAD8(k) fp4_8(row, base + (k))
    return FP4_BLOCK_SUM(FP4_LOAD8);
#undef FP4_LOAD8
}

/* Decode once into an f32 block, for the batched GEMM. */
static inline __m256 fp4_block_sum32_w(const float *wf, const float *xb)
{
#define FP4_LOAD8(k) _mm256_loadu_ps(wf + (k))
    return FP4_BLOCK_SUM(FP4_LOAD8);
#undef FP4_LOAD8
}

#endif /* DSV4_DEC_FP4_H */

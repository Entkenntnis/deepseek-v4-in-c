/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_qmat_dec.h - packed-weight decoders and block reductions shared by the
 * decode GEMV (src/core/dsv4_matmul.c) and the prefill GEMM.
 *
 * ONE DEFINITION PER ARITHMETIC OPERATION
 *   These blocks used to exist twice, once in the core matmul file and once in
 *   the prefill batch kernels. The arithmetic -- which bytes are decoded, in
 *   which order the products are accumulated, where the block scale is applied
 *   -- is part of the model's bit-identity contract, so it lives here and both
 *   paths include it. Only the scheduling around it is allowed to differ:
 *   decode keeps one token in flight, prefill keeps a block of token rows.
 *
 *   The kernels themselves still take different shapes on purpose. The point of
 *   this header is that a decoder or a reduction tree cannot drift between
 *   them; it is not a claim that a GEMV and a GEMM are the same loop.
 */
#ifndef DSV4_QMAT_DEC_H
#define DSV4_QMAT_DEC_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "dsv4_quant.h"

/* --------------------------------------------------------------- BF16 ----- */
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

/* ------------------------------------------------------- double reduction ---
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

/* ---------------------------------------------------------------- FP8 ----- */
/* FAST PATH FOR THE COMMON CASE, WITH A BLOCK-LEVEL EXACTNESS FALLBACK.
 *
 * `perf` on a real decode put 42.7% of this kernel's cycles on vpgatherdd and
 * 19.7% on the vpmovzxbd feeding it. A gather is eight load uops, and the
 * kernel issued 1.33 load uops per weight -- it was load-port bound, not
 * DRAM-bound. The table lookup is only needed for 16 of the 256 E4M3 codes
 * (the subnormals) and for the two NaNs.
 *
 * The model's own weights almost never contain either: in a sampled
 * layers.0.attn.wo_b tensor 0.0145% of bytes are subnormal and none are NaN.
 * A 128-byte block is therefore special-free 98.2% of the time. For those
 * blocks the decode is the closed form
 *
 *     0x3C000000 + ((byte & 0x7F) << 20) | sign
 *
 * -- five integer ops and one load per eight weights, against one eight-way
 * gather per eight weights. The other 1.8% take the exact gather path, so the
 * result is bitwise identical to the previous kernel, not merely inside its
 * L2 bound.
 *
 * The scan is 4 x 32 bytes and walks a block that is about to be read anyway;
 * the second read hits L1. A uniformly random matrix has 6.25% subnormals, so
 * almost every block takes the fallback and this path is slower there. That
 * is a property of the benchmark input, not of the model: random bytes are a
 * far worse approximation to E4M3 weights than the measured 0.0145%. */
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

/* ---------------------------------------------------------------- FP4 ----- */
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

/* 512 doubles covers every released geometry (Pro's widest reduce dimension,
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

#endif /* DSV4_QMAT_DEC_H */

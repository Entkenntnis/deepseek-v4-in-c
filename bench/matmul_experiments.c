/* SPDX-License-Identifier: Apache-2.0 */
/* bench/matmul_experiments.c - the experimental GEMV variants.
 *
 * These used to live behind #ifdef DSV4_MATMUL_EXPERIMENTS inside
 * src/core/dsv4_matmul.c. They are benchmark code, not engine code, so they
 * now live here and the production translation unit contains only the kernels
 * the engine ships.
 *
 * The small decoders below are deliberately standalone copies of the
 * production inlines. An experiment must be free to change them, and
 * matmul_variants checks every variant bitwise (or against the L2 bound)
 * before it is timed, so drift is caught rather than trusted. The copies are
 * extracted from the production file when this file is maintained; the gate is
 * what makes that safe.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_quant.h"

/* Production entry points the variants compare against. */
void dsv4_matmul_fp4(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_matmul_fp8_f32(float *, const float *, const uint8_t *,
                         const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_f32_hybrid(float *, const float *, const uint8_t *,
                                const uint8_t *, int, int, int, int);

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

static const uint32_t dsv4_e4m3_bits[256] = {
    0x00000000u, 0x3B000000u, 0x3B800000u, 0x3BC00000u, 0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u,
    0x3C800000u, 0x3C900000u, 0x3CA00000u, 0x3CB00000u, 0x3CC00000u, 0x3CD00000u, 0x3CE00000u, 0x3CF00000u,
    0x3D000000u, 0x3D100000u, 0x3D200000u, 0x3D300000u, 0x3D400000u, 0x3D500000u, 0x3D600000u, 0x3D700000u,
    0x3D800000u, 0x3D900000u, 0x3DA00000u, 0x3DB00000u, 0x3DC00000u, 0x3DD00000u, 0x3DE00000u, 0x3DF00000u,
    0x3E000000u, 0x3E100000u, 0x3E200000u, 0x3E300000u, 0x3E400000u, 0x3E500000u, 0x3E600000u, 0x3E700000u,
    0x3E800000u, 0x3E900000u, 0x3EA00000u, 0x3EB00000u, 0x3EC00000u, 0x3ED00000u, 0x3EE00000u, 0x3EF00000u,
    0x3F000000u, 0x3F100000u, 0x3F200000u, 0x3F300000u, 0x3F400000u, 0x3F500000u, 0x3F600000u, 0x3F700000u,
    0x3F800000u, 0x3F900000u, 0x3FA00000u, 0x3FB00000u, 0x3FC00000u, 0x3FD00000u, 0x3FE00000u, 0x3FF00000u,
    0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u, 0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u,
    0x40800000u, 0x40900000u, 0x40A00000u, 0x40B00000u, 0x40C00000u, 0x40D00000u, 0x40E00000u, 0x40F00000u,
    0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u,
    0x41800000u, 0x41900000u, 0x41A00000u, 0x41B00000u, 0x41C00000u, 0x41D00000u, 0x41E00000u, 0x41F00000u,
    0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u, 0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u,
    0x42800000u, 0x42900000u, 0x42A00000u, 0x42B00000u, 0x42C00000u, 0x42D00000u, 0x42E00000u, 0x42F00000u,
    0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u, 0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u,
    0x43800000u, 0x43900000u, 0x43A00000u, 0x43B00000u, 0x43C00000u, 0x43D00000u, 0x43E00000u, 0x7FC00000u,
    0x80000000u, 0xBB000000u, 0xBB800000u, 0xBBC00000u, 0xBC000000u, 0xBC200000u, 0xBC400000u, 0xBC600000u,
    0xBC800000u, 0xBC900000u, 0xBCA00000u, 0xBCB00000u, 0xBCC00000u, 0xBCD00000u, 0xBCE00000u, 0xBCF00000u,
    0xBD000000u, 0xBD100000u, 0xBD200000u, 0xBD300000u, 0xBD400000u, 0xBD500000u, 0xBD600000u, 0xBD700000u,
    0xBD800000u, 0xBD900000u, 0xBDA00000u, 0xBDB00000u, 0xBDC00000u, 0xBDD00000u, 0xBDE00000u, 0xBDF00000u,
    0xBE000000u, 0xBE100000u, 0xBE200000u, 0xBE300000u, 0xBE400000u, 0xBE500000u, 0xBE600000u, 0xBE700000u,
    0xBE800000u, 0xBE900000u, 0xBEA00000u, 0xBEB00000u, 0xBEC00000u, 0xBED00000u, 0xBEE00000u, 0xBEF00000u,
    0xBF000000u, 0xBF100000u, 0xBF200000u, 0xBF300000u, 0xBF400000u, 0xBF500000u, 0xBF600000u, 0xBF700000u,
    0xBF800000u, 0xBF900000u, 0xBFA00000u, 0xBFB00000u, 0xBFC00000u, 0xBFD00000u, 0xBFE00000u, 0xBFF00000u,
    0xC0000000u, 0xC0100000u, 0xC0200000u, 0xC0300000u, 0xC0400000u, 0xC0500000u, 0xC0600000u, 0xC0700000u,
    0xC0800000u, 0xC0900000u, 0xC0A00000u, 0xC0B00000u, 0xC0C00000u, 0xC0D00000u, 0xC0E00000u, 0xC0F00000u,
    0xC1000000u, 0xC1100000u, 0xC1200000u, 0xC1300000u, 0xC1400000u, 0xC1500000u, 0xC1600000u, 0xC1700000u,
    0xC1800000u, 0xC1900000u, 0xC1A00000u, 0xC1B00000u, 0xC1C00000u, 0xC1D00000u, 0xC1E00000u, 0xC1F00000u,
    0xC2000000u, 0xC2100000u, 0xC2200000u, 0xC2300000u, 0xC2400000u, 0xC2500000u, 0xC2600000u, 0xC2700000u,
    0xC2800000u, 0xC2900000u, 0xC2A00000u, 0xC2B00000u, 0xC2C00000u, 0xC2D00000u, 0xC2E00000u, 0xC2F00000u,
    0xC3000000u, 0xC3100000u, 0xC3200000u, 0xC3300000u, 0xC3400000u, 0xC3500000u, 0xC3600000u, 0xC3700000u,
    0xC3800000u, 0xC3900000u, 0xC3A00000u, 0xC3B00000u, 0xC3C00000u, 0xC3D00000u, 0xC3E00000u, 0xFFC00000u,
};

static inline __m256 load8_fp8(const uint8_t *wb, int i)
{
    const __m128i b8 = _mm_loadl_epi64((const __m128i *)(wb + i));
    const __m256i idx = _mm256_cvtepu8_epi32(b8);
    const __m256i bits = _mm256_i32gather_epi32((const int *)dsv4_e4m3_bits,
                                                idx, 4);
    return _mm256_castsi256_ps(bits);
}

static inline __m256 fp4_8(const uint8_t *row, int64_t base)
{
    uint32_t four;
    memcpy(&four, row + (base >> 1), 4);
    __m128i b = _mm_cvtsi32_si128((int)four);
    b = _mm_shuffle_epi8(b, _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3,
                                          -1, -1, -1, -1, -1, -1, -1, -1));
    __m256i idx = _mm256_cvtepu8_epi32(b);
#if DSV4_FP4_LOW_NIBBLE_FIRST
    const __m256i sh = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
#else
    const __m256i sh = _mm256_setr_epi32(4, 0, 4, 0, 4, 0, 4, 0);
#endif
    idx = _mm256_srlv_epi32(idx, sh);
    const __m256 mag = _mm256_permutevar8x32_ps(
        _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f), idx);
    const __m256i sgn = _mm256_slli_epi32(
        _mm256_and_si256(idx, _mm256_set1_epi32(8)), 28);
    return _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_castps_si256(mag), sgn));
}

#define XB4(i) _mm256_loadu_pd(xdb + (i))

#define LOAD1_FP8(k)  ((double)dsv4_e4m3_to_f32(wb[k]))

#define PD_LO(f) _mm256_cvtps_pd(_mm256_castps256_ps128(f))
#define PD_HI(f) _mm256_cvtps_pd(_mm256_extractf128_ps((f), 1))

/* ONE EIGHT-WIDE DEQUANT BODY FOR FP8.
 *
 * The loader hands back eight f32 weights per call; the double tree, its
 * grouping and its order are untouched, which is what keeps the path
 * bit-identical to the scalar definition. */
#define BLOCK_DOT_V_8(LOAD1, LOAD8)                                            \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            const __m256 f0 = LOAD8(i);                                        \
            const __m256 f1 = LOAD8(i + 8);                                    \
            v0 = _mm256_fmadd_pd(PD_LO(f0), XB4(i),      v0);                  \
            v1 = _mm256_fmadd_pd(PD_HI(f0), XB4(i + 4),  v1);                  \
            v2 = _mm256_fmadd_pd(PD_LO(f1), XB4(i + 8),  v2);                  \
            v3 = _mm256_fmadd_pd(PD_HI(f1), XB4(i + 12), v3);                  \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma(LOAD1(i), (double)xb[i], acc);            \
    } while (0)

#define LOAD8_FP8(i) load8_fp8(wb, (i))
#define BLOCK_DOT_FP8(LOAD1) BLOCK_DOT_V_8(LOAD1, LOAD8_FP8)

static double *widen_x(const float *x, int in)
{
    double *d = (double *)malloc((size_t)in * sizeof *d);
    if (!d) {
        fprintf(stderr, "dsv4_matmul: could not widen %d activations\n", in);
        return NULL;
    }
    for (int i = 0; i < in; i++) d[i] = (double)x[i];
    return d;
}

static inline float hsum_old(__m256 s)
{
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return _mm_cvtss_f32(t);
}

static inline float hsum_fast(__m256 s)
{
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
}

static inline double fp8_block_dot_f32_128(const uint8_t *wb,
                                           const float *xb)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 32) {
        a0 = _mm256_fmadd_ps(load8_fp8(wb, i),
                             _mm256_loadu_ps(xb + i), a0);
        a1 = _mm256_fmadd_ps(load8_fp8(wb, i + 8),
                             _mm256_loadu_ps(xb + i + 8), a1);
        a2 = _mm256_fmadd_ps(load8_fp8(wb, i + 16),
                             _mm256_loadu_ps(xb + i + 16), a2);
        a3 = _mm256_fmadd_ps(load8_fp8(wb, i + 24),
                             _mm256_loadu_ps(xb + i + 24), a3);
    }

    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1),
                             _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return (double)_mm_cvtss_f32(t);
}

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

static __attribute__((noinline)) double
fp8_block_dot_f32_128_gather(const uint8_t *wb, const float *xb)
{
    return fp8_block_dot_f32_128(wb, xb);
}

static inline double fp4_block_dot_f32(const uint8_t *row, int64_t base,
                                       const float *xb, int n)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        a0 = _mm256_fmadd_ps(fp4_8(row, base + i),
                             _mm256_loadu_ps(xb + i), a0);
        a1 = _mm256_fmadd_ps(fp4_8(row, base + i + 8),
                             _mm256_loadu_ps(xb + i + 8), a1);
        a2 = _mm256_fmadd_ps(fp4_8(row, base + i + 16),
                             _mm256_loadu_ps(xb + i + 16), a2);
        a3 = _mm256_fmadd_ps(fp4_8(row, base + i + 24),
                             _mm256_loadu_ps(xb + i + 24), a3);
    }

    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1),
                             _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    double acc = (double)_mm_cvtss_f32(t);

    /* Only hand-built fixtures reach this tail; the real FP4 block is 32. */
    for (; i < n; i++)
        acc += (double)dsv4_fp4_at(row, base + i) * (double)xb[i];
    return acc;
}

static inline double fp4_block_dot_f32_32(const uint8_t *row, int64_t base,
                                          const float *xb)
{
    const __m256 a0 = _mm256_fmadd_ps(fp4_8(row, base),
                                      _mm256_loadu_ps(xb),
                                      _mm256_setzero_ps());
    const __m256 a1 = _mm256_fmadd_ps(fp4_8(row, base + 8),
                                      _mm256_loadu_ps(xb + 8),
                                      _mm256_setzero_ps());
    const __m256 a2 = _mm256_fmadd_ps(fp4_8(row, base + 16),
                                      _mm256_loadu_ps(xb + 16),
                                      _mm256_setzero_ps());
    const __m256 a3 = _mm256_fmadd_ps(fp4_8(row, base + 24),
                                      _mm256_loadu_ps(xb + 24),
                                      _mm256_setzero_ps());
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1),
                             _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return (double)_mm_cvtss_f32(t);
}

#ifdef DSV4_MATMUL_EXPERIMENTS
/* --------------------------------------------------- experiment variants ---
 * Temporary kernels for the benchmark only. They are compiled into
 * bin/matmul_variants with -DDSV4_MATMUL_EXPERIMENTS and never into the engine.
 * All of them preserve the exact arithmetic of the production f32 kernel; the
 * benchmark checks that with memcmp before timing anything. */

/* ---------------------------------------------------- arithmetic FP8 decode ---
 * `perf annotate` on a real decode put 42.7% of the FP8 kernel's cycles on
 * vpgatherdd (16 gathers per 128-weight block) and another 19.7% on the
 * vpmovzxbd that feeds it. The gather is reading a 1 KB L1 table, so the cost
 * is the gather itself, not a cache miss.
 *
 * E4M3 has a closed form that does not need a 256-entry table:
 *
 *   e > 0:  bits = sign | ((e + 120) << 23) | (m << 20)
 *           and (e<<3)|m is exactly the low seven bits of the byte, so
 *           ((e+120)<<23) | (m<<20) = 0x3C000000 + ((byte & 0x7F) << 20).
 *
 * The two exceptions are the eight subnormal magnitudes and the two NaNs.
 * Both are corrected with a permute and a blend, so the result is still
 * bit-for-bit the table entry -- including the sign of a NaN and of -0.0.
 * The `e > 0` test is `(byte & 0x78) != 0`; NaN is `(byte & 0x7F) == 0x7F`.
 *
 * This variant is deliberately checked bitwise against the production gather
 * path over all 256 codes, not only through the L2 bound. */
static const uint32_t fp8_sub_bits[8] = {
    0x00000000u, 0x3B000000u, 0x3B800000u, 0x3BC00000u,
    0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u,
};

static inline __m256 fp8_8_arith(const uint8_t *wb, int i)
{
    const __m128i b8 = _mm_loadl_epi64((const __m128i *)(wb + i));
    const __m256i b = _mm256_cvtepu8_epi32(b8);
    const __m256i mag = _mm256_and_si256(b, _mm256_set1_epi32(0x7F));
    const __m256i sign = _mm256_and_si256(_mm256_slli_epi32(b, 24),
                                          _mm256_set1_epi32((int)0x80000000u));

    const __m256i normal = _mm256_add_epi32(
        _mm256_slli_epi32(mag, 20), _mm256_set1_epi32(0x3C000000));
    const __m256i sub = _mm256_loadu_si256((const __m256i *)fp8_sub_bits);
    const __m256i subv = _mm256_castps_si256(_mm256_permutevar8x32_ps(
        _mm256_castsi256_ps(sub), mag));

    const __m256i is_sub = _mm256_cmpeq_epi32(
        _mm256_and_si256(mag, _mm256_set1_epi32(0x78)), _mm256_setzero_si256());
    const __m256i is_nan = _mm256_cmpeq_epi32(mag, _mm256_set1_epi32(0x7F));

    __m256i magbits = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(normal), _mm256_castsi256_ps(subv),
        _mm256_castsi256_ps(is_sub)));
    magbits = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(magbits),
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7FC00000)),
        _mm256_castsi256_ps(is_nan)));
    return _mm256_castsi256_ps(_mm256_or_si256(magbits, sign));
}


#define FP8_BLOCK_128(LOAD8, RED)                                              \
    (__extension__({                                                           \
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();             \
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();             \
        for (int i = 0; i < 128; i += 32) {                                    \
            a0 = _mm256_fmadd_ps(LOAD8(wb, i),                                 \
                                 _mm256_loadu_ps(xb + i), a0);                 \
            a1 = _mm256_fmadd_ps(LOAD8(wb, i + 8),                             \
                                 _mm256_loadu_ps(xb + i + 8), a1);             \
            a2 = _mm256_fmadd_ps(LOAD8(wb, i + 16),                            \
                                 _mm256_loadu_ps(xb + i + 16), a2);            \
            a3 = _mm256_fmadd_ps(LOAD8(wb, i + 24),                            \
                                 _mm256_loadu_ps(xb + i + 24), a3);            \
        }                                                                      \
        __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1),                        \
                                 _mm256_add_ps(a2, a3));                        \
        (double)RED(s);                                                        \
    }))

#define GEN_FP8_F32_ALT(NAME, LOAD8, RED)                                      \
void NAME(float *y, const float *x, const uint8_t *W,                          \
          const uint8_t *S, int in, int out, int blk_r, int blk_c)             \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    if (blk_c == 128 && (in & 127) == 0) {                                     \
        _Pragma("omp parallel for schedule(static) if (out > 64)")              \
        for (int o = 0; o < out; o++) {                                        \
            const uint8_t *rw = W + (size_t)o * in;                            \
            const uint8_t *srow = S + (size_t)(o / blk_r) * sc;                \
            double total = 0.0;                                                \
            int bi = 0;                                                        \
            for (int c0 = 0; c0 < in; c0 += 128) {                             \
                const uint8_t *wb = rw + c0;                                   \
                const float   *xb = x + c0;                                    \
                total += FP8_BLOCK_128(LOAD8, RED) *                             \
                         (double)dsv4_e8m0_to_f32(srow[bi++]);                 \
            }                                                                  \
            y[o] = (float)total;                                               \
        }                                                                      \
        return;                                                                \
    }                                                                          \
    /* Only the benchmark's non-128 geometry reaches this. It uses the */      \
    /* production generic path so the variant is still defined everywhere. */  \
    dsv4_matmul_fp8_f32(y, x, W, S, in, out, blk_r, blk_c);                    \
}

GEN_FP8_F32_ALT(dsv4_matmul_fp8_f32_arith, fp8_8_arith, hsum_old)
GEN_FP8_F32_ALT(dsv4_matmul_fp8_f32_arith_red, fp8_8_arith, hsum_fast)
GEN_FP8_F32_ALT(dsv4_matmul_fp8_f32_red, load8_fp8, hsum_fast)
#undef GEN_FP8_F32_ALT
#undef FP8_BLOCK_128

/* The same normal-case block with the cheaper horizontal reduction, to put a
 * number on the 16.5% that `perf annotate` assigned to the two vhaddps. This
 * changes the summation order, so it is an L2-bound variant, not a bitwise
 * one, and is not adopted unless the full-model ids stay identical. */
static inline double fp8_block_dot_f32_128_normal_fastsum(const uint8_t *wb,
                                                          const float *xb)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 32) {
        a0 = _mm256_fmadd_ps(fp8_8_normal(wb, i),
                             _mm256_loadu_ps(xb + i), a0);
        a1 = _mm256_fmadd_ps(fp8_8_normal(wb, i + 8),
                             _mm256_loadu_ps(xb + i + 8), a1);
        a2 = _mm256_fmadd_ps(fp8_8_normal(wb, i + 16),
                             _mm256_loadu_ps(xb + i + 16), a2);
        a3 = _mm256_fmadd_ps(fp8_8_normal(wb, i + 24),
                             _mm256_loadu_ps(xb + i + 24), a3);
    }
    return (double)hsum_fast(_mm256_add_ps(
        _mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}

void dsv4_matmul_fp8_f32_rows_fastsum(float *y, const float *x,
                                      const uint8_t *W, const uint8_t *S,
                                      int in, int out, int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;
    if (!(blk_c == 128 && (in & 127) == 0 && blk_r == 128) || sc > 256) {
        dsv4_matmul_fp8_f32_hybrid(y, x, W, S, in, out, blk_r, blk_c);
        return;
    }
    const int nrb = (out + 127) / 128;
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int rb = 0; rb < nrb; rb++) {
        const uint8_t *srow = S + (size_t)rb * sc;
        double sdec[256];
        for (int i = 0; i < sc; i++)
            sdec[i] = (double)dsv4_e8m0_to_f32(srow[i]);
        const int o0 = rb * 128;
        int o1 = o0 + 128;
        if (o1 > out) o1 = out;
        for (int o = o0; o < o1; o++) {
            const uint8_t *rw = W + (size_t)o * in;
            double total = 0.0;
            int bi = 0;
            for (int c0 = 0; c0 < in; c0 += 128) {
                const uint8_t *wb = rw + c0;
                const float *xb = x + c0;
                const double acc = fp8_block_has_special(wb)
                                 ? fp8_block_dot_f32_128_gather(wb, xb)
                                 : fp8_block_dot_f32_128_normal_fastsum(wb, xb);
                total += acc * sdec[bi++];
            }
            y[o] = (float)total;
        }
    }
}




/* FP4 with only the horizontal reduction changed, so the decode and the loop
 * are identical to the shipped kernel. */
void dsv4_matmul_fp4_f32_red(float *y, const float *x, const uint8_t *W,
                             const uint8_t *S, int in, int out,
                             int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;
    if (!(blk_c == 32 && (in & 31) == 0)) {
        dsv4_matmul_fp4(y, x, W, S, in, out, blk_r, blk_c);
        return;
    }
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
        double total = 0.0;
        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += 32) {
            const __m256 a0 = _mm256_fmadd_ps(fp4_8(row, c0),
                                              _mm256_loadu_ps(x + c0),
                                              _mm256_setzero_ps());
            const __m256 a1 = _mm256_fmadd_ps(fp4_8(row, c0 + 8),
                                              _mm256_loadu_ps(x + c0 + 8),
                                              _mm256_setzero_ps());
            const __m256 a2 = _mm256_fmadd_ps(fp4_8(row, c0 + 16),
                                              _mm256_loadu_ps(x + c0 + 16),
                                              _mm256_setzero_ps());
            const __m256 a3 = _mm256_fmadd_ps(fp4_8(row, c0 + 24),
                                              _mm256_loadu_ps(x + c0 + 24),
                                              _mm256_setzero_ps());
            const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1),
                                           _mm256_add_ps(a2, a3));
            total += (double)hsum_fast(s) *
                     (double)dsv4_e8m0_to_f32(srow[bi++]);
        }
        y[o] = (float)total;
    }
}

static inline __m256 fp4_8_lut(const uint8_t *row, int64_t base)
{
    uint32_t four;
    memcpy(&four, row + (base >> 1), 4);
    __m128i b = _mm_cvtsi32_si128((int)four);
    b = _mm_shuffle_epi8(b, _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3,
                                          -1, -1, -1, -1, -1, -1, -1, -1));
    __m256i idx = _mm256_cvtepu8_epi32(b);
#if DSV4_FP4_LOW_NIBBLE_FIRST
    const __m256i sh = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
#else
    const __m256i sh = _mm256_setr_epi32(4, 0, 4, 0, 4, 0, 4, 0);
#endif
    idx = _mm256_and_si256(_mm256_srlv_epi32(idx, sh),
                           _mm256_set1_epi32(0x0F));
    return _mm256_i32gather_ps(dsv4_e2m1_lut, idx, 4);
}

static inline double fp4_block_dot_f32_lut(const uint8_t *row, int64_t base,
                                           const float *xb, int n)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        a0 = _mm256_fmadd_ps(fp4_8_lut(row, base + i),
                             _mm256_loadu_ps(xb + i), a0);
        a1 = _mm256_fmadd_ps(fp4_8_lut(row, base + i + 8),
                             _mm256_loadu_ps(xb + i + 8), a1);
        a2 = _mm256_fmadd_ps(fp4_8_lut(row, base + i + 16),
                             _mm256_loadu_ps(xb + i + 16), a2);
        a3 = _mm256_fmadd_ps(fp4_8_lut(row, base + i + 24),
                             _mm256_loadu_ps(xb + i + 24), a3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1),
                             _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    double acc = (double)_mm_cvtss_f32(t);
    for (; i < n; i++)
        acc += (double)dsv4_fp4_at(row, base + i) * (double)xb[i];
    return acc;
}

/* Keep the row loop, but replace the per-block c0/blk_c division with a
 * counter. This isolates the block-loop idiv. */
#define GEN_FP4_NODIV(NAME, DOT)                                               \
void NAME(float *y, const float *x, const uint8_t *W,                          \
          const uint8_t *S, int in, int out, int blk_r, int blk_c)             \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                  \
    for (int o = 0; o < out; o++) {                                            \
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);                 \
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;                    \
        double total = 0.0;                                                    \
        int bi = 0;                                                            \
        for (int c0 = 0; c0 < in; c0 += blk_c) {                               \
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;               \
            const double acc = DOT(row, (int64_t)c0, x + c0, n);               \
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);               \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
}

/* Also remove the per-row o/blk_r division by looping over row blocks. */
#define GEN_FP4_NODIV2(NAME, DOT)                                              \
void NAME(float *y, const float *x, const uint8_t *W,                          \
          const uint8_t *S, int in, int out, int blk_r, int blk_c)             \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    const int nrb = (out + blk_r - 1) / blk_r;                                 \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                  \
    for (int rb = 0; rb < nrb; rb++) {                                         \
        const uint8_t *srow = S + (size_t)rb * sc;                             \
        const int o0 = rb * blk_r;                                             \
        int o1 = o0 + blk_r;                                                   \
        if (o1 > out) o1 = out;                                                \
        for (int o = o0; o < o1; o++) {                                        \
            const uint8_t *row = W + (size_t)o * ((size_t)in / 2);             \
            double total = 0.0;                                                \
            int bi = 0;                                                        \
            for (int c0 = 0; c0 < in; c0 += blk_c) {                           \
                const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;           \
                const double acc = DOT(row, (int64_t)c0, x + c0, n);           \
                total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);           \
            }                                                                  \
            y[o] = (float)total;                                               \
        }                                                                      \
    }                                                                          \
}

static inline double fp4_block_dot_f32_32_wrap(const uint8_t *row,
                                               int64_t base,
                                               const float *xb, int n)
{
    (void)n;
    return fp4_block_dot_f32_32(row, base, xb);
}

GEN_FP4_NODIV(dsv4_matmul_fp4_nodiv, fp4_block_dot_f32)
GEN_FP4_NODIV(dsv4_matmul_fp4_nodiv_lut, fp4_block_dot_f32_lut)
GEN_FP4_NODIV2(dsv4_matmul_fp4_nodiv2_lut, fp4_block_dot_f32_lut)
GEN_FP4_NODIV2(dsv4_matmul_fp4_32_nd2, fp4_block_dot_f32_32_wrap)

/* Same FP8 double tree, with the 16-weight loop unrolled four times. */
#define BLOCK_DOT_V_8_UNROLL(LOAD1, LOAD8)                                     \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        _Pragma("GCC unroll 4")                                                \
        for (; i + 15 < n; i += 16) {                                          \
            const __m256 f0 = LOAD8(i);                                        \
            const __m256 f1 = LOAD8(i + 8);                                    \
            v0 = _mm256_fmadd_pd(PD_LO(f0), XB4(i),      v0);                  \
            v1 = _mm256_fmadd_pd(PD_HI(f0), XB4(i + 4),  v1);                  \
            v2 = _mm256_fmadd_pd(PD_LO(f1), XB4(i + 8),  v2);                  \
            v3 = _mm256_fmadd_pd(PD_HI(f1), XB4(i + 12), v3);                  \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma(LOAD1(i), (double)xb[i], acc);            \
    } while (0)
#define BLOCK_DOT_FP8_UNROLL(LOAD1) BLOCK_DOT_V_8_UNROLL(LOAD1, LOAD8_FP8)

/* 256-entry double table: four exact doubles per gather, so the f32->f64
 * conversions disappear. The table is built once before the timed loop. */
static double dsv4_e4m3_dbl[256];
static int dsv4_e4m3_dbl_ready;

static void ensure_e4m3_dbl(void)
{
    if (dsv4_e4m3_dbl_ready) return;
    for (int i = 0; i < 256; i++)
        dsv4_e4m3_dbl[i] = (double)dsv4_e4m3_to_f32((uint8_t)i);
    dsv4_e4m3_dbl_ready = 1;
}

static inline __m256d load4_fp8_dt(const uint8_t *wb, int i)
{
    int four;
    memcpy(&four, wb + i, 4);
    const __m128i idx = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(four));
    return _mm256_i32gather_pd(dsv4_e4m3_dbl, idx, 8);
}

#define LOAD4D_FP8(k) load4_fp8_dt(wb, (k))
#define BLOCK_DOT_V_D8(LOAD1, LOAD4D)                                          \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            v0 = _mm256_fmadd_pd(LOAD4D(i),      XB4(i),      v0);             \
            v1 = _mm256_fmadd_pd(LOAD4D(i + 4),  XB4(i + 4),  v1);             \
            v2 = _mm256_fmadd_pd(LOAD4D(i + 8),  XB4(i + 8),  v2);             \
            v3 = _mm256_fmadd_pd(LOAD4D(i + 12), XB4(i + 12), v3);             \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma(LOAD1(i), (double)xb[i], acc);            \
    } while (0)
#define BLOCK_DOT_FP8_DT(LOAD1) BLOCK_DOT_V_D8(LOAD1, LOAD4D_FP8)

#define GEN_FP8_NODIV(NAME, DOT)                                               \
void NAME(float *y, const float *x, const uint8_t *W,                          \
          const uint8_t *S, int in, int out, int blk_r, int blk_c)             \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                  \
    for (int o = 0; o < out; o++) {                                            \
        const uint8_t *rw = W + (size_t)o * in;                                \
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;                    \
        double total = 0.0;                                                    \
        int bi = 0;                                                            \
        for (int c0 = 0; c0 < in; c0 += blk_c) {                               \
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;               \
            const uint8_t *wb = rw + c0;                                       \
            const float   *xb = x + c0;                                        \
            const double  *xdb = xd + c0; (void)xdb;                           \
            double acc;                                                        \
            DOT(LOAD1_FP8);                                                    \
            (void)n; (void)xb;                                                 \
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);               \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
    free(xd);                                                                  \
}

GEN_FP8_NODIV(dsv4_matmul_fp8_nodiv, BLOCK_DOT_FP8)
GEN_FP8_NODIV(dsv4_matmul_fp8_unroll, BLOCK_DOT_FP8_UNROLL)

void dsv4_matmul_fp8_dt(float *y, const float *x, const uint8_t *W,
                        const uint8_t *S, int in, int out,
                        int blk_r, int blk_c)
{
    ensure_e4m3_dbl();
    const int sc = (in + blk_c - 1) / blk_c;
    double *xd = widen_x(x, in);
    if (!xd) return;
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *rw = W + (size_t)o * in;
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
        double total = 0.0;
        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += blk_c) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const uint8_t *wb = rw + c0;
            const float *xb = x + c0;
            const double *xdb = xd + c0;
            double acc;
            BLOCK_DOT_FP8_DT(LOAD1_FP8);
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);
        }
        y[o] = (float)total;
    }
    free(xd);
}

#endif /* DSV4_MATMUL_EXPERIMENTS */

#endif /* __AVX2__ && __FMA__ */

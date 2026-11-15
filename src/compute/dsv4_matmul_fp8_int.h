/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul_fp8_int.h - the FP8 table, decoders and block reduction.
 *
 * Only dsv4_matmul_fp8.c and dsv4_matmul_fp8_ref.c include this. The table is
 * one page and lives in L1; the block helpers are the exact arithmetic the
 * scalar reference is compared against.
 */
#ifndef DSV4_MATMUL_FP8_INT_H
#define DSV4_MATMUL_FP8_INT_H

#include <stdint.h>

#include <immintrin.h>

#include "dsv4_qmat_dec.h"
#include "dsv4_quant.h"

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

/* EIGHT FP8 VALUES FROM THE SAME 256-ENTRY TABLE, IN ONE GATHER.
 *
 * This is the only AVX2 FP8 decoder. It was rejected once because Skymont's
 * vpgatherdd is microcoded and made it ~2x slower than four scalar loads. The
 * engine now runs its compute team on the four P-cores (Lion Cove), where the
 * interleaved benchmark puts it 1.20-1.29x ahead on all four shapes, and the
 * whole-model A/B agrees: attention 7.5-7.7 -> 6.3-6.5 s, shared expert
 * 1.5-1.6 -> 1.3 s, and 1.04 -> 0.95 s/token with identical ids.
 *
 * The table entries are exact f32 bit patterns generated from
 * dsv4_e4m3_to_f32, and test_matmul checks the AVX2 path against the scalar
 * definition. */
static inline __m256 load8_fp8(const uint8_t *wb, int i)
{
    const __m128i b8 = _mm_loadl_epi64((const __m128i *)(wb + i));
    const __m256i idx = _mm256_cvtepu8_epi32(b8);
    const __m256i bits = _mm256_i32gather_epi32((const int *)dsv4_e4m3_bits,
                                                idx, 4);
    return _mm256_castsi256_ps(bits);
}

/* FP8 WITH A FLOAT-INNER BLOCK AND A DOUBLE FOLD.
 *
 * The old double-accumulator kernel lives in tests/unit/test_matmul.c. It
 * converts every weight f32 -> f64 (four `vcvtps2pd` per 16 weights).
 * `perf annotate` put ~33% of the FP8 kernel's cycles on that conversion, and
 * the FP8 kernel is ~52% of all decode compute. FP4 already took the same trade
 * and measured 1.36x in the kernel and -5.7% wall, so FP8 gets the same shape:
 * form each 128-wide scale block's dot product with 8-lane float FMAs, widen
 * the block ONCE, and keep the block scale and the cross-block sum in double.
 *
 * This is deliberately NOT bitwise-equal to the scalar double reference. The
 * factory gate bounds the L2 relative error (measured ~8e-8), exactly as it
 * already does for FP4. */
static inline double fp8_block_dot_f32_128(const uint8_t *wb,
                                           const float *xb)
{
    __m256 s;
#define FP8_LOAD8(k) load8_fp8(wb, (k))
    FP8_BLOCK_SUM(FP8_LOAD8);
#undef FP8_LOAD8
    return (double)dsv4_hsum256(s);
}

/* The same tree over the closed-form decoder. This is the fast path for a
 * block with no subnormal and no NaN. */
static inline double fp8_block_dot_f32_128_normal(const uint8_t *wb,
                                                  const float *xb)
{
    __m256 s;
#define FP8_LOAD8(k) fp8_8_normal(wb, (k))
    FP8_BLOCK_SUM(FP8_LOAD8);
#undef FP8_LOAD8
    return (double)dsv4_hsum256(s);
}

/* Keep the rare gather body OUT of the hot function. When it was `static
 * inline`, GCC hoisted its vpmovzxbd loads and spilled eight YMM registers to
 * the stack *before* the special/not-special branch, so the 98.2% fast path
 * paid for the 1.8% slow path on every block. `perf annotate` showed those
 * spills at 5-12% of the kernel's cycles. One call on a special block is
 * nothing next to that. */
static __attribute__((noinline)) double
fp8_block_dot_f32_128_gather(const uint8_t *wb, const float *xb)
{
    return fp8_block_dot_f32_128(wb, xb);
}

/* The decode-once form: the batch kernels decode a block into wf[128] and run
 * the same tree over it for every token. */
static inline __m256 fp8_block_sum_wf(const float *wf, const float *xb)
{
    __m256 s;
#define FP8_LOAD8(k) _mm256_loadu_ps(wf + (k))
    FP8_BLOCK_SUM(FP8_LOAD8);
#undef FP8_LOAD8
    return s;
}

/* Generic block width, for fixtures and any non-128 geometry. */
static inline double fp8_block_dot_f32(const uint8_t *wb, const float *xb,
                                       int n)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
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
    double acc = (double)_mm_cvtss_f32(t);

    for (; i < n; i++)
        acc += (double)dsv4_e4m3_to_f32(wb[i]) * (double)xb[i];
    return acc;
}

#endif /* DSV4_MATMUL_FP8_INT_H */

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul.c - matrix-vector products that consume the checkpoint's own
 * packed bytes, without ever widening a tensor.
 *
 * WHY NOT DEQUANTISE FIRST
 *   A Flash routed expert is 13,369,344 bytes packed. Widened to f32 it is
 *   8x that, a token touches six per layer across 43 layers, and none of them
 *   is resident. Dequantising into a scratch buffer would move ~26 GB per token
 *   through memory to save nothing. So the unpacking happens in the inner loop,
 *   where the byte is already in a register.
 *
 * THE REDUCTION ORDER IS PART OF THE CONTRACT
 *   Inherited from kimi-k3-in-c: accumulate in DOUBLE via fma(), in a fixed
 *   16-accumulator tree, so that the scalar reference and the vector path
 *   produce BITWISE identical output where that is the contract.
 *   -ffp-contract=off in the Makefile stops the compiler fusing anything
 *   behind our back.
 *
 *   BF16 still follows that tree exactly and test_matmul gates it on bit
 *   equality. FP4 and FP8 are the deliberate exceptions: their AVX2 kernels
 *   form one scale block in float and fold it in double, which is measurably
 *   faster but cannot be bitwise-equal to the scalar double reference. Those
 *   two use an L2 bound (1e-6, measured ~7-8e-8). The scalar double bodies
 *   remain as the references the gates compare against.
 *
 * WHERE THE SCALES GO
 *   inference/kernel.py fp4_gemm applies the weight scale to the ACCUMULATOR
 *   once per K block, not to each weight:
 *
 *       C_local_accum[i,j] += C_local[i,j] * scale_a[i] * scale_b[j]
 *
 *   with block_K = 32 = the weight group size. This file does the same: the dot
 *   product of one block is formed first, then multiplied by that block's scale,
 *   then added to the row total. Scaling each weight individually would be
 *   algebraically equal and numerically different, and would cost a multiply per
 *   element instead of per block.
 *
 *   Activations arrive already in f32 here, so there is no act scale: this is a
 *   dequantised-weight GEMV.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_quant.h"

/* One block's dot product, in the same 16-accumulator tree K3 uses. `n` is the
 * block width, always a multiple of 16 for the real geometries (32 and 128), so
 * the tail loop only runs on the hand-built fixtures. */

#include <immintrin.h>

/* THE SAME TREE, FOUR LANES AT A TIME -- and it is the same tree, not merely a
 * similar one. Put a[0..3] in v0, a[4..7] in v1, a[8..11] in v2, a[12..15] in
 * v3, and lane j of each vector holds a[j], a[4+j], a[8+j], a[12+j]. Then
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

/* --------------------------------------------------------------- loaders ---
 * One per storage format, in two widths. The scalar width is the definition;
 * the wider one must agree with it element for element, which the matmul gate
 * checks rather than assumes. */

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

/* FP8 IS A 256-SYMBOL ALPHABET, SO DECODE IT WITH A TABLE.
 *
 * The scalar definition (dsv4_e4m3_to_f32 in dsv4_quant.h) is ~20 ALU ops per
 * weight once GCC inlines it: sign mask, exponent shift, bias add, mantissa
 * shift, a branch for subnormals and one for NaNs. The useful operation next to
 * it is one quarter of a vfmadd. Measured with bench/fp8_decode.c at Flash's
 * geometries, replacing the computed decode with four loads from this table is
 * 6.7-10x faster (45-59 GF/s against 3.5-7 GF/s) and bitwise identical.
 *
 * The table is generated from dsv4_e4m3_to_f32 and the matmul gate compares
 * the AVX2 path against that scalar path over random matrices, so the two
 * cannot drift. 1 KB, permanently in L1.
 *
 * The table is read eight entries at a time with vpgatherdd (load8_fp8). An
 * earlier four-scalar-load form was faster on the E-cores, where Skymont's
 * gather is microcoded, but the compute team is now the four P-cores only.
 * There the gather is 1.20-1.29x ahead on every shape this model uses, so the
 * four-load path has been removed and gather is the only AVX2 FP8 decoder. */
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
 * gain in this file came from elsewhere -- see widen_x.
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


/* ONE EIGHT-WIDE DEQUANT BODY FOR FP8.
 *
 * The loader hands back eight f32 weights per call; the double tree, its
 * grouping and its order are untouched, which is what keeps the path
 * bit-identical to the scalar definition. */


#define LOAD4_BF16(k) load4_bf16(row, (k))

/* ONE f32 -> f64 PASS OVER THE ACTIVATIONS, reused by every output row.
 *
 * x is the same vector for all `out` rows, so widening it inside the row loop
 * did the work `out` times over: a 2048x4096 expert matrix converted 2.1M
 * values where 1024 suffice. (double)float is exact, so hoisting it changes the
 * count and nothing else. The BF16 path, which still uses it, stays bitwise.
 *
 * malloc rather than a fixed array: Pro's widest reduce dimension is 16384, and
 * a kernel that is correct only up to some assumed maximum is precisely the
 * kind of thing this engine keeps finding in itself. */
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

/* ---------------------------------------------------------------- kernels ---
 * bf16 is emitted here as `dsv4_matmul_bf16_avx2`, the only path the engine
 * dispatches to. Its scalar reference, the FP4 scalar body and the old
 * double-accumulator FP8 body live in bench/matmul_reference.c, where the
 * gates and microbenchmarks compare against them. The production FP8 entry
 * point is the row-block hybrid path below.
 *
 * WHY STATIC SCHEDULING. The rows are independent and equal in cost, and the
 * default compute team is four threads pinned to the four identical P-cores, so
 * there is nothing for a dynamic schedule to balance. It was introduced when
 * these loops also ran on the E-cores, where the P/E cost ratio left the fast
 * cores waiting at the barrier. That reason is gone: dynamic chunk 4 now only
 * adds a dispatch fetch per four rows. Interleaved 3-rep A/B in both orders
 * measured static at 0.945 s/token against dynamic's 0.9525, which is neutral
 * inside the noise; static is kept because it is simpler, not faster.
 *
 * This cannot change a result: a row's dot product depends only on its own
 * columns, and the reduction inside the row is untouched. Only which thread
 * computes which row changes, and rows do not interact. */
#define GEN_BF16(SUF, DOT)                                                     \
void dsv4_matmul_bf16##SUF(float *y, const float *x, const uint16_t *W,        \
                           int in, int out)                                    \
{                                                                              \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                  \
    for (int o = 0; o < out; o++) {                                            \
        const uint16_t *row = W + (size_t)o * in;                              \
        const float  *xb  = x;                                                 \
        const double *xdb = xd; (void)xdb;                                     \
        const int n = in;                                                      \
        double acc;                                                            \
        DOT(LOAD1_BF16, LOAD4_BF16);                                           \
        y[o] = (float)acc;                                                     \
    }                                                                          \
    free(xd);                                                                  \
}



GEN_BF16(_avx2, BLOCK_DOT_V)
void dsv4_matmul_fp4_avx2(float *y, const float *x, const uint8_t *W,
                          const uint8_t *S, int in, int out,
                          int blk_r, int blk_c);

/* FP8 WITH A FLOAT-INNER BLOCK AND A DOUBLE FOLD.
 *
 * The old double-accumulator kernel lives in bench/matmul_reference.c. It
 * converts every weight f32 -> f64 (four `vcvtps2pd` per 16 weights).
 * `perf annotate` put ~33% of the FP8 kernel's cycles on that conversion, and
 * the FP8 kernel is ~52% of all decode compute. FP4 already took the same trade
 * and measured 1.36x in the kernel and -5.7% wall, so FP8 gets the same shape:
 * form each 128-wide scale block's dot product with 8-lane float FMAs, widen
 * the block ONCE, and keep the block scale and the cross-block sum in double.
 *
 * This is deliberately NOT bitwise-equal to the scalar double reference. The
 * factory gate bounds the L2 relative error (measured ~8e-8), exactly as it
 * already does for FP4. bench/matmul_variants A/Bs the two kernels. */
/* The horizontal sum used by the production float-inner kernels. The
 * experiment variant that replaces the two vhaddps with an add/shuffle tree
 * lives in bench/matmul_experiments.c. */
static inline float hsum_old(__m256 s)
{
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
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

static inline double fp8_block_dot_f32_128_normal(const uint8_t *wb,
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
    return (double)hsum_old(_mm256_add_ps(
        _mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}

/* Defined below; declared here because the hybrid falls back to it for blocks
 * that contain a subnormal or a NaN. */
void dsv4_matmul_fp8_f32(float *y, const float *x, const uint8_t *W,
                         const uint8_t *S, int in, int out,
                         int blk_r, int blk_c);

void dsv4_matmul_fp8_f32_hybrid(float *y, const float *x, const uint8_t *W,
                                const uint8_t *S, int in, int out,
                                int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;
    if (!(blk_c == 128 && (in & 127) == 0)) {
        dsv4_matmul_fp8_f32(y, x, W, S, in, out, blk_r, blk_c);
        return;
    }
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *rw = W + (size_t)o * in;
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
        double total = 0.0;
        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += 128) {
            const uint8_t *wb = rw + c0;
            const float *xb = x + c0;
            const double acc = fp8_block_has_special(wb)
                             ? fp8_block_dot_f32_128_gather(wb, xb)
                             : fp8_block_dot_f32_128_normal(wb, xb);
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);
        }
        y[o] = (float)total;
    }
}

/* ROW-BLOCK SCHEDULING, SO THE SCALE ROW IS DECODED ONCE.
 *
 * FP8's scale grid is 128x128, so the same 64 scale bytes are used by 128
 * consecutive output rows. The row-wise loop decoded them 128 times -- that
 * scalar load + two-branch dsv4_e8m0_to_f32 showed up as `movzbl` at 15% of
 * the kernel's cycles in `perf annotate`. Decode the row's scales into a small
 * stack buffer once, then let every row in the block read a plain double.
 *
 * The arithmetic is untouched: each output row still sums its blocks in the
 * same order with the same scale, so this is bitwise identical to the hybrid
 * kernel, not merely within its L2 bound. Scheduling row blocks instead of
 * rows cannot change a result because rows do not interact. */
void dsv4_matmul_fp8_f32_rows(float *y, const float *x, const uint8_t *W,
                              const uint8_t *S, int in, int out,
                              int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;
    if (!(blk_c == 128 && (in & 127) == 0 && blk_r == 128)) {
        dsv4_matmul_fp8_f32_hybrid(y, x, W, S, in, out, blk_r, blk_c);
        return;
    }
    /* 256 doubles covers every released geometry (Pro's widest reduce is
     * 16384 = 128 blocks). Anything larger takes the general path instead of
     * pretending a fixed maximum is universal. */
    if (sc > 256) {
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
                                 : fp8_block_dot_f32_128_normal(wb, xb);
                total += acc * sdec[bi++];
            }
            y[o] = (float)total;
        }
    }
}

/* The reference gather kernel. Kept callable so the gate and matmul_variants
 * can compare the hybrid path against it bit for bit. */
void dsv4_matmul_fp8_f32(float *y, const float *x, const uint8_t *W,
                         const uint8_t *S, int in, int out,
                         int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;

    if (blk_c == 128 && (in & 127) == 0) {
        _Pragma("omp parallel for schedule(static) if (out > 64)")
        for (int o = 0; o < out; o++) {
            const uint8_t *rw = W + (size_t)o * in;
            const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
            double total = 0.0;
            int bi = 0;
            for (int c0 = 0; c0 < in; c0 += 128) {
                const double acc = fp8_block_dot_f32_128(rw + c0, x + c0);
                total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);
            }
            y[o] = (float)total;
        }
        return;
    }

    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *rw = W + (size_t)o * in;
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
        double total = 0.0;
        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += blk_c) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const double acc = fp8_block_dot_f32(rw + c0, x + c0, n);
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);
        }
        y[o] = (float)total;
    }
}

/* The names the rest of the engine calls. */
void dsv4_matmul_bf16(float *y, const float *x, const uint16_t *W,
                      int in, int out)
{
    dsv4_matmul_bf16_avx2(y, x, W, in, out);
}

void dsv4_matmul_fp8(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    dsv4_matmul_fp8_f32_rows(y, x, W, S, in, out, blk_r, blk_c);
}

void dsv4_matmul_fp4(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    dsv4_matmul_fp4_avx2(y, x, W, S, in, out, blk_r, blk_c);
}

/* FP4 with a float inner block and a double fold.
 *
 * This is the CPU analogue of the GPU path's "float inner, double fold": the
 * products inside one 32-wide scale block are accumulated with 8-lane float
 * FMAs, then that block's dot product is widened to double and the block scale
 * and cross-block sum stay in double. It is intentionally not bitwise-equal to
 * the scalar double kernel; the matmul gate bounds the difference in L2 norm
 * instead.
 *
 * Four independent float accumulators cover the 32 weights of one block, so
 * the FMAs have no serial dependency inside the block. x is already f32, so
 * this path also skips widen_x entirely. */
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

/* The real FP4 block is exactly 32 wide. This removes the generic block
 * loop, the tail check, and the loop setup from that case. It returns the
 * eight lane sums BEFORE the horizontal reduction, so the batched path below
 * can reduce four blocks at once. */
static inline __m256 fp4_block_sum32(const uint8_t *row, int64_t base,
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
    return _mm256_add_ps(_mm256_add_ps(a0, a1),
                         _mm256_add_ps(a2, a3));
}

static inline double fp4_hsum256(__m256 s)
{
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 t = _mm_add_ps(lo, hi);
    t = _mm_hadd_ps(t, t);
    t = _mm_hadd_ps(t, t);
    return (double)_mm_cvtss_f32(t);
}

static inline double fp4_block_dot_f32_32(const uint8_t *row, int64_t base,
                                          const float *xb)
{
    return fp4_hsum256(fp4_block_sum32(row, base, xb));
}

/* Four 32-wide blocks at a time, so the horizontal reduction and the double
 * fold happen once per 128 weights instead of once per 32, and the scale row
 * is decoded once per output row rather than once per block.
 *
 * Why this is not just a micro-optimisation: the old loop ended every block
 * with `total += dot * scale`, a scalar double dependency chain of one
 * multiply and one add per 32 weights. At the real geometry that is 128
 * serial steps per output row. The four-lane double accumulator below keeps
 * four independent partial sums alive and only reduces them once at the end.
 *
 * The within-block arithmetic is unchanged (same four FMA accumulators, same
 * pair tree), but the horizontal reduction now sums the lanes in a different
 * order and the cross-block sum is grouped by block index modulo four. Both
 * are float-level rounding changes inside the FP4 kernel's existing L2-bound
 * contract, not a different dot product. The gate bounds it at 1e-6 and the
 * full-model ids must still agree.
 *
 * `sdec` is fixed rather than a VLA because -Wvla is on. 512 doubles covers
 * every released geometry (Pro's widest reduce dimension, 16384, is 512
 * blocks); anything wider takes the old single-block loop instead of
 * pretending the maximum is universal. */
#define FP4_SC_MAX 512

void dsv4_matmul_fp4_avx2(float *y, const float *x, const uint8_t *W,
                          const uint8_t *S, int in, int out,
                          int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;

    if (blk_c == 32 && (in & 31) == 0 && sc <= FP4_SC_MAX) {
        _Pragma("omp parallel for schedule(static) if (out > 64)")
        for (int o = 0; o < out; o++) {
            const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
            const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
            double sdec[FP4_SC_MAX];
            for (int i = 0; i < sc; i++)
                sdec[i] = (double)dsv4_e8m0_to_f32(srow[i]);

            double total = 0.0;
            __m256d acc4 = _mm256_setzero_pd();
            int bi = 0;
            int c0 = 0;
            for (; c0 + 127 < in; c0 += 128, bi += 4) {
                const __m256 s0 = fp4_block_sum32(row, c0, x + c0);
                const __m256 s1 = fp4_block_sum32(row, c0 + 32, x + c0 + 32);
                const __m256 s2 = fp4_block_sum32(row, c0 + 64, x + c0 + 64);
                const __m256 s3 = fp4_block_sum32(row, c0 + 96, x + c0 + 96);

                /* Four 8-lane sums -> four scalars in one xmm. The old path
                 * did this with two vhaddps per block, eight Port-5 uops
                 * here; this tree needs three vhaddps plus one extract/add. */
                const __m256 h01 = _mm256_hadd_ps(s0, s1);
                const __m256 h23 = _mm256_hadd_ps(s2, s3);
                const __m256 h   = _mm256_hadd_ps(h01, h23);
                const __m128 lo  = _mm256_castps256_ps128(h);
                const __m128 hi  = _mm256_extractf128_ps(h, 1);
                const __m128 b4  = _mm_add_ps(lo, hi);

                /* No _mm256_fmadd_pd: the engine builds with
                 * -ffp-contract=off, and the scalar fold this replaces was a
                 * separate multiply and add. Keep the same rounding. */
                acc4 = _mm256_add_pd(
                    acc4,
                    _mm256_mul_pd(_mm256_cvtps_pd(b4),
                                  _mm256_loadu_pd(sdec + bi)));
            }
            for (; c0 < in; c0 += 32, bi++)
                total += fp4_block_dot_f32_32(row, (int64_t)c0, x + c0) *
                         sdec[bi];

            const __m128d dlo = _mm256_castpd256_pd128(acc4);
            const __m128d dhi = _mm256_extractf128_pd(acc4, 1);
            const __m128d d2  = _mm_add_pd(dlo, dhi);
            const __m128d d1  = _mm_unpackhi_pd(d2, d2);
            total += _mm_cvtsd_f64(_mm_add_sd(d2, d1));
            y[o] = (float)total;
        }
        return;
    }

    if (blk_c == 32 && (in & 31) == 0) {
        _Pragma("omp parallel for schedule(static) if (out > 64)")
        for (int o = 0; o < out; o++) {
            const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
            const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
            double total = 0.0;
            int bi = 0;
            for (int c0 = 0; c0 < in; c0 += 32)
                total += fp4_block_dot_f32_32(row, (int64_t)c0, x + c0) *
                         (double)dsv4_e8m0_to_f32(srow[bi++]);
            y[o] = (float)total;
        }
        return;
    }

    /* Fallback for hand-built fixtures and any future non-32 block width.
     * The counter removes the per-block integer division. */
    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;
        double total = 0.0;
        int bi = 0;
        for (int c0 = 0; c0 < in; c0 += blk_c) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const double acc = fp4_block_dot_f32(row, (int64_t)c0, x + c0, n);
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);
        }
        y[o] = (float)total;
    }
}

/* ---------------------------------------------------------------- dispatch --
 * The one call every weight matrix goes through. The branch is on a per-matrix
 * tag decided at bind time, outside the inner loops, so it costs nothing
 * measurable. A matrix carries its own block geometry precisely so that no
 * caller has to remember whether this one is 128x128 or 1x32. */
void dsv4_mmq(float *y, const float *x, const DSV4QMat *m)
{
    switch (m->wdt) {
    case DSV4_WBF16:
        dsv4_matmul_bf16(y, x, (const uint16_t *)m->w, m->cols, m->rows);
        break;
    case DSV4_WFP8:
        dsv4_matmul_fp8(y, x, (const uint8_t *)m->w, (const uint8_t *)m->s,
                        m->cols, m->rows, m->blk_r, m->blk_c);
        break;
    case DSV4_WFP4:
        dsv4_matmul_fp4(y, x, (const uint8_t *)m->w, (const uint8_t *)m->s,
                        m->cols, m->rows, m->blk_r, m->blk_c);
        break;
    default:
        /* Reaching here means a matrix was bound with a tag nothing implements.
         * Producing zeros would be a running model with a missing projection. */
        fprintf(stderr, "dsv4_mmq: matrix has unimplemented wdt %d "
                        "(%dx%d)\n", m->wdt, m->rows, m->cols);
        abort();
    }
}

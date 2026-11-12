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
 *   16-accumulator tree, so that a scalar build, an OpenMP build and (later) an
 *   AVX2 build produce BITWISE identical output. -ffp-contract=off in the
 *   Makefile stops the compiler fusing anything behind our back. The point is
 *   that a performance change can never quietly become an accuracy change.
 *
 *   An AVX2 path exists and reproduces this tree lane for lane; test_matmul
 *   gates it on bit equality against the scalar path, exactly as K3 does.
 *   Adding a faster kernel that is merely *close* would forfeit the property
 *   this whole file is arranged around.
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
#define BLOCK_DOT_S(LOAD)                                                      \
    do {                                                                       \
        double a[16] = {0};                                                    \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16)                                            \
            for (int l = 0; l < 16; l++)                                       \
                a[l] = fma(LOAD(i + l), (double)xb[i + l], a[l]);              \
        double b0 = (a[0] + a[4]) + (a[8]  + a[12]);                           \
        double b1 = (a[1] + a[5]) + (a[9]  + a[13]);                           \
        double b2 = (a[2] + a[6]) + (a[10] + a[14]);                           \
        double b3 = (a[3] + a[7]) + (a[11] + a[15]);                           \
        acc = (b0 + b1) + (b2 + b3);                                           \
        for (; i < n; i++) acc = fma(LOAD(i), (double)xb[i], acc);             \
    } while (0)

#if defined(__AVX2__) && defined(__FMA__)
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
#endif

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
#define LOAD1_FP8(k)  ((double)dsv4_e4m3_to_f32(wb[k]))
#define LOAD1_FP4(k)  ((double)dsv4_fp4_at(row, (int64_t)c0 + (k)))

#if defined(__AVX2__) && defined(__FMA__)
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
 * gain in this file came from elsewhere -- see widen_x. */
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
    idx = _mm256_and_si256(_mm256_srlv_epi32(idx, sh), _mm256_set1_epi32(0x0F));
    const __m256 mag = _mm256_permutevar8x32_ps(
        _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f),
        _mm256_and_si256(idx, _mm256_set1_epi32(7)));
    const __m256i sgn = _mm256_slli_epi32(
        _mm256_and_si256(idx, _mm256_set1_epi32(8)), 28);
    return _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_castps_si256(mag), sgn));
}

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

#define LOAD4_BF16(k) load4_bf16(row, (k))
#else
/* Never expanded without AVX2, but they must still parse. */
#define LOAD4_BF16(k) 0
#endif

/* ONE f32 -> f64 PASS OVER THE ACTIVATIONS, reused by every output row.
 *
 * x is the same vector for all `out` rows, so widening it inside the row loop
 * did the work `out` times over: a 2048x4096 expert matrix converted 2.1M
 * values where 1024 suffice. (double)float is exact, so hoisting it changes the
 * count and nothing else -- the gate still reports bit equality.
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

/* The scalar tree, called with the vector signature so one body serves both.
 * The extra arguments the vector forms take are ignored. */
#define BLOCK_DOT_S2(LOAD1, ...) BLOCK_DOT_S(LOAD1)

/* ---------------------------------------------------------------- kernels ---
 * bf16 and FP8 are emitted twice: once always, as `..._scalar`, and once as
 * `..._avx2` where the machine has AVX2+FMA. Two instantiations of ONE body, so
 * they cannot drift apart, and both are callable at runtime so the gate can
 * compare them on real data instead of trusting the argument that they must
 * agree. FP4 is the deliberate exception: its AVX2 kernel is the float-inner
 * path below, and the scalar double body remains the reference.
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
/* One FP8 GEMV body, emitted for the scalar and AVX2 variants. Kept apart from
 * the bf16/FP4 generators so the kernel is the SAME body with a different
 * loader, not a second implementation that could drift. */
#define GEN_FP8_BODY(NAME, DOT)                                                \
void NAME(float *y, const float *x, const uint8_t *W,                          \
                          const uint8_t *S, int in, int out,                   \
                          int blk_r, int blk_c)                                \
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
            const uint8_t *wb  = rw + c0;                                      \
            const float   *xb  = x + c0;                                       \
            const double  *xdb = xd + c0; (void)xdb;                           \
            double acc;                                                        \
            DOT(LOAD1_FP8);                                                    \
            /* One multiply per block, as kernel.py does -- not per element. */\
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);               \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
    free(xd);                                                                  \
}

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

/* The scalar FP4 body stays as the double reference. The AVX2 FP4 kernel is
 * the float-inner path below, not a second copy of this one. */
#define GEN_FP4_SCALAR(SUF)                                                    \
void dsv4_matmul_fp4##SUF(float *y, const float *x, const uint8_t *W,          \
                          const uint8_t *S, int in, int out,                   \
                          int blk_r, int blk_c)                                \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                  \
    for (int o = 0; o < out; o++) {                                            \
        /* Stored row is in/2 bytes wide. Getting this stride wrong reads half  \
         * a row of the wrong values and still produces finite numbers. */     \
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);                 \
        const uint8_t *srow = S + (size_t)(o / blk_r) * sc;                    \
        double total = 0.0;                                                    \
        int bi = 0;                                                            \
        for (int c0 = 0; c0 < in; c0 += blk_c) {                               \
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;               \
            const float  *xb  = x + c0;                                        \
            const double *xdb = xd + c0; (void)xdb;                            \
            double acc;                                                        \
            BLOCK_DOT_S2(LOAD1_FP4);                                           \
            total += acc * (double)dsv4_e8m0_to_f32(srow[bi++]);               \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
    free(xd);                                                                  \
}


GEN_BF16(_scalar, BLOCK_DOT_S2)
GEN_FP4_SCALAR(_scalar)
GEN_FP8_BODY(dsv4_matmul_fp8_scalar, BLOCK_DOT_S2)

#if defined(__AVX2__) && defined(__FMA__)
GEN_BF16(_avx2, BLOCK_DOT_V)
GEN_FP8_BODY(dsv4_matmul_fp8_avx2, BLOCK_DOT_FP8)
void dsv4_matmul_fp4_avx2(float *y, const float *x, const uint8_t *W,
                          const uint8_t *S, int in, int out,
                          int blk_r, int blk_c);
#define PICK(name) name##_avx2
int dsv4_matmul_has_avx2(void) { return 1; }
#else
#define PICK(name) name##_scalar
int dsv4_matmul_has_avx2(void) { return 0; }
#endif

/* The names the rest of the engine calls. */
void dsv4_matmul_bf16(float *y, const float *x, const uint16_t *W,
                      int in, int out)
{
    PICK(dsv4_matmul_bf16)(y, x, W, in, out);
}

void dsv4_matmul_fp8(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    PICK(dsv4_matmul_fp8)(y, x, W, S, in, out, blk_r, blk_c);
}

void dsv4_matmul_fp4(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    PICK(dsv4_matmul_fp4)(y, x, W, S, in, out, blk_r, blk_c);
}

#if defined(__AVX2__) && defined(__FMA__)
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
 * loop, the tail check, and the loop setup from that case. */
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

void dsv4_matmul_fp4_avx2(float *y, const float *x, const uint8_t *W,
                          const uint8_t *S, int in, int out,
                          int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;

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

#ifdef DSV4_MATMUL_EXPERIMENTS
/* --------------------------------------------------- experiment variants ---
 * Temporary kernels for the benchmark only. They are compiled into
 * bin/matmul_variants with -DDSV4_MATMUL_EXPERIMENTS and never into the engine.
 * All of them preserve the exact arithmetic of the production f32 kernel; the
 * benchmark checks that with memcmp before timing anything. */

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
#endif


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

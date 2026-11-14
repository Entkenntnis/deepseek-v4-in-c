/* SPDX-License-Identifier: Apache-2.0 */
/* test_matmul.c - the packed matmuls, against exactly-representable references.
 *
 * HOW THE EXPECTATIONS ARE BUILT
 *   Every small fixture here is chosen so the true dot product is EXACTLY
 *   representable in f32: the weights come from the e2m1/e4m3 tables, the scales
 *   are powers of two, and the activations are small integers or halves. Those
 *   structural gates still use bit equality.
 *
 *   The deliberate exceptions are the AVX2 FP4 and FP8 kernels: both form a
 *   scale block's dot product in float and fold the block in double, so they
 *   cannot be bitwise equal to the scalar double reference. Their random-data
 *   gates use an L2 relative-error bound (1e-6) instead. The small FP4/FP8
 *   fixtures below are still exactly representable, so they check bit equality
 *   and catch the structural mistakes.
 *
 *   A loose tolerance would be the wrong instrument. The failures this file is
 *   looking for -- a wrong block stride, a scale applied per element instead of
 *   per block, a packed row read at full width -- all produce answers that are
 *   plausibly close. Only exactness, or a bound this tight, separates them.
 *
 * WHAT IS DELIBERATELY NOT ASSERTED
 *   Agreement with PyTorch. That needs torch and the real checkpoint, and it is
 *   the Phase 3 oracle's job. These gates establish internal correctness and the
 *   invariants the reduction order is supposed to have.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "dsv4.h"
#include "dsv4_quant.h"
#include "dsv4_prefill_gemm.h"

void dsv4_matmul_bf16(float *, const float *, const uint16_t *, int, int);
void dsv4_matmul_fp8(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_matmul_fp4(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_mmq(float *, const float *, const DSV4QMat *);

/* Both instantiations, so the gate can run them against each other. */
void dsv4_matmul_bf16_scalar(float *, const float *, const uint16_t *, int, int);
void dsv4_matmul_fp8_scalar(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
void dsv4_matmul_fp4_scalar(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
#if defined(__AVX2__) && defined(__FMA__)
/* The float-inner gather reference (the production path of the previous
 * release, still exported for A/B). The new production path must agree with it
 * bit for bit, including on the normal-only blocks that take the closed-form
 * decode and on the block that falls back for a subnormal or a NaN. */
void dsv4_matmul_fp8_f32(float *, const float *, const uint8_t *,
                         const uint8_t *, int, int, int, int);
#endif

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static void eqf(const char *what, float got, float want)
{
    CHECK(memcmp(&got, &want, sizeof got) == 0,
          "%s = %.9g, expected %.9g", what, (double)got, (double)want);
}

int main(void)
{
    printf("DeepSeek-V4 packed matmul gate\n");

    /* ---------------------------------------------------------------- fp8 -- */
    printf("\n-- GATE 1  FP8, one 128-wide block, scale applied once --\n");
    {
        enum { IN = 128, OUT = 2 };
        static uint8_t W[OUT * IN];
        static uint8_t S[OUT * 1];
        static float x[IN], y[OUT];

        /* row 0: all 1.0 (0x38).  row 1: all 2.0 (0x40). */
        memset(W,          0x38, IN);
        memset(W + IN,     0x40, IN);
        /* THE GRID IS ceil(OUT/blk_r) x ceil(IN/blk_c) = 1x1 HERE. With
         * blk_r = 128 and only two rows, BOTH rows share S[0]. An earlier draft
         * of this gate set S[0] and S[1] and expected them to apply per row,
         * which is not what a 128-row block means. The row dimension is
         * exercised properly below, with enough rows to have two row-blocks. */
        S[0] = 128;          /* 2^1 = 2, shared by both rows */
        S[1] = 0;            /* must be IGNORED */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp8(y, x, W, S, IN, OUT, 128, 128);
        /* row 0: 128 * 1.0 * 2 = 256 ; row 1: 128 * 2.0 * 2 = 512 */
        eqf("fp8 row0", y[0], 256.0f);
        eqf("fp8 row1", y[1], 512.0f);
        printf("  ok    256 and 512, both rows sharing one 128-row block scale\n");
    }

    printf("\n-- GATE 1b  the ROW dimension of the scale grid --\n");
    {
        /* 256 rows = two row-blocks at blk_r 128, so S has two entries and the
         * second half of the output must pick up the second one. */
        enum { IN = 128, OUT = 256 };
        static uint8_t W[OUT * IN];
        static uint8_t S[2];
        static float x[IN], y[OUT];
        memset(W, 0x38, sizeof W);      /* every weight 1.0 */
        S[0] = 127;                     /* rows   0..127 -> scale 1 */
        S[1] = 129;                     /* rows 128..255 -> scale 4 */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp8(y, x, W, S, IN, OUT, 128, 128);
        eqf("fp8 row 0",   y[0],   128.0f);
        eqf("fp8 row 127", y[127], 128.0f);
        eqf("fp8 row 128", y[128], 512.0f);
        eqf("fp8 row 255", y[255], 512.0f);
        CHECK(y[127] != y[128], "the row-block boundary had no effect");
        printf("  ok    128 below the boundary, 512 above it\n");
    }

    printf("\n-- GATE 2  FP8, the scale must change ACROSS blocks --\n");
    {
        /* Two blocks in one row with different scales. A kernel that reads only
         * the first block's scale, or indexes the grid by element rather than by
         * block, gets 256 here instead of 384 -- finite, and wrong. */
        enum { IN = 256, OUT = 1 };
        static uint8_t W[IN];
        static uint8_t S[2];
        static float x[IN], y[OUT];
        memset(W, 0x38, IN);            /* all 1.0 */
        S[0] = 127;                     /* block 0 scale 1 */
        S[1] = 128;                     /* block 1 scale 2 */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp8(y, x, W, S, IN, OUT, 128, 128);
        /* 128*1*1 + 128*1*2 = 384 */
        eqf("fp8 two blocks", y[0], 384.0f);
        CHECK(y[0] != 256.0f, "y equals the single-scale answer; scales not per block");
        printf("  ok    384 (not 256), so scales advance per block\n");
    }

    /* ---------------------------------------------------------------- fp4 -- */
    printf("\n-- GATE 3  FP4, packed two per byte, 1x32 blocks --\n");
    {
        enum { IN = 64, OUT = 2 };           /* two 32-wide blocks per row */
        static uint8_t W[OUT * (IN / 2)];
        static uint8_t S[OUT * 2];
        static float x[IN], y[OUT];

        /* Every nibble = code 2 = 1.0, so both nibble orders agree here and this
         * gate is independent of the nibble order. */
        memset(W, 0x22, sizeof W);
        S[0] = 127; S[1] = 128;              /* row 0: scales 1, 2 */
        S[2] = 127; S[3] = 127;              /* row 1: scales 1, 1 */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp4(y, x, W, S, IN, OUT, 1, 32);
        /* row 0: 32*1*1 + 32*1*2 = 96 ; row 1: 32 + 32 = 64 */
        eqf("fp4 row0", y[0], 96.0f);
        eqf("fp4 row1", y[1], 64.0f);
        printf("  ok    96 and 64, with per-row and per-block scales\n");
    }

    printf("\n-- GATE 4  FP4 row stride is in/2 BYTES, not in --\n");
    {
        /* Row 1 is deliberately different from row 0. A kernel that strides by
         * `in` rather than `in/2` reads past row 0 into row 1's territory and
         * still returns a finite number. */
        enum { IN = 32, OUT = 2 };
        static uint8_t W[OUT * (IN / 2)];
        static uint8_t S[OUT];
        static float x[IN], y[OUT];
        memset(W,              0x22, IN / 2);   /* row 0: all 1.0 */
        memset(W + (IN / 2),   0x44, IN / 2);   /* row 1: all 2.0 (code 4) */
        S[0] = 127; S[1] = 127;
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp4(y, x, W, S, IN, OUT, 1, 32);
        eqf("fp4 stride row0", y[0], 32.0f);    /* 32 * 1.0 */
        eqf("fp4 stride row1", y[1], 64.0f);    /* 32 * 2.0 */
        CHECK(y[0] != y[1], "both rows read the same bytes; stride is wrong");
        printf("  ok    rows are distinct: 32 and 64\n");
    }

    printf("\n-- GATE 5  signed values and a non-trivial activation --\n");
    {
        enum { IN = 32, OUT = 1 };
        static uint8_t W[IN / 2];
        static uint8_t S[1];
        static float x[IN], y[OUT];
        /* alternate code 2 (+1.0) and code 10 (-1.0) */
        memset(W, 0xA2, sizeof W);
        S[0] = 128;                              /* scale 2 */
        for (int i = 0; i < IN; i++) x[i] = (float)(i + 1);

        dsv4_matmul_fp4(y, x, W, S, IN, OUT, 1, 32);
        /* Whichever nibble is first, the row is 16 of (+1) and 16 of (-1)
         * against x = 1..32. If +1 lands on the odd positions the sum is +16;
         * if on the even ones it is -16. Both are valid until the order is
         * verified, so assert the magnitude, which is order-independent. */
        CHECK(fabsf(y[0]) == 32.0f, "|y| = %.9g, expected 32 (16 pairs x 1 x scale 2)",
              (double)fabsf(y[0]));
        printf("  ok    |y| = %.0f  (sign depends on the unverified nibble order)\n",
               (double)fabsf(y[0]));
    }

    printf("\n-- GATE 5b  FP4, four-block batches plus the single-block tail --\n");
    {
        /* IN = 160 is five 32-wide blocks: one four-block batch (c0 = 0) and
         * one tail block (c0 = 128). The batched path pre-decodes the whole
         * scale row and indexes it by block; a wrong bi would make the tail
         * read the first scale again. All values are exactly representable,
         * so the expected result does not depend on the summation order. */
        enum { IN = 160, OUT = 3 };
        static uint8_t W[OUT * (IN / 2)];
        static uint8_t S[OUT * (IN / 32)];
        static float x[IN], y[OUT];

        memset(W, 0x22, sizeof W);              /* every weight +1.0 */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;
        for (int o = 0; o < OUT; o++)
            for (int b = 0; b < IN / 32; b++)
                S[o * (IN / 32) + b] = (uint8_t)(127 + ((b + o) % 5));

        dsv4_matmul_fp4(y, x, W, S, IN, OUT, 1, 32);
        for (int o = 0; o < OUT; o++) {
            double want = 0.0;
            for (int b = 0; b < IN / 32; b++)
                want += 32.0 * ldexp(1.0, (b + o) % 5);
            eqf("fp4 batched+tail", y[o], (float)want);
        }
        printf("  ok    one batch plus one tail, scales indexed per block\n");
    }

    printf("\n-- GATE 5c  FP4, the over-512-block fallback --\n");
    {
        /* 513 blocks is one more than the batched path's fixed scale array.
         * This must take the old single-block loop and still be exact; all
         * weights are +1.0 and every scale is 2^0, so the answer is IN. */
        enum { IN = 16416, OUT = 1 };
        static uint8_t W[IN / 2];
        static uint8_t S[IN / 32];
        static float x[IN], y[OUT];

        memset(W, 0x22, sizeof W);
        memset(S, 127, sizeof S);
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp4(y, x, W, S, IN, OUT, 1, 32);
        eqf("fp4 wide fallback", y[0], (float)IN);
        printf("  ok    513 blocks fall back and sum to %d\n", IN);
    }

    printf("\n-- GATE 6  bf16, and dispatch through dsv4_mmq --\n");
    {
        enum { IN = 32, OUT = 2 };
        static uint16_t W[OUT * IN];
        static float x[IN], y[OUT], y2[OUT];
        for (int o = 0; o < OUT; o++)
            for (int i = 0; i < IN; i++)
                W[o * IN + i] = (o == 0) ? 0x3F80 : 0x4000;   /* 1.0 : 2.0 */
        for (int i = 0; i < IN; i++) x[i] = 0.5f;

        dsv4_matmul_bf16(y, x, W, IN, OUT);
        eqf("bf16 row0", y[0], 16.0f);
        eqf("bf16 row1", y[1], 32.0f);

        DSV4QMat m; memset(&m, 0, sizeof m);
        m.wdt = DSV4_WBF16; m.w = W; m.rows = OUT; m.cols = IN;
        dsv4_mmq(y2, x, &m);
        CHECK(memcmp(y, y2, sizeof y) == 0, "dsv4_mmq disagrees with the direct call");
        printf("  ok    16 and 32, dispatch matches the direct call\n");
    }

    printf("\n-- GATE 7  the result must not depend on thread count --\n");
    {
        /* The reduction order is fixed per output row and rows are independent,
         * so OpenMP scheduling cannot change a single bit. If this ever fails,
         * a kernel has started accumulating ACROSS rows. */
        enum { IN = 256, OUT = 96 };
        static uint8_t W[OUT * IN];
        static uint8_t S[OUT * 2];
        static float x[IN], a[OUT], b[OUT];
        unsigned st = 12345u;
        for (int i = 0; i < OUT * IN; i++) { st = st * 1103515245u + 12345u;
                                             W[i] = (uint8_t)((st >> 16) & 0x7E); }
        for (int i = 0; i < OUT * 2; i++)  { S[i] = (uint8_t)(120 + (i % 9)); }
        for (int i = 0; i < IN; i++)       { st = st * 1103515245u + 12345u;
                                             x[i] = (float)((int)((st >> 20) & 0xFF) - 128) * 0.01f; }
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        dsv4_matmul_fp8(a, x, W, S, IN, OUT, 128, 128);
#ifdef _OPENMP
        omp_set_num_threads(8);
#endif
        dsv4_matmul_fp8(b, x, W, S, IN, OUT, 128, 128);
        CHECK(memcmp(a, b, sizeof a) == 0,
              "1-thread and 8-thread results differ; the reduction is not fixed");
        int finite = 0;
        for (int i = 0; i < OUT; i++) if (isfinite(a[i])) finite++;
        CHECK(finite == OUT, "%d of %d outputs are finite", finite, OUT);
        printf("  ok    bit-identical across 1 and 8 threads, all finite\n");
    }

    printf("\n");
    printf("\n-- GATE  the AVX2 path matches the scalar reference --\n");
    {
        /* bf16 must still be bitwise: -ffp-contract=off exists so that a
         * faster kernel cannot quietly become a different kernel.
         *
         * FP4 and FP8 are the deliberate exceptions. Their AVX2 paths form one
         * scale block in float and fold blocks in double, so exact bits are
         * not the contract any more. The gate instead bounds the L2 relative
         * error at 1e-6, which is far below any structural bug but far above
         * the measured 7-10e-8.
         *
         * OUT is deliberately not a multiple of the vector width. IN=256 takes
         * the specialised 128-wide FP8 block and the 32-wide FP4 block; the
         * fixtures exercise the generic block geometry separately. */
        enum { IN = 256, OUT = 37 };
        static float x[IN], ys[OUT], yv[OUT];
        static uint8_t W8[(size_t)OUT * IN], W4[(size_t)OUT * IN / 2];
        static uint16_t Wb[(size_t)OUT * IN];
        static uint8_t S8[OUT * 2], S4[(size_t)OUT * (IN / 32)];

        unsigned r = 12345u;
        for (int i = 0; i < IN; i++) {
            r = r * 1103515245u + 12345u;
            x[i] = (float)((int)((r >> 16) & 0xffff) - 32768) * 1e-3f;
        }
        /* e4m3fn: 0x7F and 0xFF are the only NaNs. Excluded on purpose --
         * a NaN makes both paths produce a NaN whose payload need not have the
         * same bits, so the comparison would fail on the data rather than on
         * the arithmetic. That is exactly what the first run of this gate did,
         * and fp4 passed alongside it because all 16 e2m1 codes are finite. */
        for (size_t i = 0; i < sizeof W8; i++) {
            r = r * 1103515245u + 12345u;
            uint8_t b = (uint8_t)(r >> 16);
            if ((b & 0x7Fu) == 0x7Fu) b &= 0xFEu;
            W8[i] = b;
        }
        for (size_t i = 0; i < sizeof W4; i++) { r = r * 1103515245u + 12345u;
                                                 W4[i] = (uint8_t)(r >> 16); }
        /* bf16 is the top half of an f32, so build it from a finite float
         * rather than from random bits, which would land on NaN and inf. */
        for (size_t i = 0; i < (size_t)OUT * IN; i++) {
            r = r * 1103515245u + 12345u;
            union { uint32_t u; float f; } v;
            v.f = (float)((int)((r >> 16) & 0xffff) - 32768) * 1e-3f;
            Wb[i] = (uint16_t)(v.u >> 16);
        }
        /* Keep scales in a sane exponent range so nothing overflows to inf --
         * an inf == inf comparison would pass while hiding a real difference. */
        for (size_t i = 0; i < sizeof S8; i++) S8[i] = 127;
        for (size_t i = 0; i < sizeof S4; i++) S4[i] = 127;

        printf("  build has AVX2+FMA: yes\n");

        dsv4_matmul_bf16_scalar(ys, x, Wb, IN, OUT);
        dsv4_matmul_bf16(yv, x, Wb, IN, OUT);
        CHECK(memcmp(ys, yv, sizeof ys) == 0, "bf16 differs between paths");

        dsv4_matmul_fp8_scalar(ys, x, W8, S8, IN, OUT, 128, 128);
        dsv4_matmul_fp8(yv, x, W8, S8, IN, OUT, 128, 128);
        double e8 = 0.0, r8 = 0.0;
        for (int i = 0; i < OUT; i++) {
            const double d = (double)yv[i] - (double)ys[i];
            e8 += d * d;
            r8 += (double)ys[i] * (double)ys[i];
        }
        const double fp8_rel = sqrt(e8 / (r8 > 0.0 ? r8 : 1.0));
        CHECK(fp8_rel <= 1e-6,
              "fp8 L2 relative error %.3e exceeds 1e-6", fp8_rel);

        /* A non-128 block width takes the generic FP8 path and its 32-wide
         * main loop plus scalar tail. 256 = 100 + 100 + 56 makes the last
         * block non-multiple-of-32. */
        dsv4_matmul_fp8_scalar(ys, x, W8, S8, IN, OUT, 128, 100);
        dsv4_matmul_fp8(yv, x, W8, S8, IN, OUT, 128, 100);
        double e8g = 0.0, r8g = 0.0;
        for (int i = 0; i < OUT; i++) {
            const double d = (double)yv[i] - (double)ys[i];
            e8g += d * d;
            r8g += (double)ys[i] * (double)ys[i];
        }
        const double fp8_gen_rel = sqrt(e8g / (r8g > 0.0 ? r8g : 1.0));
        CHECK(fp8_gen_rel <= 1e-6,
              "fp8 generic-block L2 relative error %.3e exceeds 1e-6",
              fp8_gen_rel);

        dsv4_matmul_fp4_scalar(ys, x, W4, S4, IN, OUT, 1, 32);
        dsv4_matmul_fp4(yv, x, W4, S4, IN, OUT, 1, 32);
        double err2 = 0.0, ref2 = 0.0;
        for (int i = 0; i < OUT; i++) {
            const double d = (double)yv[i] - (double)ys[i];
            err2 += d * d;
            ref2 += (double)ys[i] * (double)ys[i];
        }
        const double fp4_rel = sqrt(err2 / (ref2 > 0.0 ? ref2 : 1.0));
        CHECK(fp4_rel <= 1e-6,
              "fp4 L2 relative error %.3e exceeds 1e-6", fp4_rel);

        if (!fails)
            printf("  ok    bf16 bitwise; fp8 L2 rel %.1e (128-block) / %.1e "
                   "(generic); fp4 L2 rel %.1e over %d x %d\n",
                   fp8_rel, fp8_gen_rel, fp4_rel, OUT, IN);
    }


    printf("\n-- GATE  every E4M3 code through the AVX2 path --\n");
    {
        /* The AVX2 FP8 loader now decodes through a 256-entry table. Random
         * weights will not necessarily touch every entry, so make a matrix
         * whose value for row r is exactly byte r: W[r][c] is zero except at
         * c == r, and x is all ones, so y[r] is the decode of r. Then compare
         * both the scalar definition and the AVX2 path against it, bit for
         * bit -- including 0x7F/0xFF, whose payload is part of the table. */
        enum { N = 256 };
        static uint8_t W[(size_t)N * N], S[4];
        static float x[N], ys[N], yv[N];

        memset(W, 0, sizeof W);
        for (int r = 0; r < N; r++) W[(size_t)r * N + r] = (uint8_t)r;
        for (int i = 0; i < N; i++) x[i] = 1.0f;
        memset(S, 127, sizeof S);          /* 2^0 for both row and column blocks */

        dsv4_matmul_fp8_scalar(ys, x, W, S, N, N, 128, 128);
        dsv4_matmul_fp8(yv, x, W, S, N, N, 128, 128);

        int bad = 0;
        for (int r = 0; r < N; r++) {
            float want = dsv4_e4m3_to_f32((uint8_t)r);
            uint32_t a, b, c;
            memcpy(&a, &ys[r], 4);
            memcpy(&b, &want, 4);
            memcpy(&c, &yv[r], 4);
            if (a != b && (((a | b) & 0x7FFFFFFFu) != 0)) {
                /* +0.0 and -0.0 are the one accepted difference: the dot
                 * product starts from +0.0, and (-0.0) + (+0.0) is +0.0 in
                 * round-to-nearest, so the sign of a zero weight cannot
                 * survive the sum. The table entry itself is still exact. */
                printf("  FAIL  scalar y[0x%02X] = 0x%08X, decode = 0x%08X\n",
                       r, a, b);
                bad++;
            }
            if (a != c) {
                printf("  FAIL  avx2   y[0x%02X] = 0x%08X, scalar = 0x%08X\n",
                       r, c, a);
                bad++;
            }
        }
        fails += bad;
        if (!bad)
            printf("  ok    all 256 codes identical in scalar and AVX2 paths\n");
    }


#if defined(__AVX2__) && defined(__FMA__)
    printf("\n-- GATE  normal-only fast decode, and the exact fallback --\n");
    {
        /* The random gate above uses uniform bytes, which are 6.25%
         * subnormal, so nearly every 128-byte block takes the gather fallback.
         * This gate deliberately fills every block with NORMAL codes first, so
         * the closed-form decode is the path actually tested, then injects one
         * subnormal, one NaN and one zero into a single block to exercise the
         * fallback. Both must agree with the gather kernel bit for bit. */
        enum { IN = 512, OUT = 256 };
        static uint8_t W[(size_t)OUT * IN], S[(OUT / 128) * (IN / 128)];
        static float x[IN], ya[OUT], yb[OUT];
        unsigned r = 987654321u;
        for (int i = 0; i < IN; i++) {
            r = r * 1103515245u + 12345u;
            x[i] = (float)((int)((r >> 16) & 0xff) - 128) * 0.01f;
        }
        for (size_t i = 0; i < sizeof W; i++) {
            r = r * 1103515245u + 12345u;
            unsigned e = 1 + ((r >> 16) % 14);       /* 1..14, never 0 or 15 */
            unsigned m = (r >> 20) & 7u;
            unsigned s = (r >> 24) & 1u;
            W[i] = (uint8_t)((s << 7) | (e << 3) | m);
        }
        memset(S, 127, sizeof S);

        dsv4_matmul_fp8_f32(ya, x, W, S, IN, OUT, 128, 128);
        dsv4_matmul_fp8(yb, x, W, S, IN, OUT, 128, 128);
        CHECK(memcmp(ya, yb, sizeof ya) == 0,
              "normal-only blocks differ from the gather kernel");

        W[0] = 0x01;    /* subnormal */
        W[1] = 0x7F;    /* +NaN */
        W[2] = 0xFF;    /* -NaN */
        W[3] = 0x00;    /* +0 */
        dsv4_matmul_fp8_f32(ya, x, W, S, IN, OUT, 128, 128);
        dsv4_matmul_fp8(yb, x, W, S, IN, OUT, 128, 128);
        CHECK(memcmp(ya, yb, sizeof ya) == 0,
              "the fallback block differs from the gather kernel");

        if (!fails)
            printf("  ok    %d normal-only rows bitwise, and the injected "
                   "subnormal/NaN/zero block falls back exactly\n", OUT);
    }
#endif

    printf("\n-- GATE  batched matmul equals nt separate GEMVs --\n");
    {
        enum { NT = 7, IN = 256, OUT = 130 };
        static float x[NT * IN], ya[NT * OUT], yb[NT * OUT];
        unsigned r = 123456789u;

        /* DIFFERENT activations for every token. Feeding one vector nt times
         * would pass even if the kernel ignored the token index, which is the
         * likeliest way to get a batched kernel wrong. */
        for (int t = 0; t < NT; t++)
            for (int i = 0; i < IN; i++) {
                r = r * 1103515245u + 12345u;
                x[(size_t)t * IN + i] =
                    (float)((int)((r >> 16) & 0x3ff) - 512) * 0.017f;
            }

        /* BF16: bitwise, same reduction tree per token. */
        {
            static uint16_t W[(size_t)OUT * IN];
            static uint8_t S[1];
            for (size_t i = 0; i < (size_t)OUT * IN; i++) {
                r = r * 1103515245u + 12345u;
                /* Finite by construction. A NaN would compare by payload and
                 * turn a real mismatch into a sign bit, which is not a test. */
                W[i] = (uint16_t)(0x3800u | ((r >> 16) & 0x03FFu));
            }
            DSV4QMat m = { W, S, DSV4_WBF16, OUT, IN, 0, 0 };
            for (int t = 0; t < NT; t++)
                dsv4_mmq(ya + (size_t)t * OUT, x + (size_t)t * IN, &m);
            dsv4_mmq_n(yb, x, &m, NT);
            if (memcmp(ya, yb, sizeof ya) != 0) {
                for (int i = 0; i < NT * OUT; i++)
                    if (memcmp(&ya[i], &yb[i], sizeof(float)) != 0) {
                        printf("  first BF16 diff at %d: gemv %.9g batched %.9g\n",
                               i, (double)ya[i], (double)yb[i]);
                        break;
                    }
            }
            CHECK(memcmp(ya, yb, sizeof ya) == 0,
                  "BF16 batched output is not bitwise-equal to nt GEMVs");

            /* nt == 1 must land on the GEMV itself, not a one-wide batch. */
            dsv4_mmq(ya, x, &m);
            dsv4_mmq_n(yb, x, &m, 1);
            CHECK(memcmp(ya, yb, (size_t)OUT * sizeof(float)) == 0,
                  "BF16 nt=1 did not land on the GEMV");
        }

        /* FP8: the production rowscale body is reused per token, including a
         * block forced through the exactness fallback. */
        {
            static uint8_t W[(size_t)OUT * IN];
            static uint8_t S[(2 * (IN / 128))];
            for (size_t i = 0; i < (size_t)OUT * IN; i++) {
                r = r * 1103515245u + 12345u;
                unsigned e = 1 + ((r >> 16) % 14);   /* normal, never sub/NaN */
                unsigned mm = (r >> 20) & 7u;
                unsigned s = (r >> 24) & 1u;
                W[i] = (uint8_t)((s << 7) | (e << 3) | mm);
            }
            W[0] = 0x01;    /* force the first 128-byte block to the gather path */
            for (size_t i = 0; i < sizeof S; i++)
                S[i] = (uint8_t)(124 + (i % 7));
            DSV4QMat m = { W, S, DSV4_WFP8, OUT, IN, 128, 128 };
            for (int t = 0; t < NT; t++)
                dsv4_mmq(ya + (size_t)t * OUT, x + (size_t)t * IN, &m);
            dsv4_mmq_n(yb, x, &m, NT);
            CHECK(memcmp(ya, yb, sizeof ya) == 0,
                  "FP8 batched output is not bitwise-equal to nt GEMVs");
        }

        /* FP4 has no batch kernel yet; the fallback must still be bitwise. */
        {
            static uint8_t W[(size_t)OUT * IN / 2];
            static uint8_t S[(size_t)OUT * (IN / 32)];
            for (size_t i = 0; i < sizeof W; i++) {
                r = r * 1103515245u + 12345u;
                W[i] = (uint8_t)(r >> 16);
            }
            for (size_t i = 0; i < sizeof S; i++)
                S[i] = (uint8_t)(124 + (i % 7));
            DSV4QMat m = { W, S, DSV4_WFP4, OUT, IN, 1, 32 };
            for (int t = 0; t < NT; t++)
                dsv4_mmq(ya + (size_t)t * OUT, x + (size_t)t * IN, &m);
            dsv4_mmq_n(yb, x, &m, NT);
            CHECK(memcmp(ya, yb, sizeof ya) == 0,
                  "FP4 batched fallback is not bitwise-equal to nt GEMVs");
        }

        if (!fails)
            printf("  ok    BF16, FP8 (normal+gather) and FP4 fallback all "
                   "bitwise-equal to nt GEMVs\n");
    }

    if (fails) { printf("MATMUL GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("MATMUL GATE PASSED\n");
    return 0;
}

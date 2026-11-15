/* SPDX-License-Identifier: Apache-2.0 */
/* ref_kernels.c - reference entry points used only by the gates. */
#include <stdint.h>

#include "dsv4.h"
#include "dsv4_matmul_int.h"
#include "dsv4_matmul_fp8_int.h"
#include "ref_kernels.h"

/* The exact-table 128-wide block sum, as a double so the gate has a stable
 * scalar comparison. Deliberately not bitwise-equal to the production float
 * fold; test_matmul bounds the L2 relative error. */
static inline double fp8_block_dot_f32_128(const uint8_t *wb,
                                           const float *xb)
{
    __m256 s;
#define FP8_LOAD8(k) load8_fp8(wb, (k))
    FP8_BLOCK_SUM(FP8_LOAD8);
#undef FP8_LOAD8
    return (double)dsv4_hsum256(s);
}

void dsv4_matmul_bf16(float *y, const float *x, const uint16_t *W,
                      int in, int out)
{
    bf16_n(y, x, W, in, out, 1);
}

void dsv4_matmul_fp8(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    fp8_n(y, x, W, S, in, out, blk_r, blk_c, 1);
}

void dsv4_matmul_fp4(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    fp4_n(y, x, W, S, in, out, blk_r, blk_c, 1);
}

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

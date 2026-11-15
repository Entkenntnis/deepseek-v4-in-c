/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul_fp8_ref.c - exact-table FP8 reference used by the matmul gate. */
#include <stdint.h>

#include "dsv4_matmul.h"
#include "dsv4_matmul_fp8_int.h"

/* The reference gather kernel: exact f32 values from the 256-entry table, with
 * no closed-form shortcut. Kept callable so the matmul gate can compare the
 * production path against it bit for bit, including on blocks with subnormals
 * and NaNs. Production is fp8_n(), not this. */
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

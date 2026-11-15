/* SPDX-License-Identifier: Apache-2.0 */
/* ref_kernels.h - test-only reference entry points.
 *
 * These used to live in src/ and were linked into the serving binary although
 * only the matmul gate called them. The production path is dsv4_mmq()/_n();
 * nothing here is part of the engine. */
#ifndef DSV4_TEST_REF_KERNELS_H
#define DSV4_TEST_REF_KERNELS_H

#include <stdint.h>

void dsv4_matmul_bf16(float *y, const float *x, const uint16_t *W,
                      int in, int out);
void dsv4_matmul_fp8(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c);
void dsv4_matmul_fp4(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c);

/* Exact-table FP8 reference: no closed-form shortcut. The production path is
 * fp8_n(); this one is what the gate compares it against, including blocks
 * with subnormals and NaNs. */
void dsv4_matmul_fp8_f32(float *y, const float *x, const uint8_t *W,
                         const uint8_t *S, int in, int out,
                         int blk_r, int blk_c);

#endif /* DSV4_TEST_REF_KERNELS_H */

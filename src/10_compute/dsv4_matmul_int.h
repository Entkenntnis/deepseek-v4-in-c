/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul_int.h - the per-format bodies the dispatch calls.
 *
 * Not part of the public matmul API. Kept separate so dsv4_matmul.c contains
 * only the format switch and the token tiling, and each numeric format can be
 * read (and replaced) on its own. */
#ifndef DSV4_MATMUL_INT_H
#define DSV4_MATMUL_INT_H

#include <stdint.h>

void bf16_n(float *y, const float *x, const uint16_t *W,
            int in, int out, int nt);
void fp8_n(float *y, const float *x, const uint8_t *W, const uint8_t *S,
           int in, int out, int blk_r, int blk_c, int nt);
void fp4_n(float *y, const float *x, const uint8_t *W, const uint8_t *S,
           int in, int out, int blk_r, int blk_c, int nt);

#endif /* DSV4_MATMUL_INT_H */

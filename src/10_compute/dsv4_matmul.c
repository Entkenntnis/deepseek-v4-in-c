/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul.c - packed-weight matrix product dispatch and token tiling.
 *
 * ONE IMPLEMENTATION, TWO SCHEDULES
 *   dsv4_mmq_n() is the only packed matmul entry point. Decode calls it with
 *   nt == 1 (dsv4_mmq) and prefill calls it with a block of token rows. Per
 *   format the bodies live in dsv4_matmul_{bf16,fp8,fp4}.c; they share the
 *   decoders and reduction trees in dsv4_qmat_dec.h, so the schedules cannot
 *   drift arithmetically.
 *
 * WHY NOT DEQUANTISE FIRST
 *   A Flash routed expert is 13,369,344 bytes packed. Widened to f32 it is 8x
 *   that, a token touches six per layer across 43 layers, and none of them is
 *   resident. Unpacking happens in the inner loop, where the byte is already
 *   in a register.
 *
 * THE REDUCTION ORDER IS PART OF THE CPU CONTRACT
 *   The scalar reference and the CPU vector paths accumulate in a fixed tree.
 *   BF16 happens to match that reference bit for bit; FP4 and FP8 form and
 *   accumulate their block sums in float with four partials and a pairwise
 *   final tree. The numerical contract is L2 relative error, checked by the
 *   gates in `make test` against the double scalar reference. The Vulkan
 *   prefill GEMMs use a different reduction order and are held to the same
 *   L2/oracle rule, not to bit identity. The scalar reference stays double;
 *   -ffp-contract=off keeps the compiler out of the CPU path.
 */
#include <stdio.h>
#include <stdlib.h>

#include "dsv4.h"
#include "dsv4_matmul.h"
#include "dsv4_matmul_int.h"

void dsv4_mmq(float *y, const float *x, const DSV4QMat *m)
{
    dsv4_mmq_n(y, x, m, 1);
}

/* One kernel tile. nt is 1..DSV4_MAX_BATCH, which bounds the fixed-size
 * accumulators in the format bodies and keeps the activation working set inside
 * LLC. dsv4_mmq_n() walks as many tiles as the caller's token block needs. */
static void mmq_n_tile(float *y, const float *x, const DSV4QMat *m, int nt)
{
    switch (m->wdt) {
    case DSV4_WBF16:
        bf16_n(y, x, (const uint16_t *)m->w, m->cols, m->rows, nt);
        break;
    case DSV4_WFP8:
        fp8_n(y, x, (const uint8_t *)m->w, (const uint8_t *)m->s,
              m->cols, m->rows, m->blk_r, m->blk_c, nt);
        break;
    case DSV4_WFP4:
        fp4_n(y, x, (const uint8_t *)m->w, (const uint8_t *)m->s,
              m->cols, m->rows, m->blk_r, m->blk_c, nt);
        break;
    default:
        fprintf(stderr, "dsv4_mmq_n: matrix has unimplemented wdt %d "
                        "(%dx%d)\n", m->wdt, m->rows, m->cols);
        abort();
    }
}

void dsv4_mmq_n(float *y, const float *x, const DSV4QMat *m, int nt)
{
    if (nt <= 0) return;

    for (int off = 0; off < nt; off += DSV4_MAX_BATCH) {
        int t = nt - off;
        if (t > DSV4_MAX_BATCH) t = DSV4_MAX_BATCH;
        mmq_n_tile(y + (size_t)off * (size_t)m->rows,
                   x + (size_t)off * (size_t)m->cols, m, t);
    }
}

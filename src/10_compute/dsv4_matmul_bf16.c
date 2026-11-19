/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul_bf16.c - BF16 batch body and its nt == 1 entry point. */
#include <stdio.h>
#include <stdlib.h>

#include "dsv4_matmul.h"
#include "dsv4_matmul_int.h"
#include "dsv4_qmat_dec.h"

/* --------------------------------------------------------- batch kernels --
 * WHY STATIC SCHEDULING. The rows are independent and equal in cost, and the
 * default compute team is four threads pinned to the four identical P-cores, so
 * there is nothing for a dynamic schedule to balance. It was introduced when
 * these loops also ran on the E-cores, where the P/E cost ratio left the fast
 * cores waiting at the barrier. That reason is gone: dynamic chunk 4 now only
 * adds a dispatch fetch per four rows. A/B runs are neutral inside the noise
 * (see the old microbenchmarks); static is kept because it is simpler, not because it is
 * faster.
 *
 * This cannot change a result: a row's dot product depends only on its own
 * columns, and the reduction inside the row is untouched. Only which thread
 * computes which row changes, and rows do not interact. */
void bf16_n(float *y, const float *x, const uint16_t *W,
                   int in, int out, int nt)
{
    double *xd = (double *)malloc((size_t)nt * (size_t)in * sizeof *xd);
    if (!xd) {
        fprintf(stderr, "dsv4_matmul: out of memory\n");
        return;
    }
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < in; i++)
            xd[(size_t)t * in + i] = (double)x[(size_t)t * in + i];

    _Pragma("omp parallel for schedule(static) if (out > 64)")
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        const int n = in;
        for (int t = 0; t < nt; t++) {
            const float  *xb  = x  + (size_t)t * in;
            const double *xdb = xd + (size_t)t * in; (void)xb;
            double acc;
            BLOCK_DOT_V(LOAD1_BF16, LOAD4_BF16);
            y[(size_t)t * out + o] = (float)acc;
        }
    }
    free(xd);
}

/* FP8. One token keeps the register schedule: the block is decoded straight
 * into accumulators, with the exact fallback for subnormal/NaN blocks. More
 * than one token decodes the block once into wf[128] and runs the same tree
 * over it per token. Both apply the scale to the block dot product in double,
 * and both use FP8_BLOCK_SUM, so only the schedule differs. */

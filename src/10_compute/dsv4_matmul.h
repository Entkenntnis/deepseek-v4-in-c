/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul.h - the packed matmul entry points.
 *
 * dsv4_mmq() is the decode path: one token, one weight matrix, one GEMV.
 * dsv4_mmq_n() is the batched form prefill runs the prompt through: nt token
 * rows share each decoded weight block. Both dispatch on the matrix's own
 * format tag, so no caller has to remember whether this one is 128x128 FP8 or
 * 1x32 FP4.
 *
 * NUMERICAL CONTRACT
 *   The CPU batch kernels share the decoders and accumulator tree with the
 *   GEMV, so a batch row is the same arithmetic as the one-token call and
 *   test_matmul keeps that as a strong scheduling check. The model-level
 *   contract is L2 relative error, not bit identity: test_matmul's random-data
 *   gates bound the production kernels against the scalar double reference,
 *   and the Vulkan prefill path has its own reduction order. Run `make test`
 *   for the gates; `make e2e` reports throughput.
 */
#ifndef DSV4_MATMUL_H
#define DSV4_MATMUL_H

#include "dsv4.h"

/* Rows per kernel tile. This is NOT the prefill block size: the whole prompt
 * is one block, and dsv4_mmq_n() walks it in tiles of at most this many rows.
 * The value bounds the fixed-size accumulator arrays in the batch kernels and
 * keeps the activation working set in LLC. The kernel curve flattens well
 * before this size (see the nt filters in the old microbenchmarks); 128 is deliberately
 * above that point and costs only stack/workspace, not another copy of the
 * weights. */
#define DSV4_MAX_BATCH     128

/* y[t*rows + o] = sum_c x[t*cols + c] * W[o,c], for t in 0..nt-1.
 *
 * nt may be any positive number: the call is split into DSV4_MAX_BATCH-row
 * tiles internally. Token rows are independent, so tiling changes neither the
 * arithmetic nor the per-(row, token) reduction order. nt == 1 must take the
 * same path as dsv4_mmq().
 */
void dsv4_mmq_n(float *y, const float *x, const DSV4QMat *m, int nt);

/* The single-token dispatch. */
void dsv4_mmq(float *y, const float *x, const DSV4QMat *m);

#endif /* DSV4_MATMUL_H */

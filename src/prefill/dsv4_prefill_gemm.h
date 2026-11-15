/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_gemm.h - batched dense matmuls for the prefill path.
 *
 * These live in src/prefill on purpose. They are not part of the decode
 * engine: decode is a GEMV and stays one. Keeping the batch kernels here means
 * src/core/dsv4_matmul.c remains the single-token reference, and a change to
 * one path cannot silently move the other.
 *
 * dsv4 requires AVX2+FMA (dsv4.h refuses to compile without them), so there is
 * no scalar or runtime-dispatch fallback to maintain here.
 */
#ifndef DSV4_PREFILL_GEMM_H
#define DSV4_PREFILL_GEMM_H

#include "dsv4.h"

/* Rows per kernel tile. This is NOT the prefill block size: the whole prompt
 * is one block now, and dsv4_mmq_n() walks it in tiles of at most this many
 * rows. The value bounds the fixed-size accumulator arrays in the batch
 * kernels and keeps the activation working set in LLC. The measured kernel
 * curve plateaus by 16-32 rows; 128 is deliberately above that and costs only
 * stack/workspace, not another copy of the weights. */
#define DSV4_MAX_BATCH     128

/* y[t*rows + o] = sum_c x[t*cols + c] * W[o,c], for t in 0..nt-1.
 *
 * nt may be any positive number: the call is split into DSV4_MAX_BATCH-row
 * tiles internally. Token rows are independent, so tiling changes neither the
 * arithmetic nor the per-(row, token) reduction order. nt == 1 must behave
 * exactly like dsv4_mmq().
 */
void dsv4_mmq_n(float *y, const float *x, const DSV4QMat *m, int nt);

/* The single-token dispatch, declared here because the batched entry point
 * falls back to it for nt == 1 and for formats without a batch kernel. */
void dsv4_mmq(float *y, const float *x, const DSV4QMat *m);

#endif /* DSV4_PREFILL_GEMM_H */

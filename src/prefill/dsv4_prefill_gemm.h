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

/* Prompt tokens processed together by the batched prefill path.
 *
 * A cap, not a tuning parameter: it bounds the fixed-size accumulator arrays
 * in the batched kernels and the per-token scratch the prefill module holds.
 * The measured kernel curve plateaus by 16-32, so 64 leaves room to test
 * without pretending an arbitrary prompt length is free. */
#define DSV4_MAX_BATCH     64

/* y[t*rows + o] = sum_c x[t*cols + c] * W[o,c], for t in 0..nt-1.
 *
 * nt == 1 must behave exactly like dsv4_mmq(). The kernels keep the
 * per-(row, token) reduction order of the GEMV; only the order in which
 * independent dot products are issued changes. nt must be 1..DSV4_MAX_BATCH.
 */
void dsv4_mmq_n(float *y, const float *x, const DSV4QMat *m, int nt);

/* The single-token dispatch, declared here because the batched entry point
 * falls back to it for nt == 1 and for formats without a batch kernel. */
void dsv4_mmq(float *y, const float *x, const DSV4QMat *m);

#endif /* DSV4_PREFILL_GEMM_H */

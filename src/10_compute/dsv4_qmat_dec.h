/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_qmat_dec.h - packed-weight decoders and block reductions shared by the
 * decode GEMV (src/10_compute/dsv4_matmul.c) and the prefill GEMM.
 *
 * ONE DEFINITION PER ARITHMETIC OPERATION
 *   These blocks used to exist twice, once in the core matmul file and once in
 *   the prefill batch kernels. The arithmetic -- which bytes are decoded, in
 *   which order the products are accumulated, where the block scale is applied
 *   -- is part of the CPU reference contract, so it lives here and both paths
 *   include it. It is checked in L2 against the scalar reference, not by bit
 *   identity; the Vulkan prefill path has its own reduction order. Only the
 *   scheduling around it is allowed to differ: decode keeps one token in
 *   flight, prefill keeps a block of token rows.
 *
 *   The kernels themselves still take different shapes on purpose. The point of
 *   this header is that a decoder or a reduction tree cannot drift between
 *   them; it is not a claim that a GEMV and a GEMM are the same loop.
 */
#ifndef DSV4_QMAT_DEC_H
#define DSV4_QMAT_DEC_H

#include "dsv4_quant.h"
#include "dsv4_dec_common.h"
#include "dsv4_dec_bf16.h"
#include "dsv4_dec_reduce.h"
#include "dsv4_dec_fp8.h"
#include "dsv4_dec_fp4.h"
#include "dsv4_dec_scale.h"

#endif /* DSV4_QMAT_DEC_H */

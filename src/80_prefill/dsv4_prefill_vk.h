/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_vk.h - prefill-only Vulkan GEMM offload. */
#ifndef DSV4_PREFILL_VK_H
#define DSV4_PREFILL_VK_H

#include "dsv4.h"
#include "dsv4_cache.h"
#include "dsv4_trunk.h"

typedef struct DSV4PrefillVk DSV4PrefillVk;

/* Returns 0 on success, -1 when Vulkan or host import is unavailable. The
 * caller keeps its CPU path then. */
int  dsv4_prefill_vk_init(DSV4PrefillVk **out, const DSV4Cfg *cfg,
                          DSV4Trunk *tr, DSV4Cache *cache);
void dsv4_prefill_vk_free(DSV4PrefillVk *v);

/* Y[n][m->rows] = X[n][m->cols] * W[m->rows][m->cols]^T. Returns 0 when the
 * whole GEMM ran on the GPU, -1 for any unsupported geometry/allocation, in
 * which case the caller must use dsv4_mmq_n(). */
int  dsv4_prefill_vk_gemm(DSV4PrefillVk *v, float *y, const float *x,
                          const DSV4QMat *m, int n);

#endif /* DSV4_PREFILL_VK_H */

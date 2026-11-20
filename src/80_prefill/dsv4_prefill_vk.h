/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_vk.h - prefill-only Vulkan GEMM offload.
 *
 * NUMERICS. The shaders accumulate in float and use a tile-dependent reduction
 * order, so their output is not bit-identical to the CPU kernels. The contract
 * is the same L2 relative-error rule as the rest of the matmul layer; `make
 * test` holds the CPU gates, and `make e2e` exercises the Vulkan backend end
 * to end when it initializes.
 */
#ifndef DSV4_PREFILL_VK_H
#define DSV4_PREFILL_VK_H

#include "dsv4.h"
#include "dsv4_cache.h"
#include "dsv4_trunk.h"

typedef struct DSV4PrefillVk DSV4PrefillVk;

/* Host-pointer imports are created in chunks of this size. The cache guard
 * must evict at least one whole chunk beyond the prefill block, because a
 * chunk that still contains a used slot re-commits all pages in that chunk --
 * including evicted ones -- when the prefill imports it. */
#define DSV4_VK_IMPORT_CHUNK ((int64_t)1 << 30)

/* Called after a tile is submitted and before the fence is waited on. It must
 * not touch Vulkan state or the caller's in-flight X/Y buffers. Returns 1 when
 * it did work and 0 when no CPU bucket is ready. The backend calls it in a
 * loop until the fence signals or it returns 0. */
typedef int (*DSV4PrefillVkFillFn)(void *ctx);

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

/* One routed expert: w1/w3, SwiGLU and w2 in one command buffer. `inter` is
 * the routed-expert intermediate width; `limit` is the SwiGLU clamp. */
int  dsv4_prefill_vk_expert_fused(DSV4PrefillVk *v, float *y, const float *x,
                                  const DSV4QMat *w1, const DSV4QMat *w3,
                                  const DSV4QMat *w2, int n, int inter,
                                  float limit,
                                  DSV4PrefillVkFillFn fill, void *fill_ctx);

#endif /* DSV4_PREFILL_VK_H */

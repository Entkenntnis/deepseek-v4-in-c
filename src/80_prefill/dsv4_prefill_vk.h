/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_vk.h - prefill-only Vulkan GEMM offload.
 *
 * NUMERICS. The shaders accumulate in float and use a tile-dependent reduction
 * order, so their output is not bit-identical to the CPU kernels. The contract
 * is the same L2 relative-error rule as the rest of the matmul layer; `make
 * test` holds the CPU gates, and `make e2e` versus `DSV4_PREFILL_CPU=1 make
 * e2e` compares the backends end to end.
 */
#ifndef DSV4_PREFILL_VK_H
#define DSV4_PREFILL_VK_H

#include "dsv4.h"
#include "dsv4_cache.h"
#include "dsv4_trunk.h"

typedef struct DSV4PrefillVk DSV4PrefillVk;

/* Which caller owns a GEMM. Kept separate so the instrumentation can tell
 * dense projections from routed-expert buckets. */
enum {
    DSV4_VK_KIND_DENSE = 0,
    DSV4_VK_KIND_EXPERT = 1,
    DSV4_VK_KIND_COUNT = 2
};

/* Per-kind counters accumulated inside the Vulkan backend. All times are
 * wall-clock on the submitting thread. submit_s is now only vkQueueSubmit;
 * wait_s is the fence wait. fill_s is CPU work the caller ran between those
 * two while the GPU was busy, so it is not GPU execution time. */
typedef struct {
    uint64_t calls, tiles, xbytes, ybytes;
    double desc_s;       /* vkUpdateDescriptorSets                        */
    double copy_in_s;    /* host memcpy into the mapped X buffer          */
    double record_s;     /* reset/begin/bind/push/dispatch/end            */
    double submit_s;     /* vkQueueSubmit                                 */
    double wait_s;       /* vkWaitForFences                               */
    double fill_s;       /* caller's CPU fill while the fence was pending */
    double copy_out_s;   /* host memcpy out of the mapped Y buffer        */
    double total_s;      /* whole dsv4_prefill_vk_gemm() call             */
} DSV4PrefillVkStats;

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

/* Copy the accumulated per-kind counters. `out` must hold
 * DSV4_VK_KIND_COUNT entries. */
void dsv4_prefill_vk_stats(const DSV4PrefillVk *v,
                           DSV4PrefillVkStats out[DSV4_VK_KIND_COUNT]);

/* Y[n][m->rows] = X[n][m->cols] * W[m->rows][m->cols]^T. Returns 0 when the
 * whole GEMM ran on the GPU, -1 for any unsupported geometry/allocation, in
 * which case the caller must use dsv4_mmq_n(). `kind` only selects the stats
 * bucket. */
int  dsv4_prefill_vk_gemm(DSV4PrefillVk *v, float *y, const float *x,
                          const DSV4QMat *m, int n, int kind);

/* Same GEMM, but calls fill(fill_ctx) once after each tile submit and before
 * waiting for that tile. fill may be NULL (then this is exactly the synchronous
 * path). *fill_s, when non-NULL, receives the CPU time spent in fill. */
int  dsv4_prefill_vk_gemm_overlap(DSV4PrefillVk *v, float *y, const float *x,
                                  const DSV4QMat *m, int n, int kind,
                                  DSV4PrefillVkFillFn fill, void *fill_ctx,
                                  double *fill_s);

/* One routed expert: w1/w3, SwiGLU and w2 in one command buffer. `inter` is
 * the routed-expert intermediate width; `limit` is the SwiGLU clamp. */
int  dsv4_prefill_vk_expert_fused(DSV4PrefillVk *v, float *y, const float *x,
                                  const DSV4QMat *w1, const DSV4QMat *w3,
                                  const DSV4QMat *w2, int n, int inter,
                                  float limit,
                                  DSV4PrefillVkFillFn fill, void *fill_ctx,
                                  double *fill_s);

#endif /* DSV4_PREFILL_VK_H */

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
 * wall-clock on the submitting thread; submit_s includes the fence wait, so it
 * is GPU execution plus driver latency, not a pure "submit" cost. */
typedef struct {
    uint64_t calls, tiles, xbytes, ybytes;
    double desc_s;       /* vkUpdateDescriptorSets                        */
    double copy_in_s;    /* host memcpy into the mapped X buffer          */
    double record_s;     /* reset/begin/bind/push/dispatch/end            */
    double submit_s;     /* vkQueueSubmit + vkWaitForFences               */
    double copy_out_s;   /* host memcpy out of the mapped Y buffer        */
    double total_s;      /* whole dsv4_prefill_vk_gemm() call             */
} DSV4PrefillVkStats;

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

#endif /* DSV4_PREFILL_VK_H */

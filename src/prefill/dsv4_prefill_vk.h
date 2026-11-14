/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_vk.h - Vulkan backend for the batched prefill path.
 *
 * The decode path is a GEMV per matmul and is deliberately untouched. This
 * backend exists for one shape only: eight prompt tokens against one FP8
 * weight matrix. The weights are uploaded once per trunk layer into
 * HOST_VISIBLE|HOST_CACHED|DEVICE_LOCAL memory (Vulkan memory type 2 on this
 * iGPU), which measured ~74 GB/s for the prefill shader. The host then only
 * copies the small activation batch in and out.
 */
#ifndef DSV4_PREFILL_VK_H
#define DSV4_PREFILL_VK_H

#include <stddef.h>

#include "dsv4.h"

#define DSV4_PV_MAX_LAYERS 128
#define DSV4_PV_NTOK 8
#define DSV4_PV_MAX_JOBS 16
/* x and y are each this many floats; the largest real batch is one 8-token
 * wq_b (8*32768) or eight wo_a groups (8*8*4096). 2M floats covers both. */
#define DSV4_PV_SCRATCH_FLOATS (DSV4_PV_NTOK * 32768u * 2u)

typedef struct DSV4PrefillVk DSV4PrefillVk;

DSV4PrefillVk *dsv4_pv_init(void);
void dsv4_pv_free(DSV4PrefillVk *v);

/* Upload one trunk layer's packed run. `run` is the exact base the layer's
 * DSV4QMat pointers point into (tr->pin[L] for pinned layers). Returns 0 on
 * success, -1 when this matrix set must stay on the CPU. */
int dsv4_pv_prepare_layer(DSV4PrefillVk *v, int layer, const void *run,
                          size_t nbytes);

typedef struct {
    const DSV4QMat *m;
    const float    *x;
    float          *y;
    int             n;   /* 1..DSV4_PV_NTOK */
} DSV4PVJob;

/* y[0..n-1][m->rows] = x[0..n-1][m->cols] * m, for an FP8 matrix already
 * prepared on `layer`. n <= DSV4_PV_NTOK. Returns 0 on success; -1 means the
 * caller should run its CPU per-token loop instead. */
int dsv4_pv_mmq_fp8(DSV4PrefillVk *v, int layer, const DSV4QMat *m,
                    const float *x, float *y, int n);

/* Up to DSV4_PV_MAX_JOBS independent GEMMs against the same layer, recorded in
 * one command buffer and waited on once. This is the difference between a
 * useful prefill and one that spends all its time in vkQueueSubmit: the eight
 * wo_a groups and the w1/w3 pair have no dependency on each other. */
int dsv4_pv_mmq_fp8_multi(DSV4PrefillVk *v, int layer,
                          const DSV4PVJob *jobs, int njobs);

void dsv4_pv_report(const DSV4PrefillVk *v, const char *label);

#endif /* DSV4_PREFILL_VK_H */

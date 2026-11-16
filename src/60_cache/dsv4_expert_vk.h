/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_expert_vk.h - routed FP4 experts on the iGPU, submitted in waves.
 *
 * One routed expert is still w1/w3, a barrier, SwiGLU, a barrier, w2. The
 * difference is that several experts that are available at the same time are
 * recorded into one command buffer: all w1/w3 dispatches, one barrier, all
 * SwiGLU dispatches, one barrier, all w2 dispatches, one submit and one fence.
 *
 * Waves are independent. The caller can submit a wave and keep working (read
 * the next expert, run the shared expert on the CPU) and waits for the wave
 * fences only when the outputs are needed.
 */
#ifndef DSV4_EXPERT_VK_H
#define DSV4_EXPERT_VK_H

#include "dsv4.h"
#include "dsv4_cache_vk.h"

typedef struct DSV4ExpertVk DSV4ExpertVk;

/* One forward pass, split into host wall time and device time. All durations
 * are milliseconds. `gpu_ts_valid` is 0 when the queue has no timestamp bits
 * or the query readback failed; the host fields are still valid then.
 *
 * `gpu_start_ms` is the start of this expert's GPU work relative to the first
 * expert of its wave. It lets the trace draw overlapping experts on a common
 * device timeline instead of pretending they ran one after another. */
typedef struct {
    int    valid;
    int    gpu_ts_valid;

    double copy_in_ms;
    double record_ms;
    double submit_ms;
    double fence_ms;
    double copy_out_ms;
    double total_ms;

    double gpu_w1_ms;
    double gpu_w3_ms;
    double gpu_swiglu_ms;
    double gpu_w2_ms;
    double gpu_total_ms;
    double gpu_start_ms;

    double submit_at;
    double fence_done_at;
} DSV4ExpertVkTiming;

/* One expert in a wave. k is the layer's top-k index and selects the scratch
 * region, so a wave must not contain the same k twice. out receives the w2
 * result; it is written when the wave is waited for, not when it is submitted. */
typedef struct {
    int                k;
    const DSV4ExpertW *e;
    float             *out;
} DSV4ExpertVkJob;

/* Timing for a complete wave. `expert[j]` belongs to job j. The host fields
 * describe the one submit/fence of the wave; the per-expert fields describe the
 * four device dispatches. */
typedef struct {
    int    valid;
    int    gpu_ts_valid;
    int    n;
    int    slot;
    int    timing;
    int    ks[DSV4_MAX_TOPK];
    float *out[DSV4_MAX_TOPK];

    double start_at;
    double submit_at;
    double fence_done_at;
    double copy_in_ms;
    double record_ms;
    double submit_ms;
    double wait_ms;
    double copy_out_ms;
    double total_ms;
    double gpu_span_ms;     /* first to last device timestamp of the wave */
    double gpu_anchor_at;   /* host time where that device span starts */

    DSV4ExpertVkTiming expert[DSV4_MAX_TOPK];
} DSV4ExpertVkWave;

/* cache_vk must outlive the returned context. Returns 0 on success. */
int  dsv4_expert_vk_init(DSV4ExpertVk **out, DSV4CacheVk *cache_vk,
                         const DSV4Cfg *cfg);

void dsv4_expert_vk_free(DSV4ExpertVk *g);

/* Record and submit one wave. Returns as soon as the queue submit returned; it
 * does not wait. When timing is non-zero the host phases and device timestamps
 * are collected. The wave must later be passed to dsv4_expert_vk_wait_wave(). */
int  dsv4_expert_vk_submit_wave(DSV4ExpertVk *g,
                                const DSV4ExpertVkJob *jobs, int n,
                                const float *x, int timing,
                                DSV4ExpertVkWave *wave);

/* Wait for a submitted wave, read its timestamps and copy every expert's y into
 * jobs[j].out. Releases the wave's command buffer/fence slot. */
int  dsv4_expert_vk_wait_wave(DSV4ExpertVk *g, DSV4ExpertVkWave *wave);

/* Single-expert convenience wrapper. Uses a one-job wave internally. */
int  dsv4_expert_vk_forward(DSV4ExpertVk *g, const DSV4ExpertW *e,
                            const float *x, float *out);

/* Same, and when `t` is non-NULL it is filled with the timing breakdown. */
int  dsv4_expert_vk_forward_timed(DSV4ExpertVk *g, const DSV4ExpertW *e,
                                  const float *x, float *out,
                                  DSV4ExpertVkTiming *t);

#endif /* DSV4_EXPERT_VK_H */

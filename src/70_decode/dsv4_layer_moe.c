/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer_moe.c - routed + shared expert for one token.
 *
 * Every k has its own accumulator, and the weighted sum runs in k order, so
 * read completion order can never reach the arithmetic. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_prof.h"
#include "dsv4_expert_vk.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_layer.h"
#include "dsv4_layer_int.h"
#include "dsv4_trace.h"

static void trace_mm(int layer, int k, const char *which, double t0)
{
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_MM, .layer = layer, .k = k,
                                  .ms = (dsv4_now_s() - t0) * 1000.0,
                                  .at = t0, .tag = which });
}

static void trace_expert(int kind, int layer, int k, int expert,
                         const char *tag)
{
    if (!dsv4_trace_active()) return;
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = kind, .layer = layer, .k = k,
                                  .expert = expert, .tag = tag });
}

/* One Vulkan expert is a single submit with four dispatches. The host phases
 * come from dsv4_now_s(), the four GPU stages from VkQueryPool timestamps. We
 * place the device stages on the host timeline by anchoring the busy window at
 * the fence return, so the trace's `t` stays in one clock domain. */
static void trace_gpu_breakdown(const DSV4ExpertVkTiming *tm, int layer, int k)
{
    if (!dsv4_trace_active() || !tm->valid) return;

    const double host_start =
        tm->submit_at - (tm->copy_in_ms + tm->record_ms + tm->submit_ms) / 1000.0;

    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = k,
                                  .at = host_start,
                                  .ms = tm->copy_in_ms, .tag = "cin" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = k,
                                  .at = host_start + tm->copy_in_ms / 1000.0,
                                  .ms = tm->record_ms, .tag = "rec" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = k,
                                  .at = tm->submit_at - tm->submit_ms / 1000.0,
                                  .ms = tm->submit_ms, .tag = "sub" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = k,
                                  .at = tm->fence_done_at - tm->fence_ms / 1000.0,
                                  .ms = tm->fence_ms, .tag = "wait" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = k,
                                  .at = tm->fence_done_at,
                                  .ms = tm->copy_out_ms, .tag = "cout" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = k,
                                  .at = host_start,
                                  .ms = tm->total_ms, .tag = "tot" });

    if (!tm->gpu_ts_valid) return;

    double at = tm->fence_done_at - tm->gpu_total_ms / 1000.0;
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_MM, .layer = layer, .k = k,
                                  .at = at, .ms = tm->gpu_w1_ms,
                                  .tag = "gw1" });
    at += tm->gpu_w1_ms / 1000.0;
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_MM, .layer = layer, .k = k,
                                  .at = at, .ms = tm->gpu_w3_ms,
                                  .tag = "gw3" });
    at += tm->gpu_w3_ms / 1000.0;
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_SWIGLU, .layer = layer,
                                  .k = k, .at = at, .ms = tm->gpu_swiglu_ms,
                                  .tag = "gswiglu" });
    at += tm->gpu_swiglu_ms / 1000.0;
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_MM, .layer = layer, .k = k,
                                  .at = at, .ms = tm->gpu_w2_ms,
                                  .tag = "gw2" });
}

/* A wave is one command buffer with all available experts: w1/w3 for every
 * expert, one barrier, SwiGLU for every expert, one barrier, w2 for every
 * expert. Emit the host phases and the per-expert device stages after the wait,
 * when all timings are known. */
static void trace_gpu_wave(const DSV4ExpertVkWave *w, int layer)
{
    if (!dsv4_trace_active() || !w->valid) return;

    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = -1,
                                  .slot = w->slot, .n = w->n,
                                  .at = w->start_at, .ms = w->copy_in_ms,
                                  .tag = "cin" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = -1,
                                  .slot = w->slot, .n = w->n,
                                  .at = w->start_at + w->copy_in_ms / 1000.0,
                                  .ms = w->record_ms, .tag = "rec" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = -1,
                                  .slot = w->slot, .n = w->n,
                                  .at = w->submit_at - w->submit_ms / 1000.0,
                                  .ms = w->submit_ms, .tag = "sub" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = -1,
                                  .slot = w->slot, .n = w->n,
                                  .at = w->fence_done_at - w->wait_ms / 1000.0,
                                  .ms = w->wait_ms, .tag = "wait" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = -1,
                                  .slot = w->slot, .n = w->n,
                                  .at = w->fence_done_at, .ms = w->copy_out_ms,
                                  .tag = "cout" });
    dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_VK, .layer = layer, .k = -1,
                                  .slot = w->slot, .n = w->n,
                                  .at = w->start_at, .ms = w->total_ms,
                                  .tag = "tot" });

    if (!w->gpu_ts_valid) return;

    for (int j = 0; j < w->n; j++) {
        const DSV4ExpertVkTiming *e = &w->expert[j];
        const int k = w->ks[j];
        double at = w->gpu_anchor_at + e->gpu_start_ms / 1000.0;
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_MM, .layer = layer,
                                      .k = k, .slot = w->slot, .n = w->n,
                                      .at = at, .ms = e->gpu_w1_ms,
                                      .tag = "gw1" });
        at += e->gpu_w1_ms / 1000.0;
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_MM, .layer = layer,
                                      .k = k, .slot = w->slot, .n = w->n,
                                      .at = at, .ms = e->gpu_w3_ms,
                                      .tag = "gw3" });
        at += e->gpu_w3_ms / 1000.0;
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_SWIGLU, .layer = layer,
                                      .k = k, .slot = w->slot, .n = w->n,
                                      .at = at, .ms = e->gpu_swiglu_ms,
                                      .tag = "gswiglu" });
        at += e->gpu_swiglu_ms / 1000.0;
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_MM, .layer = layer,
                                      .k = k, .slot = w->slot, .n = w->n,
                                      .at = at, .ms = e->gpu_w2_ms,
                                      .tag = "gw2" });
    }
}

static void expert_fail(int layer, int k, int topk)
{
    fprintf(stderr, "dsv4_layer: layer %d could not obtain routed expert "
                    "%d of %d; refusing to emit a token routed through fewer "
                    "experts than the model specifies\n", layer, k + 1, topk);
    abort();
}

static int submit_wave_ks(DSV4ExpertVk *gpu, DSV4ExpertVkWave *wave,
                          const int *ks, int n, const DSV4ExpertW **ex,
                          const float *x, DSV4Scratch *s, int timing)
{
    DSV4ExpertVkJob jobs[DSV4_MAX_TOPK];
    for (int j = 0; j < n; j++) {
        const int k = ks[j];
        jobs[j] = (DSV4ExpertVkJob){
            .k = k, .e = ex[k],
            .out = s->expert_acc + (size_t)k * s->expert_acc_stride,
        };
    }
    return dsv4_expert_vk_submit_wave(gpu, jobs, n, x, timing, wave);
}

/* Diagnostic A/B switch for traces: 0/negative means "all available experts".
 * DSV4_VK_WAVE_SIZE=1 reproduces one submit per expert under the same deferred
 * fence model, so the only difference is the wave grouping. */
static int wave_size_limit(void)
{
    static int ready, limit;
    if (!ready) {
        const char *s = getenv("DSV4_VK_WAVE_SIZE");
        limit = s && *s ? atoi(s) : 0;
        ready = 1;
    }
    return limit;
}

/* SwiGLU expert: w2( silu(clamp(w1 x)) * clamp(w3 x) ). */
static void expert_forward(float *out, const float *x, const DSV4ExpertW *e,
                           DSV4Scratch *s, int hidden, int inter, float limit,
                           int layer, int k, DSV4ExpertVk *gpu)
{
    const int tr = dsv4_trace_active();
    double t0 = tr ? dsv4_now_s() : 0.0;

    /* Cache-backed routed experts carry a Vulkan binding. Everything else
     * (the shared expert, synthetic test experts) has vk_base == NULL and
     * stays on the CPU. */
    if (gpu && e->vk_base) {
        DSV4ExpertVkTiming tm;
        const int rc = dsv4_expert_vk_forward_timed(gpu, e, x, out,
                                                    tr ? &tm : NULL);
        if (rc != 0) {
            fprintf(stderr, "dsv4_layer: Vulkan expert matmul failed at layer "
                            "%d k %d; refusing to continue\n", layer, k);
            abort();
        }
        if (tr) {
            if (tm.valid) trace_gpu_breakdown(&tm, layer, k);
            else trace_mm(layer, k, "gpu", t0);
        }
        return;
    }

    dsv4_mmq(s->expert_gate, x, &e->w1);
    if (tr) trace_mm(layer, k, "w1", t0);
    t0 = tr ? dsv4_now_s() : 0.0;

    dsv4_mmq(s->expert_up,   x, &e->w3);
    if (tr) trace_mm(layer, k, "w3", t0);
    t0 = tr ? dsv4_now_s() : 0.0;

    dsv4_swiglu(s->expert_out, s->expert_gate, s->expert_up, inter, limit);
    if (tr)
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_SWIGLU, .layer = layer,
                                      .k = k, .at = t0,
                                      .ms = (dsv4_now_s() - t0) * 1000.0 });
    t0 = tr ? dsv4_now_s() : 0.0;

    dsv4_mmq(out, s->expert_out, &e->w2);
    if (tr) trace_mm(layer, k, "w2", t0);
}

void dsv4_layer_moe(float *out, const float *x, const DSV4LayerW *w,
                    const DSV4Cfg *c, DSV4Scratch *s, const DSV4ExpertSrc *src,
                    int layer, int token_id)
{
    DSV4ExpertVk *gpu = (DSV4ExpertVk *)src->gpu;
    const int tr = dsv4_trace_active();
    const double t_moe = dsv4_now_s();
    double t0 = dsv4_now_s();
    dsv4_mmq(s->gate_scores, x, &w->moe.gate);
    dsv4_route(s->topk_idx, s->topk_w, s->gate_scores, s->gate_orig,
               w->moe.bias, w->moe.tid2eid, token_id,
               c->n_experts, c->topk, c->routed_scale);
    dsv4_prof.gate += dsv4_now_s() - t0;

    for (int i = 0; i < c->hidden; i++) out[i] = 0.0f;

    const DSV4ExpertW *ex[DSV4_MAX_TOPK];
    int done[DSV4_MAX_TOPK];
    const char *tag[DSV4_MAX_TOPK];
    for (int k = 0; k < c->topk; k++) {
        ex[k] = NULL;
        done[k] = 0;
        tag[k] = NULL;
    }

    const int overlapped = (src->begin && src->end);

    t0 = dsv4_now_s();
    if (overlapped)
        src->begin(src->ctx, layer, s->topk_idx, c->topk, ex);
    else
        for (int k = 0; k < c->topk; k++)
            ex[k] = src->get(src->ctx, layer, s->topk_idx[k]);
    dsv4_prof.expert_io += dsv4_now_s() - t0;

    /* Everything available right now goes into one wave: all hits after
     * begin(), or all experts in the non-overlapped get() path. */
    DSV4ExpertVkWave waves[DSV4_MAX_TOPK + 1];
    int nwaves = 0;
    int ks[DSV4_MAX_TOPK], nks = 0;
    for (int k = 0; k < c->topk; k++) {
        if (!ex[k]) continue;
        tag[k] = "hit";
        if (gpu && ex[k]->vk_base) {
            ks[nks++] = k;
            continue;
        }
        trace_expert(DSV4_TR_COMPUTE, layer, k, s->topk_idx[k], "hit");
        t0 = dsv4_now_s();
        expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                       x, ex[k], s, c->hidden, c->moe_inter,
                       c->swiglu_limit, layer, k, gpu);
        dsv4_prof.expert_mm += dsv4_now_s() - t0;
        trace_expert(DSV4_TR_COMPUTE_END, layer, k, s->topk_idx[k], "hit");
        done[k] = 1;
    }
    if (nks) {
        int chunk = wave_size_limit();
        if (chunk <= 0 || chunk > nks) chunk = nks;
        for (int off = 0; off < nks; off += chunk) {
            int cn = nks - off;
            if (cn > chunk) cn = chunk;
            for (int j = 0; j < cn; j++)
                trace_expert(DSV4_TR_COMPUTE, layer, ks[off + j],
                             s->topk_idx[ks[off + j]], "hit");
            t0 = dsv4_now_s();
            const int rc = submit_wave_ks(gpu, &waves[nwaves], ks + off, cn,
                                          ex, x, s, tr);
            dsv4_prof.expert_mm += dsv4_now_s() - t0;
            if (rc != 0) {
                fprintf(stderr, "dsv4_layer: Vulkan wave submit failed at "
                                "layer %d; refusing to continue\n", layer);
                abort();
            }
            for (int j = 0; j < cn; j++) done[ks[off + j]] = 1;
            nwaves++;
        }
    }

    /* The shared expert does not depend on routing, so it fills the window
     * while the misses are read. Its output must not land in expert_acc: the
     * streamed experts overwrite those buffers when they arrive. */
    t0 = dsv4_now_s();
    trace_expert(DSV4_TR_COMPUTE, layer, -1, -1, "shared");
    expert_forward(s->shared_acc, x, &w->moe.shared, s, c->hidden,
                   c->moe_inter * c->n_shared, c->swiglu_limit, layer, -1, gpu);
    trace_expert(DSV4_TR_COMPUTE_END, layer, -1, -1, "shared");
    const double shared_ms = (dsv4_now_s() - t0) * 1000.0;
    dsv4_prof.shared += shared_ms / 1000.0;
    if (tr)
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_SHARED, .layer = layer,
                                      .k = -1, .ms = shared_ms });

    /* A miss whose read has landed is submitted immediately, but nobody waits
     * for its fence here. The k-order sum below is the only consumer. */
    if (overlapped) {
        if (src->wait_any) {
            for (;;) {
                int k = -1;
                t0 = dsv4_now_s();
                const int got = src->wait_any(src->ctx, &k);
                dsv4_prof.expert_io += dsv4_now_s() - t0;  /* time blocked */
                if (!got) break;
                if (k < 0 || k >= c->topk || !ex[k]) continue;
                tag[k] = "miss";
                if (gpu && ex[k]->vk_base) {
                    const int one[1] = { k };
                    trace_expert(DSV4_TR_COMPUTE, layer, k, s->topk_idx[k],
                                 "miss");
                    t0 = dsv4_now_s();
                    const int rc = submit_wave_ks(gpu, &waves[nwaves], one, 1,
                                                  ex, x, s, tr);
                    dsv4_prof.expert_mm += dsv4_now_s() - t0;
                    if (rc != 0) {
                        fprintf(stderr, "dsv4_layer: Vulkan wave submit failed "
                                        "at layer %d; refusing to continue\n",
                                layer);
                        abort();
                    }
                    done[k] = 1;
                    nwaves++;
                } else {
                    trace_expert(DSV4_TR_COMPUTE, layer, k, s->topk_idx[k],
                                 "miss");
                    t0 = dsv4_now_s();
                    expert_forward(s->expert_acc + (size_t)k *
                                       s->expert_acc_stride,
                                   x, ex[k], s, c->hidden, c->moe_inter,
                                   c->swiglu_limit, layer, k, gpu);
                    dsv4_prof.expert_mm += dsv4_now_s() - t0;
                    trace_expert(DSV4_TR_COMPUTE_END, layer, k, s->topk_idx[k],
                                 "miss");
                    done[k] = 1;
                }
            }
        }

        /* Whatever the drive still owes; counted as I/O because that is what it
         * is: this thread blocked with nothing left to overlap. */
        t0 = dsv4_now_s();
        src->end(src->ctx, ex);
        const double end_ms = (dsv4_now_s() - t0) * 1000.0;
        dsv4_prof.expert_io += end_ms / 1000.0;
        if (tr)
            dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_BARRIER, .k = -1,
                                          .layer = layer, .ms = end_ms,
                                          .at = t0 });
    }

    /* Anything the source only returned in end(). Batch those leftovers into
     * one last wave instead of paying another submit per expert. */
    nks = 0;
    for (int k = 0; k < c->topk; k++) {
        if (!ex[k]) expert_fail(layer, k, c->topk);
        if (done[k]) continue;
        tag[k] = "late";
        if (gpu && ex[k]->vk_base) {
            ks[nks++] = k;
            continue;
        }
        trace_expert(DSV4_TR_COMPUTE, layer, k, s->topk_idx[k], "late");
        t0 = dsv4_now_s();
        expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                       x, ex[k], s, c->hidden, c->moe_inter,
                       c->swiglu_limit, layer, k, gpu);
        dsv4_prof.expert_mm += dsv4_now_s() - t0;
        trace_expert(DSV4_TR_COMPUTE_END, layer, k, s->topk_idx[k], "late");
        done[k] = 1;
    }
    if (nks) {
        int chunk = wave_size_limit();
        if (chunk <= 0 || chunk > nks) chunk = nks;
        for (int off = 0; off < nks; off += chunk) {
            int cn = nks - off;
            if (cn > chunk) cn = chunk;
            for (int j = 0; j < cn; j++)
                trace_expert(DSV4_TR_COMPUTE, layer, ks[off + j],
                             s->topk_idx[ks[off + j]], "late");
            t0 = dsv4_now_s();
            const int rc = submit_wave_ks(gpu, &waves[nwaves], ks + off, cn,
                                          ex, x, s, tr);
            dsv4_prof.expert_mm += dsv4_now_s() - t0;
            if (rc != 0) {
                fprintf(stderr, "dsv4_layer: Vulkan wave submit failed at "
                                "layer %d; refusing to continue\n", layer);
                abort();
            }
            for (int j = 0; j < cn; j++) done[ks[off + j]] = 1;
            nwaves++;
        }
    }

    /* Collect every wave only now. The per-k accumulator and the k-order sum
     * below keep the result identical regardless of completion order. */
    for (int i = 0; i < nwaves; i++) {
        t0 = dsv4_now_s();
        if (dsv4_expert_vk_wait_wave(gpu, &waves[i]) != 0) {
            fprintf(stderr, "dsv4_layer: Vulkan wave wait failed at layer %d; "
                            "refusing to continue\n", layer);
            abort();
        }
        dsv4_prof.expert_mm += dsv4_now_s() - t0;
        if (tr) {
            trace_gpu_wave(&waves[i], layer);
            for (int j = 0; j < waves[i].n; j++) {
                const int k = waves[i].ks[j];
                trace_expert(DSV4_TR_COMPUTE_END, layer, k,
                             s->topk_idx[k], tag[k] ? tag[k] : "gpu");
            }
        }
    }

    t0 = dsv4_now_s();
    /* IN k ORDER, ALWAYS. */
    for (int k = 0; k < c->topk; k++) {
        const float *acc = s->expert_acc + (size_t)k * s->expert_acc_stride;
        const float wk = s->topk_w[k];
        for (int i = 0; i < c->hidden; i++) out[i] += wk * acc[i];
    }
    dsv4_prof.expert_mm += dsv4_now_s() - t0;

    /* The shared expert is added UNWEIGHTED, after the routed sum. */
    for (int i = 0; i < c->hidden; i++) out[i] += s->shared_acc[i];

    if (tr)
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_LAYER_END,
                                      .layer = layer, .k = -1,
                                      .ms = (dsv4_now_s() - t_moe) * 1000.0 });
}

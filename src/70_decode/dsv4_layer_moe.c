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

/* SwiGLU expert: w2( silu(clamp(w1 x)) * clamp(w3 x) ). */
static void expert_forward(float *out, const float *x, const DSV4ExpertW *e,
                           DSV4Scratch *s, int hidden, int inter, float limit,
                           int layer, int k)
{
    const int tr = dsv4_trace_active();
    double t0 = tr ? dsv4_now_s() : 0.0;

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
    for (int k = 0; k < c->topk; k++) { ex[k] = NULL; done[k] = 0; }

    const int overlapped = (src->begin && src->end);

    t0 = dsv4_now_s();
    if (overlapped)
        src->begin(src->ctx, layer, s->topk_idx, c->topk, ex);
    else
        for (int k = 0; k < c->topk; k++)
            ex[k] = src->get(src->ctx, layer, s->topk_idx[k]);
    dsv4_prof.expert_io += dsv4_now_s() - t0;

    if (overlapped) {
        /* The residents, while the misses are still on the wire. */
        t0 = dsv4_now_s();
        for (int k = 0; k < c->topk; k++) {
            if (!ex[k]) continue;
            trace_expert(DSV4_TR_COMPUTE, layer, k, s->topk_idx[k], "hit");
            expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                           x, ex[k], s, c->hidden, c->moe_inter,
                           c->swiglu_limit, layer, k);
            trace_expert(DSV4_TR_COMPUTE_END, layer, k, s->topk_idx[k], "hit");
            done[k] = 1;
        }
        dsv4_prof.expert_mm += dsv4_now_s() - t0;

        /* The shared expert does not depend on routing, so it fills the window
         * while the misses are read. Its output must not land in expert_acc:
         * the streamed experts overwrite those buffers when they arrive. */
        t0 = dsv4_now_s();
        trace_expert(DSV4_TR_COMPUTE, layer, -1, -1, "shared");
        expert_forward(s->shared_acc, x, &w->moe.shared, s, c->hidden,
                       c->moe_inter * c->n_shared, c->swiglu_limit, layer, -1);
        trace_expert(DSV4_TR_COMPUTE_END, layer, -1, -1, "shared");
        const double shared_ms = (dsv4_now_s() - t0) * 1000.0;
        dsv4_prof.shared += shared_ms / 1000.0;
        if (dsv4_trace_active())
            dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_SHARED, .layer = layer,
                                          .k = -1, .ms = shared_ms });

        /* A miss whose read has landed can be computed NOW. The per-k
         * accumulator and the k-order sum below keep the result identical. */
        if (src->wait_any) {
            for (;;) {
                int k = -1;
                t0 = dsv4_now_s();
                const int got = src->wait_any(src->ctx, &k);
                dsv4_prof.expert_io += dsv4_now_s() - t0;  /* time blocked */
                if (!got) break;
                if (k < 0 || k >= c->topk || !ex[k]) continue;
                t0 = dsv4_now_s();
                trace_expert(DSV4_TR_COMPUTE, layer, k, s->topk_idx[k], "miss");
                expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                               x, ex[k], s, c->hidden, c->moe_inter,
                               c->swiglu_limit, layer, k);
                trace_expert(DSV4_TR_COMPUTE_END, layer, k, s->topk_idx[k],
                             "miss");
                dsv4_prof.expert_mm += dsv4_now_s() - t0;
                done[k] = 1;
            }
        }

        /* Whatever the drive still owes; counted as I/O because that is what it
         * is: this thread blocked with nothing left to overlap. */
        t0 = dsv4_now_s();
        src->end(src->ctx, ex);
        const double end_ms = (dsv4_now_s() - t0) * 1000.0;
        dsv4_prof.expert_io += end_ms / 1000.0;
        if (dsv4_trace_active())
            dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_BARRIER, .k = -1,
                                          .layer = layer, .ms = end_ms,
                                          .at = t0 });
    } else {
        /* No reads to hide behind: the pre-overlap path, used by the gates. */
        t0 = dsv4_now_s();
        trace_expert(DSV4_TR_COMPUTE, layer, -1, -1, "shared");
        expert_forward(s->shared_acc, x, &w->moe.shared, s, c->hidden,
                       c->moe_inter * c->n_shared, c->swiglu_limit, layer, -1);
        trace_expert(DSV4_TR_COMPUTE_END, layer, -1, -1, "shared");
        const double shms = (dsv4_now_s() - t0) * 1000.0;
        dsv4_prof.shared += shms / 1000.0;
        if (dsv4_trace_active())
            dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_SHARED, .layer = layer,
                                          .k = -1, .ms = shms });
    }

    t0 = dsv4_now_s();
    for (int k = 0; k < c->topk; k++) {
        if (!ex[k]) {
            /* NOT survivable: skipping an expert leaves the token routed
             * through fewer experts than the model specifies, which is a
             * different model producing fluent and wrong output. */
            fprintf(stderr, "dsv4_layer: layer %d could not obtain routed "
                            "expert %d of %d; refusing to emit a token routed "
                            "through fewer experts than the model specifies\n",
                    layer, k + 1, c->topk);
            abort();
        }
        if (!done[k]) {
            trace_expert(DSV4_TR_COMPUTE, layer, k, s->topk_idx[k], "late");
            expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                           x, ex[k], s, c->hidden, c->moe_inter,
                           c->swiglu_limit, layer, k);
            trace_expert(DSV4_TR_COMPUTE_END, layer, k, s->topk_idx[k], "late");
        }
    }
    /* IN k ORDER, ALWAYS. */
    for (int k = 0; k < c->topk; k++) {
        const float *acc = s->expert_acc + (size_t)k * s->expert_acc_stride;
        const float wk = s->topk_w[k];
        for (int i = 0; i < c->hidden; i++) out[i] += wk * acc[i];
    }
    dsv4_prof.expert_mm += dsv4_now_s() - t0;

    /* The shared expert is added UNWEIGHTED, after the routed sum. */
    for (int i = 0; i < c->hidden; i++) out[i] += s->shared_acc[i];

    if (dsv4_trace_active())
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_LAYER_END,
                                      .layer = layer, .k = -1,
                                      .ms = (dsv4_now_s() - t_moe) * 1000.0 });
}

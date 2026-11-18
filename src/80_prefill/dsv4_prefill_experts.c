/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_experts.c - one expert over one token, or over a token bucket. */
#include <string.h>

#include "dsv4.h"
#include "dsv4_prof.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"
#include "dsv4_prefill_vk.h"

/* Below this bucket size the fused GPU pipeline (w1/w3/SwiGLU/w2) and its
 * copies still cost more than the CPU FP4 GEMM. The threshold is empirical;
 * compare the paths with `DSV4_PREFILL_CPU=1 make e2e` and `make e2e`. With
 * the fused kernel the old "three submissions" reason no longer applies, so
 * this is worth re-measuring if small-bucket work becomes the bottleneck. */
#define VK_EXPERT_MIN_COUNT 8

static int pf_bucket_bin(int count)
{
    if (count <= 1)  return 0;
    if (count <= 4)  return 1;
    if (count <= 8)  return 2;
    if (count <= 16) return 3;
    if (count <= 32) return 4;
    if (count <= 64) return 5;
    if (count <= 128) return 6;
    return 7;
}

int dsv4_prefill_expert_is_gpu(const DSV4Prefill *p, int count)
{
    return p && p->vk && count >= VK_EXPERT_MIN_COUNT;
}

/* Routed-expert matmul, timed per matrix so FP4 MM is not confused with the
 * SwiGLU that sits between w1/w3 and w2. gate/up/mid are supplied by the
 * caller: the normal path uses ex_*, the overlap fill uses sm_*, so a GPU
 * bucket and a CPU fill bucket can be live at the same time. */
static void expert_ws(DSV4Prefill *p, float *out, const float *x,
                      const DSV4ExpertW *e, float *gate, float *up,
                      float *mid, int hidden, int inter, float limit,
                      DSV4PFStatsSink *sink)
{
    DSV4PrefillStats *st = sink->st;
    double t0 = dsv4_now_s();
    dsv4_mmq(gate, x, &e->w1);
    dsv4_mmq(up,   x, &e->w3);
    double dt = dsv4_now_s() - t0;
    st->expert_mm += dt;
    if (sink->instr) {
        st->expert_mm_cpu += dt;
        st->bucket_mm_s[0] += dt;
        if (sink->li) sink->li->cpu_mm += dt;
    }

    t0 = dsv4_now_s();
    dsv4_swiglu(mid, gate, up, inter, limit);
    st->expert_swiglu += dsv4_now_s() - t0;

    t0 = dsv4_now_s();
    dsv4_mmq(out, mid, &e->w2);
    dt = dsv4_now_s() - t0;
    st->expert_mm += dt;
    if (sink->instr) {
        st->expert_mm_cpu += dt;
        st->bucket_mm_s[0] += dt;
        if (sink->li) sink->li->cpu_mm += dt;
    }

    /* w1 [inter,hidden], w3 [inter,hidden], w2 [hidden,inter] -> 6*inter*hidden
     * flops per expert forward. */
    st->f_expert += 6.0 * (double)inter * (double)hidden * 1e-9;
}

void dsv4_prefill_expert(DSV4Prefill *p, float *out, const float *x,
                         const DSV4ExpertW *e, int hidden, int inter,
                         float limit, DSV4PFStatsSink *sink)
{
    expert_ws(p, out, x, e, p->ex_gate, p->ex_up, p->ex_mid,
              hidden, inter, limit, sink);
}

/* One expert applied to every token that routed to it: the expert-major seam.
 * count == 1 stays on the GEMV. For count > 1 the activation rows are gathered
 * once, each matrix runs as one count-wide batch, and the result is scattered
 * back into the per-(token, k) accumulators the weighted sum expects. */
void dsv4_prefill_expert_bucket(DSV4Prefill *p, const DSV4ExpertW *ew,
                                int expert, int count, const int *tok,
                                const int *ks, DSV4PrefillVkFillFn fill,
                                void *fill_ctx, DSV4PFStatsSink *sink)
{
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, rinter = c->moe_inter;
    const double tm0 = dsv4_now_s();
    DSV4PrefillLayerInstr *li =
        (p->instr && p->cur_layer >= 0 && p->cur_layer < p->c->n_layers)
            ? &p->linstr[p->cur_layer] : NULL;
    DSV4PFStatsSink live = { &p->st, li, &p->t_moe, p->instr };
    if (!sink) sink = &live;
    DSV4PrefillStats *st = sink->st;
    li = sink->li;
    const int tr = dsv4_trace_active();

    const int bin = pf_bucket_bin(count);
    if (sink->instr) {
        st->buckets_total++;
        st->bucket_hist[bin]++;
        st->bucket_tokens[bin] += count;
        if (li) {
            li->hist[bin]++;
            li->hist_tokens[bin] += count;
        }
    }

    if (count == 1) {
        const int t = tok[0], k = ks[0];
        DSV4PrefillTok *s = &p->scratch[t];
        dsv4_prefill_expert(p, s->expert_acc + (size_t)k * s->expert_acc_stride,
                            p->x1 + (size_t)t * d, ew, d, rinter,
                            c->swiglu_limit, sink);
        if (sink->instr) {
            st->buckets_cpu++;
            st->bucket_s[bin] += dsv4_now_s() - tm0;
            if (li) li->buckets_cpu++;
        }
        if (tr)
            dsv4_trace_ev(&(DSV4TraceEv){
                .kind = DSV4_TR_BUCKET, .layer = p->cur_layer, .k = expert,
                .expert = expert, .n = count,
                .ms = (dsv4_now_s() - tm0) * 1000.0, .at = tm0,
                .tag = "cpu" });
        *sink->t_moe += dsv4_now_s() - tm0;
        return;
    }

    double fill_s = 0.0;
    double t0 = dsv4_now_s();
    for (int j = 0; j < count; j++) {
        const int t = tok[j];
        memcpy(p->ex_x + (size_t)j * d, p->x1 + (size_t)t * d,
               (size_t)d * sizeof(float));
    }
    st->expert_pack += dsv4_now_s() - t0;

    const int gpu_tried = dsv4_prefill_expert_is_gpu(p, count);
    int gpu = 0;
    t0 = dsv4_now_s();
    if (gpu_tried) {
        gpu = dsv4_prefill_vk_expert_fused(p->vk, p->ex_out, p->ex_x,
                                           &ew->w1, &ew->w3, &ew->w2,
                                           count, rinter, c->swiglu_limit,
                                           fill, fill_ctx, &fill_s) == 0;
    }
    const double gpu_wall = gpu_tried ? dsv4_now_s() - t0 : 0.0;
    const double gpu_sec = gpu_wall > fill_s ? gpu_wall - fill_s : 0.0;
    double cpu_sec = 0.0;
    if (!gpu) {
        t0 = dsv4_now_s();
        dsv4_mmq_n(p->ex_gate, p->ex_x, &ew->w1, count);
        dsv4_mmq_n(p->ex_up,   p->ex_x, &ew->w3, count);
        cpu_sec += dsv4_now_s() - t0;

        t0 = dsv4_now_s();
        for (int j = 0; j < count; j++)
            dsv4_swiglu(p->ex_mid + (size_t)j * rinter,
                        p->ex_gate + (size_t)j * rinter,
                        p->ex_up + (size_t)j * rinter, rinter,
                        c->swiglu_limit);
        st->expert_swiglu += dsv4_now_s() - t0;

        t0 = dsv4_now_s();
        dsv4_mmq_n(p->ex_out, p->ex_mid, &ew->w2, count);
        cpu_sec += dsv4_now_s() - t0;
    }
    st->expert_mm += gpu_sec + cpu_sec;
    if (sink->instr) {
        st->expert_mm_gpu += gpu_sec;
        st->expert_mm_cpu += cpu_sec;
        st->bucket_mm_s[bin] += gpu_sec + cpu_sec;
        if (li) { li->gpu_mm += gpu_sec; li->cpu_mm += cpu_sec; }
        if (gpu_tried && !gpu) st->gpu_fallbacks++;
    }

    t0 = dsv4_now_s();
    for (int j = 0; j < count; j++) {
        const int t = tok[j], k = ks[j];
        DSV4PrefillTok *s = &p->scratch[t];
        memcpy(s->expert_acc + (size_t)k * s->expert_acc_stride,
               p->ex_out + (size_t)j * d, (size_t)d * sizeof(float));
    }
    st->expert_pack += dsv4_now_s() - t0;

    if (sink->instr) {
        if (gpu) {
            st->buckets_gpu++;
            if (li) li->buckets_gpu++;
        } else {
            st->buckets_cpu++;
            if (li) li->buckets_cpu++;
        }
        st->bucket_s[bin] += (dsv4_now_s() - tm0) - fill_s;
    }
    if (tr)
        dsv4_trace_ev(&(DSV4TraceEv){
            .kind = DSV4_TR_BUCKET, .layer = p->cur_layer, .k = expert,
            .expert = expert, .n = count,
            .ms = (dsv4_now_s() - tm0) * 1000.0, .at = tm0,
            .tag = gpu ? "gpu" : "cpu" });

    st->f_expert += 6.0 * (double)rinter * (double)d *
                    (double)count * 1e-9;
    *sink->t_moe += (dsv4_now_s() - tm0) - fill_s;
}

/* CPU-only fill bucket. It must never touch ex_* or the Vulkan buffers: a GPU
 * bucket owns ex_* for the whole submit/wait sequence. */
void dsv4_prefill_expert_bucket_fill(DSV4Prefill *p, const DSV4ExpertW *ew,
                                     int expert, int count, const int *tok,
                                     const int *ks, DSV4PFStatsSink *sink,
                                     const char *tag)
{
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, rinter = c->moe_inter;
    const double tm0 = dsv4_now_s();
    DSV4PrefillStats *st = sink->st;
    DSV4PrefillLayerInstr *li = sink->li;
    const int tr = dsv4_trace_active();
    const int bin = pf_bucket_bin(count);

    if (sink->instr) {
        st->buckets_total++;
        st->fill_buckets++;
        st->bucket_hist[bin]++;
        st->bucket_tokens[bin] += count;
        if (li) {
            li->hist[bin]++;
            li->hist_tokens[bin] += count;
            li->fill_buckets++;
        }
    }

    if (count == 1) {
        const int t = tok[0], k = ks[0];
        DSV4PrefillTok *s = &p->scratch[t];
        expert_ws(p, s->expert_acc + (size_t)k * s->expert_acc_stride,
                  p->x1 + (size_t)t * d, ew, p->sm_gate, p->sm_up,
                  p->sm_mid, d, rinter, c->swiglu_limit, sink);
    } else {
        double t0 = dsv4_now_s();
        for (int j = 0; j < count; j++) {
            const int t = tok[j];
            memcpy(p->sm_x + (size_t)j * d, p->x1 + (size_t)t * d,
                   (size_t)d * sizeof(float));
        }
        st->expert_pack += dsv4_now_s() - t0;

        t0 = dsv4_now_s();
        dsv4_mmq_n(p->sm_gate, p->sm_x, &ew->w1, count);
        dsv4_mmq_n(p->sm_up,   p->sm_x, &ew->w3, count);
        const double mm1 = dsv4_now_s() - t0;
        st->expert_mm += mm1;
        if (sink->instr) {
            st->expert_mm_cpu += mm1;
            st->bucket_mm_s[bin] += mm1;
            if (li) li->cpu_mm += mm1;
        }

        t0 = dsv4_now_s();
        for (int j = 0; j < count; j++)
            dsv4_swiglu(p->sm_mid + (size_t)j * rinter,
                        p->sm_gate + (size_t)j * rinter,
                        p->sm_up + (size_t)j * rinter, rinter,
                        c->swiglu_limit);
        st->expert_swiglu += dsv4_now_s() - t0;

        t0 = dsv4_now_s();
        dsv4_mmq_n(p->sm_out, p->sm_mid, &ew->w2, count);
        const double mm2 = dsv4_now_s() - t0;
        st->expert_mm += mm2;
        if (sink->instr) {
            st->expert_mm_cpu += mm2;
            st->bucket_mm_s[bin] += mm2;
            if (li) li->cpu_mm += mm2;
        }

        t0 = dsv4_now_s();
        for (int j = 0; j < count; j++) {
            const int t = tok[j], k = ks[j];
            DSV4PrefillTok *s = &p->scratch[t];
            memcpy(s->expert_acc + (size_t)k * s->expert_acc_stride,
                   p->sm_out + (size_t)j * d, (size_t)d * sizeof(float));
        }
        st->expert_pack += dsv4_now_s() - t0;
    }

    if (sink->instr) {
        st->buckets_cpu++;
        if (li) li->buckets_cpu++;
        st->bucket_s[bin] += dsv4_now_s() - tm0;
    }
    if (count > 1)
        st->f_expert += 6.0 * (double)rinter * (double)d *
                        (double)count * 1e-9;
    if (tr)
        dsv4_trace_ev(&(DSV4TraceEv){
            .kind = DSV4_TR_BUCKET, .layer = p->cur_layer, .k = expert,
            .expert = expert, .n = count,
            .ms = (dsv4_now_s() - tm0) * 1000.0, .at = tm0, .tag = tag });
    *sink->t_moe += dsv4_now_s() - tm0;
}

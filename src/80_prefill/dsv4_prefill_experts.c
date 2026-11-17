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

/* Below this bucket size the three GPU submissions and their copies cost more
 * than the CPU FP4 GEMM. The threshold is empirical and leaves room for the
 * synchronous submit/fence overhead; compare the paths with
 * `DSV4_PREFILL_CPU=1 make e2e` and `make e2e`. */
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

/* Routed-expert matmul, timed per matrix so FP4 MM is not confused with the
 * SwiGLU that sits between w1/w3 and w2. */
void dsv4_prefill_expert(DSV4Prefill *p, float *out, const float *x,
                         const DSV4ExpertW *e, int hidden, int inter,
                         float limit)
{
    /* The count==1 path runs one token at a time, so it borrows row 0 of the
     * block-wide expert work buffers instead of making every token carry its
     * own three moe_inter-sized temporaries. */
    float *gate = p->ex_gate;
    float *up   = p->ex_up;
    float *mid  = p->ex_mid;

    double t0 = dsv4_now_s();
    dsv4_mmq(gate, x, &e->w1);
    dsv4_mmq(up,   x, &e->w3);
    double dt = dsv4_now_s() - t0;
    p->st.expert_mm += dt;
    if (p->instr) {
        p->st.expert_mm_cpu += dt;
        p->st.bucket_mm_s[0] += dt;
        if (p->cur_layer >= 0 && p->cur_layer < p->c->n_layers)
            p->linstr[p->cur_layer].cpu_mm += dt;
    }

    t0 = dsv4_now_s();
    dsv4_swiglu(mid, gate, up, inter, limit);
    p->st.expert_swiglu += dsv4_now_s() - t0;

    t0 = dsv4_now_s();
    dsv4_mmq(out, mid, &e->w2);
    dt = dsv4_now_s() - t0;
    p->st.expert_mm += dt;
    if (p->instr) {
        p->st.expert_mm_cpu += dt;
        p->st.bucket_mm_s[0] += dt;
        if (p->cur_layer >= 0 && p->cur_layer < p->c->n_layers)
            p->linstr[p->cur_layer].cpu_mm += dt;
    }

    /* w1 [inter,hidden], w3 [inter,hidden], w2 [hidden,inter] -> 6*inter*hidden
     * flops per expert forward. */
    p->st.f_expert += 6.0 * (double)inter * (double)hidden * 1e-9;
}

/* One expert applied to every token that routed to it: the expert-major seam.
 * count == 1 stays on the GEMV. For count > 1 the activation rows are gathered
 * once, each matrix runs as one count-wide batch, and the result is scattered
 * back into the per-(token, k) accumulators the weighted sum expects. */
void dsv4_prefill_expert_bucket(DSV4Prefill *p, const DSV4ExpertW *ew,
                                int count, const int *tok, const int *ks)
{
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, rinter = c->moe_inter;
    const double tm0 = dsv4_now_s();
    DSV4PrefillLayerInstr *li =
        (p->instr && p->cur_layer >= 0 && p->cur_layer < p->c->n_layers)
            ? &p->linstr[p->cur_layer] : NULL;

    const int bin = pf_bucket_bin(count);
    if (p->instr) {
        p->st.buckets_total++;
        p->st.bucket_hist[bin]++;
        p->st.bucket_tokens[bin] += count;
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
                            c->swiglu_limit);
        if (p->instr) {
            p->st.buckets_cpu++;
            p->st.bucket_s[bin] += dsv4_now_s() - tm0;
            if (li) li->buckets_cpu++;
        }
        p->t_moe += dsv4_now_s() - tm0;
        return;
    }

    double t0 = dsv4_now_s();
    for (int j = 0; j < count; j++) {
        const int t = tok[j];
        memcpy(p->ex_x + (size_t)j * d, p->x1 + (size_t)t * d,
               (size_t)d * sizeof(float));
    }
    p->st.expert_pack += dsv4_now_s() - t0;

    const int gpu_tried = (p->vk && count >= VK_EXPERT_MIN_COUNT);
    int gpu = 0;
    t0 = dsv4_now_s();
    if (gpu_tried) {
        gpu = dsv4_prefill_vk_gemm(p->vk, p->ex_gate, p->ex_x,
                                   &ew->w1, count, DSV4_VK_KIND_EXPERT) == 0
           && dsv4_prefill_vk_gemm(p->vk, p->ex_up, p->ex_x,
                                   &ew->w3, count, DSV4_VK_KIND_EXPERT) == 0;
    }
    const double gpu_sec = gpu_tried ? dsv4_now_s() - t0 : 0.0;
    double cpu_sec = 0.0;
    if (!gpu) {
        t0 = dsv4_now_s();
        dsv4_mmq_n(p->ex_gate, p->ex_x, &ew->w1, count);
        dsv4_mmq_n(p->ex_up,   p->ex_x, &ew->w3, count);
        cpu_sec += dsv4_now_s() - t0;
    }
    p->st.expert_mm += gpu_sec + cpu_sec;
    if (p->instr) {
        p->st.expert_mm_gpu += gpu_sec;
        p->st.expert_mm_cpu += cpu_sec;
        if (li) { li->gpu_mm += gpu_sec; li->cpu_mm += cpu_sec; }
        if (gpu_tried && !gpu) p->st.gpu_fallbacks++;
    }

    t0 = dsv4_now_s();
    for (int j = 0; j < count; j++)
        dsv4_swiglu(p->ex_mid + (size_t)j * rinter,
                    p->ex_gate + (size_t)j * rinter,
                    p->ex_up + (size_t)j * rinter, rinter,
                    c->swiglu_limit);
    p->st.expert_swiglu += dsv4_now_s() - t0;

    const int w2_tried = (p->vk && count >= VK_EXPERT_MIN_COUNT);
    int w2_gpu = 0;
    t0 = dsv4_now_s();
    if (w2_tried)
        w2_gpu = dsv4_prefill_vk_gemm(p->vk, p->ex_out, p->ex_mid,
                                      &ew->w2, count,
                                      DSV4_VK_KIND_EXPERT) == 0;
    const double w2_gpu_sec = w2_tried ? dsv4_now_s() - t0 : 0.0;
    double w2_cpu_sec = 0.0;
    if (!w2_gpu) {
        t0 = dsv4_now_s();
        dsv4_mmq_n(p->ex_out, p->ex_mid, &ew->w2, count);
        w2_cpu_sec = dsv4_now_s() - t0;
    }
    p->st.expert_mm += w2_gpu_sec + w2_cpu_sec;
    if (p->instr) {
        p->st.expert_mm_gpu += w2_gpu_sec;
        p->st.expert_mm_cpu += w2_cpu_sec;
        p->st.bucket_mm_s[bin] += gpu_sec + cpu_sec + w2_gpu_sec + w2_cpu_sec;
        if (li) { li->gpu_mm += w2_gpu_sec; li->cpu_mm += w2_cpu_sec; }
        if (w2_tried && !w2_gpu) p->st.gpu_fallbacks++;
    }

    t0 = dsv4_now_s();
    for (int j = 0; j < count; j++) {
        const int t = tok[j], k = ks[j];
        DSV4PrefillTok *s = &p->scratch[t];
        memcpy(s->expert_acc + (size_t)k * s->expert_acc_stride,
               p->ex_out + (size_t)j * d, (size_t)d * sizeof(float));
    }
    p->st.expert_pack += dsv4_now_s() - t0;

    if (p->instr) {
        const int any_gpu = gpu || w2_gpu;
        if (any_gpu) {
            p->st.buckets_gpu++;
            if (li) li->buckets_gpu++;
        } else {
            p->st.buckets_cpu++;
            if (li) li->buckets_cpu++;
        }
        p->st.bucket_s[bin] += dsv4_now_s() - tm0;
    }

    p->st.f_expert += 6.0 * (double)rinter * (double)d *
                      (double)count * 1e-9;
    p->t_moe += dsv4_now_s() - tm0;
}

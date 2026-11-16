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
 * than the CPU FP4 GEMM. Measured break-even is ~6 tokens/expert; 8 leaves
 * room for the synchronous submit/fence overhead. */
#define VK_EXPERT_MIN_COUNT 8

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
    p->st.expert_mm += dsv4_now_s() - t0;

    t0 = dsv4_now_s();
    dsv4_swiglu(mid, gate, up, inter, limit);
    p->st.expert_swiglu += dsv4_now_s() - t0;

    t0 = dsv4_now_s();
    dsv4_mmq(out, mid, &e->w2);
    p->st.expert_mm += dsv4_now_s() - t0;

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

    if (count == 1) {
        const int t = tok[0], k = ks[0];
        DSV4PrefillTok *s = &p->scratch[t];
        dsv4_prefill_expert(p, s->expert_acc + (size_t)k * s->expert_acc_stride,
                            p->x1 + (size_t)t * d, ew, d, rinter,
                            c->swiglu_limit);
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

    t0 = dsv4_now_s();
    int gpu = 0;
    if (p->vk && count >= VK_EXPERT_MIN_COUNT) {
        gpu = dsv4_prefill_vk_gemm(p->vk, p->ex_gate, p->ex_x,
                                   &ew->w1, count) == 0
           && dsv4_prefill_vk_gemm(p->vk, p->ex_up, p->ex_x,
                                   &ew->w3, count) == 0;
    }
    if (!gpu) {
        dsv4_mmq_n(p->ex_gate, p->ex_x, &ew->w1, count);
        dsv4_mmq_n(p->ex_up,   p->ex_x, &ew->w3, count);
    }
    p->st.expert_mm += dsv4_now_s() - t0;

    t0 = dsv4_now_s();
    for (int j = 0; j < count; j++)
        dsv4_swiglu(p->ex_mid + (size_t)j * rinter,
                    p->ex_gate + (size_t)j * rinter,
                    p->ex_up + (size_t)j * rinter, rinter,
                    c->swiglu_limit);
    p->st.expert_swiglu += dsv4_now_s() - t0;

    t0 = dsv4_now_s();
    if (!(p->vk && count >= VK_EXPERT_MIN_COUNT
          && dsv4_prefill_vk_gemm(p->vk, p->ex_out, p->ex_mid,
                                  &ew->w2, count) == 0))
        dsv4_mmq_n(p->ex_out, p->ex_mid, &ew->w2, count);
    p->st.expert_mm += dsv4_now_s() - t0;

    t0 = dsv4_now_s();
    for (int j = 0; j < count; j++) {
        const int t = tok[j], k = ks[j];
        DSV4PrefillTok *s = &p->scratch[t];
        memcpy(s->expert_acc + (size_t)k * s->expert_acc_stride,
               p->ex_out + (size_t)j * d, (size_t)d * sizeof(float));
    }
    p->st.expert_pack += dsv4_now_s() - t0;

    p->st.f_expert += 6.0 * (double)rinter * (double)d *
                      (double)count * 1e-9;
    p->t_moe += dsv4_now_s() - tm0;
}

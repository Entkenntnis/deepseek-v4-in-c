/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_experts.c - one expert over one token, or over a token bucket. */
#include <string.h>

#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"
#include "dsv4_prefill_vk.h"

int dsv4_prefill_expert_is_gpu(const DSV4Prefill *p, int count)
{
    return p && p->vk && count >= DSV4_PF_GPU_ALWAYS_MIN;
}

/* One expert applied to every token that routed to it: the expert-major seam.
 * count == 1 stays on the CPU GEMV unless the scheduler explicitly forces the
 * GPU. For count > 1 the activation rows are gathered once, each matrix runs
 * as one count-wide batch, and the result is scattered back into the
 * per-(token, k) accumulators the weighted sum expects. `route` lets the union
 * scheduler place a contested 5..8 bucket on either backend. */
void dsv4_prefill_expert_bucket(DSV4Prefill *p, const DSV4ExpertW *ew,
                                int expert, int count, const int *tok,
                                const int *ks, int route,
                                DSV4PrefillVkFillFn fill, void *fill_ctx)
{
    (void)expert;
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, rinter = c->moe_inter;

    const int gpu_wanted =
        route == DSV4_EXPERT_ROUTE_GPU ? (p->vk != NULL)
      : route == DSV4_EXPERT_ROUTE_CPU ? 0
      : dsv4_prefill_expert_is_gpu(p, count);

    if (count == 1 && !gpu_wanted) {
        const int t = tok[0], k = ks[0];
        DSV4PrefillTok *s = &p->scratch[t];
        dsv4_expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                            p->x1 + (size_t)t * d, ew,
                            p->ex_gate, p->ex_up, p->ex_mid, rinter,
                            c->swiglu_limit);
        return;
    }

    for (int j = 0; j < count; j++) {
        const int t = tok[j];
        memcpy(p->ex_x + (size_t)j * d, p->x1 + (size_t)t * d,
               (size_t)d * sizeof(float));
    }

    int gpu = 0;
    if (gpu_wanted) {
        gpu = dsv4_prefill_vk_expert_fused(p->vk, p->ex_out, p->ex_x,
                                           &ew->w1, &ew->w3, &ew->w2,
                                           count, rinter, c->swiglu_limit,
                                           fill, fill_ctx) == 0;
    }
    if (!gpu) {
        dsv4_mmq_n(p->ex_gate, p->ex_x, &ew->w1, count);
        dsv4_mmq_n(p->ex_up,   p->ex_x, &ew->w3, count);
        for (int j = 0; j < count; j++)
            dsv4_swiglu(p->ex_mid + (size_t)j * rinter,
                        p->ex_gate + (size_t)j * rinter,
                        p->ex_up + (size_t)j * rinter, rinter,
                        c->swiglu_limit);
        dsv4_mmq_n(p->ex_out, p->ex_mid, &ew->w2, count);
    }

    for (int j = 0; j < count; j++) {
        const int t = tok[j], k = ks[j];
        DSV4PrefillTok *s = &p->scratch[t];
        memcpy(s->expert_acc + (size_t)k * s->expert_acc_stride,
               p->ex_out + (size_t)j * d, (size_t)d * sizeof(float));
    }
}

/* CPU-only fill bucket. It must never touch ex_* or the Vulkan buffers: a GPU
 * bucket owns ex_* for the whole submit/wait sequence. */
void dsv4_prefill_expert_bucket_fill(DSV4Prefill *p, const DSV4ExpertW *ew,
                                     int expert, int count, const int *tok,
                                     const int *ks)
{
    (void)expert;
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, rinter = c->moe_inter;

    if (count == 1) {
        const int t = tok[0], k = ks[0];
        DSV4PrefillTok *s = &p->scratch[t];
        dsv4_expert_forward(s->expert_acc + (size_t)k * s->expert_acc_stride,
                            p->x1 + (size_t)t * d, ew,
                            p->sm_gate, p->sm_up, p->sm_mid, rinter,
                            c->swiglu_limit);
    } else {
        for (int j = 0; j < count; j++) {
            const int t = tok[j];
            memcpy(p->sm_x + (size_t)j * d, p->x1 + (size_t)t * d,
                   (size_t)d * sizeof(float));
        }
        dsv4_mmq_n(p->sm_gate, p->sm_x, &ew->w1, count);
        dsv4_mmq_n(p->sm_up,   p->sm_x, &ew->w3, count);
        for (int j = 0; j < count; j++)
            dsv4_swiglu(p->sm_mid + (size_t)j * rinter,
                        p->sm_gate + (size_t)j * rinter,
                        p->sm_up + (size_t)j * rinter, rinter,
                        c->swiglu_limit);
        dsv4_mmq_n(p->sm_out, p->sm_mid, &ew->w2, count);

        for (int j = 0; j < count; j++) {
            const int t = tok[j], k = ks[j];
            DSV4PrefillTok *s = &p->scratch[t];
            memcpy(s->expert_acc + (size_t)k * s->expert_acc_stride,
                   p->sm_out + (size_t)j * d, (size_t)d * sizeof(float));
        }
    }
}

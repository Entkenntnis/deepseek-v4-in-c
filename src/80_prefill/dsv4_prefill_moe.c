/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_moe.c - router, shared expert, then routed experts. */
#include <stdio.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_prof.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

int dsv4_prefill_moe(DSV4Prefill *p, int L, const DSV4LayerW *w,
                     int n, const int32_t *ids)
{
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const int inter = c->moe_inter * c->n_shared;
    const int rinter = c->moe_inter;
    const int topk = c->topk;

    const double tg = dsv4_now_s();
    if (dsv4_prefill_pm_stats(p, p->gate, p->x1, &w->moe.gate, n,
                              &p->st.gate_mm, &p->st.f_gate)) return -1;
    {
        const double tr = dsv4_now_s();
        for (int t = 0; t < n; t++) {
            dsv4_route(p->topk_idx + (size_t)t * topk,
                       p->topk_w + (size_t)t * topk,
                       p->gate + (size_t)t * c->n_experts,
                       p->gate_orig + (size_t)t * c->n_experts,
                       w->moe.bias, w->moe.tid2eid, ids[t],
                       c->n_experts, topk, c->routed_scale);
        }
        p->st.route += dsv4_now_s() - tr;
    }
    p->t_gate += dsv4_now_s() - tg;

    const double tsh = dsv4_now_s();
    {
        DSV4PMJob two[2] = {
            { &w->moe.shared.w1, p->x1, p->sh_gate, n },
            { &w->moe.shared.w3, p->x1, p->sh_up, n },
        };
        if (dsv4_prefill_pm_multi_stats(p, two, 2,
                                        &p->st.shared_mm,
                                        &p->st.f_shared)) return -1;
    }
    {
        const double ts = dsv4_now_s();
        for (int t = 0; t < n; t++)
            dsv4_swiglu(p->sh_mid + (size_t)t * inter,
                        p->sh_gate + (size_t)t * inter,
                        p->sh_up + (size_t)t * inter, inter,
                        c->swiglu_limit);
        p->st.shared_swiglu += dsv4_now_s() - ts;
    }
    if (dsv4_prefill_pm_stats(p, p->sh_out, p->sh_mid, &w->moe.shared.w2, n,
                              &p->st.shared_mm, &p->st.f_shared)) return -1;
    p->t_shared += dsv4_now_s() - tsh;

    /* ROUTED EXPERTS: layer-level union and expert-major when the source can
     * overlap; otherwise the per-token path. Both write the same per-(token, k)
     * accumulators, so the arithmetic is unchanged. */
    if (p->src->begin && p->src->end) {
        if (dsv4_prefill_moe_union(p, L, n)) return -1;
    } else {
        for (int t = 0; t < n; t++) {
            DSV4Scratch *s = &p->scratch[t];
            const float *x = p->x1 + (size_t)t * d;
            float *o = p->out + (size_t)t * d;
            const int *tidx = p->topk_idx + (size_t)t * topk;
            const float *tw = p->topk_w + (size_t)t * topk;
            const DSV4ExpertW *ex[DSV4_MAX_TOPK];
            for (int k = 0; k < topk; k++) ex[k] = NULL;

            double ti0 = dsv4_now_s();
            if (p->src->get_many)
                p->src->get_many(p->src->ctx, L, tidx, topk, ex);
            else
                for (int k = 0; k < topk; k++)
                    ex[k] = p->src->get(p->src->ctx, L, tidx[k]);
            {
                const double td = dsv4_now_s() - ti0;
                p->t_io += td;
                p->st.expert_io += td;
            }

            const double tm0 = dsv4_now_s();
            for (int k = 0; k < topk; k++) {
                if (!ex[k]) {
                    fprintf(stderr, "prefill: layer %d could not obtain routed "
                                    "expert %d of %d for token %d; refusing to "
                                    "emit a token routed through fewer experts\n",
                            L, k + 1, topk, t);
                    return -1;
                }
                dsv4_prefill_expert(p, s->expert_acc +
                                       (size_t)k * s->expert_acc_stride,
                                    x, ex[k], s, d, rinter, c->swiglu_limit);
            }
            {
                const double tsum = dsv4_now_s();
                for (int i = 0; i < d; i++) o[i] = 0.0f;
                for (int k = 0; k < topk; k++) {
                    const float wk = tw[k];
                    const float *acc = s->expert_acc +
                                       (size_t)k * s->expert_acc_stride;
                    for (int i = 0; i < d; i++) o[i] += wk * acc[i];
                }
                for (int i = 0; i < d; i++) o[i] += p->sh_out[(size_t)t * d + i];
                p->st.expert_sum += dsv4_now_s() - tsum;
            }
            p->t_moe += dsv4_now_s() - tm0;
        }
    }

    {
        const double ts = dsv4_now_s();
        for (int t = 0; t < n; t++)
            dsv4_hc_post(p->H + (size_t)t * (size_t)hc * d,
                         p->out + (size_t)t * d,
                         p->resid + (size_t)t * (size_t)hc * d,
                         p->post + (size_t)t * hc,
                         p->comb + (size_t)t * hc * hc, hc, d);
        p->st.hc_post_moe += dsv4_now_s() - ts;
    }
    return 0;
}

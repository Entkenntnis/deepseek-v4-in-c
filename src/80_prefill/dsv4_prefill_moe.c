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
    const int topk = c->topk;

    if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);
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

    /* Routing is known now. Start the whole layer's expert reads before the
     * shared expert so the reader pool fills slots while the GPU runs w1/w3,
     * swiglu and w2. Hash layers already started theirs at the top of
     * prefill_layer(), where no gate matmul was needed to know the experts. */
    if (!(p->pf_active && p->pf_layer == L)) {
        const int prc = dsv4_prefill_moe_prefetch(p, L, n);
        if (prc) return prc;
    }

    const double tsh = dsv4_now_s();
    {
        DSV4PMJob two[2] = {
            { &w->moe.shared.w1, p->x1, p->sh_gate, n },
            { &w->moe.shared.w3, p->x1, p->sh_up, n },
        };
        if (dsv4_prefill_pm_multi_stats(p, two, 2,
                                        &p->st.shared_mm,
                                        &p->st.f_shared)) {
            dsv4_prefill_moe_abort(p);
            return -1;
        }
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
                              &p->st.shared_mm, &p->st.f_shared)) {
        dsv4_prefill_moe_abort(p);
        return -1;
    }
    p->t_shared += dsv4_now_s() - tsh;

    if (dsv4_prefill_cancelled(p)) {
        dsv4_prefill_moe_abort(p);
        return dsv4_prefill_cancel_code(p);
    }

    /* ROUTED EXPERTS: layer-level union and expert-major. begin()/end() are
     * required by dsv4_prefill_init(), so this is the only path. */
    const int urc = dsv4_prefill_moe_union(p, L, n);
    if (urc) return urc;

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

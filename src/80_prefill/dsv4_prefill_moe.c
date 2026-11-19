/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_moe.c - router, shared expert, then routed experts. */
#include <stdio.h>
#include <string.h>

#include "dsv4.h"
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
    if (dsv4_prefill_pm(p, p->gate, p->x1, &w->moe.gate, n)) return -1;
    for (int t = 0; t < n; t++) {
        dsv4_route(p->topk_idx + (size_t)t * topk,
                   p->topk_w + (size_t)t * topk,
                   p->gate + (size_t)t * c->n_experts,
                   p->gate_orig + (size_t)t * c->n_experts,
                   w->moe.bias, w->moe.tid2eid, ids[t],
                   c->n_experts, topk, c->routed_scale);
    }

    /* Routing is known now. Start the whole layer's expert reads before the
     * shared expert so the reader pool fills slots while the GPU runs w1/w3,
     * swiglu and w2. Hash layers already started theirs at the top of
     * prefill_layer(), where no gate matmul was needed to know the experts. */
    if (!(p->pf_active && p->pf_layer == L)) {
        const int prc = dsv4_prefill_moe_prefetch(p, L, n);
        if (prc) return prc;
    }

    {
        DSV4PMJob two[2] = {
            { &w->moe.shared.w1, p->x1, p->sh_gate, n },
            { &w->moe.shared.w3, p->x1, p->sh_up, n },
        };
        if (dsv4_prefill_pm_multi(p, two, 2)) {
            dsv4_prefill_moe_abort(p);
            return -1;
        }
    }
    for (int t = 0; t < n; t++)
        dsv4_swiglu(p->sh_mid + (size_t)t * inter,
                    p->sh_gate + (size_t)t * inter,
                    p->sh_up + (size_t)t * inter, inter,
                    c->swiglu_limit);
    if (dsv4_prefill_pm(p, p->sh_out, p->sh_mid, &w->moe.shared.w2, n)) {
        dsv4_prefill_moe_abort(p);
        return -1;
    }

    if (dsv4_prefill_cancelled(p)) {
        dsv4_prefill_moe_abort(p);
        return dsv4_prefill_cancel_code(p);
    }

    /* ROUTED EXPERTS: layer-level union and expert-major. begin()/end() are
     * required by dsv4_prefill_init(), so this is the only path. */
    const int urc = dsv4_prefill_moe_union(p, L, n);
    if (urc) return urc;

    for (int t = 0; t < n; t++)
        dsv4_hc_post(p->H + (size_t)t * (size_t)hc * d,
                     p->out + (size_t)t * d,
                     p->resid + (size_t)t * (size_t)hc * d,
                     p->post + (size_t)t * hc,
                     p->comb + (size_t)t * hc * hc, hc, d);
    return 0;
}

int dsv4_prefill_oproj(DSV4Prefill *p, const DSV4LayerW *w, int n)
{
    if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const int H = c->n_heads, hd = c->head_dim;

    /* Grouped low-rank output projection, one GEMM per group. */
    const int gw = H * hd / c->o_groups;
    const int gr = c->o_lora;
    if (c->o_groups > 16) return -1;
    DSV4QMat slices[16];
    DSV4PMJob jobs[16];
    for (int g = 0; g < c->o_groups; g++) {
        for (int t = 0; t < n; t++)
            memcpy(p->xg + ((size_t)g * p->cap + t) * gw,
                   p->o + (size_t)t * (size_t)H * hd + (size_t)g * gw,
                   (size_t)gw * sizeof(float));
        slices[g] = dsv4_wo_a_group(&w->attn, g, c->o_groups, gr, H * hd);
        jobs[g].m = &slices[g];
        jobs[g].x = p->xg + (size_t)g * p->cap * gw;
        jobs[g].y = p->yg + (size_t)g * p->cap * gr;
        jobs[g].n = n;
    }
    if (dsv4_prefill_pm_multi(p, jobs, c->o_groups)) return -1;
    for (int g = 0; g < c->o_groups; g++)
        for (int t = 0; t < n; t++)
            memcpy(p->ogrp + (size_t)t * (size_t)c->o_groups * c->o_lora +
                       (size_t)g * gr,
                   p->yg + (size_t)g * p->cap * gr + (size_t)t * gr,
                   (size_t)gr * sizeof(float));
    if (dsv4_prefill_pm(p, p->out, p->ogrp, &w->attn.wo_b, n)) return -1;
    for (int t = 0; t < n; t++)
        dsv4_hc_post(p->H + (size_t)t * (size_t)hc * d,
                     p->out + (size_t)t * d,
                     p->resid + (size_t)t * (size_t)hc * d,
                     p->post + (size_t)t * hc,
                     p->comb + (size_t)t * hc * hc, hc, d);
    return 0;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_oproj.c - grouped low-rank output projection, then hc_post. */
#include <math.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_prof.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

int dsv4_prefill_oproj(DSV4Prefill *p, const DSV4LayerW *w, int n)
{
    if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const int H = c->n_heads, hd = c->head_dim;

    /* Grouped low-rank output projection, one GEMM per group. */
    const int gw = H * hd / c->o_groups;
    const int gr = c->o_lora;
    const size_t esz = (w->attn.wo_a.wdt == DSV4_WBF16) ? 2u : 1u;
    const size_t row_bytes = (w->attn.wo_a.wdt == DSV4_WFP4)
                           ? (size_t)gw / 2u : (size_t)gw * esz;
    if (c->o_groups > 16) return -1;
    DSV4QMat slices[16];
    DSV4PMJob jobs[16];
    double ts = dsv4_now_s();
    for (int g = 0; g < c->o_groups; g++) {
        for (int t = 0; t < n; t++)
            memcpy(p->xg + ((size_t)g * p->cap + t) * gw,
                   p->o + (size_t)t * (size_t)H * hd + (size_t)g * gw,
                   (size_t)gw * sizeof(float));
        DSV4QMat slice = w->attn.wo_a;
        slice.rows = gr;
        slice.cols = gw;
        slice.w = (const unsigned char *)w->attn.wo_a.w +
                  (size_t)g * gr * row_bytes;
        if (slice.s) {
            const size_t sc_row = (size_t)((gw + slice.blk_c - 1) / slice.blk_c);
            slice.s = (const unsigned char *)w->attn.wo_a.s +
                      (size_t)g * (size_t)(gr / (slice.blk_r ? slice.blk_r : 1))
                      * sc_row;
        }
        slices[g] = slice;
        jobs[g].m = &slices[g];
        jobs[g].x = p->xg + (size_t)g * p->cap * gw;
        jobs[g].y = p->yg + (size_t)g * p->cap * gr;
        jobs[g].n = n;
    }
    p->st.oproj_pack += dsv4_now_s() - ts;
    if (dsv4_prefill_pm_multi_stats(p, jobs, c->o_groups,
                                    &p->st.oproj, &p->st.f_oproj)) return -1;
    ts = dsv4_now_s();
    for (int g = 0; g < c->o_groups; g++)
        for (int t = 0; t < n; t++)
            memcpy(p->ogrp + (size_t)t * (size_t)c->o_groups * c->o_lora +
                       (size_t)g * gr,
                   p->yg + (size_t)g * p->cap * gr + (size_t)t * gr,
                   (size_t)gr * sizeof(float));
    p->st.oproj_unpack += dsv4_now_s() - ts;
    if (dsv4_prefill_pm_stats(p, p->out, p->ogrp, &w->attn.wo_b, n,
                              &p->st.oproj, &p->st.f_oproj)) return -1;
    ts = dsv4_now_s();
    for (int t = 0; t < n; t++)
        dsv4_hc_post(p->H + (size_t)t * (size_t)hc * d,
                     p->out + (size_t)t * d,
                     p->resid + (size_t)t * (size_t)hc * d,
                     p->post + (size_t)t * hc,
                     p->comb + (size_t)t * hc * hc, hc, d);
    p->st.hc_post += dsv4_now_s() - ts;

    return 0;
}

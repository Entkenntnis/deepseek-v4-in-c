/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_proj.c - batched projection GEMMs shared by attention and MoE.
 *
 * Keeping these here means prefill_attn.c and prefill_moe.c depend on this
 * file, and this file depends only on the matmul and Vulkan seams. The
 * orchestrator (dsv4_prefill.c) no longer has to expose projection helpers to
 * its own submodules, which removes the old prefill module cycle.
 */
#include <string.h>

#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

int dsv4_prefill_pm_multi(DSV4Prefill *p, const DSV4PMJob *jobs, int nj)
{
    for (int j = 0; j < nj; j++) {
        if (jobs[j].n < 1) continue;
        if (p->vk && dsv4_prefill_vk_gemm(p->vk, jobs[j].y, jobs[j].x,
                                          jobs[j].m, jobs[j].n) == 0)
            continue;
        dsv4_mmq_n(jobs[j].y, jobs[j].x, jobs[j].m, jobs[j].n);
    }
    return 0;
}

int dsv4_prefill_pm(DSV4Prefill *p, float *y, const float *x,
                    const DSV4QMat *m, int n)
{
    DSV4PMJob job = { m, x, y, n };
    return dsv4_prefill_pm_multi(p, &job, 1);
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

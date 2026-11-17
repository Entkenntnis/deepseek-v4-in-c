/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_pm.c - the one batched-matmul seam and its timed wrappers.
 * Every dense projection goes through these so none can silently fall out of
 * the report. */
#include "dsv4.h"
#include "dsv4_prof.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"
#include "dsv4_prefill_vk.h"

static void pf_flops(double *acc, const DSV4QMat *m, int n)
{
    *acc += 2.0 * (double)m->rows * (double)m->cols * (double)n * 1e-9;
}

static int pm(DSV4Prefill *p, int layer, const DSV4PMJob *jobs, int nj)
{
    (void)layer;
    for (int j = 0; j < nj; j++) {
        if (jobs[j].n < 1) continue;
        if (p->vk && dsv4_prefill_vk_gemm(p->vk, jobs[j].y, jobs[j].x,
                                          jobs[j].m, jobs[j].n,
                                          DSV4_VK_KIND_DENSE) == 0)
            continue;
        dsv4_mmq_n(jobs[j].y, jobs[j].x, jobs[j].m, jobs[j].n);
    }
    return 0;
}

static int pm1(DSV4Prefill *p, int layer, float *y, const float *x,
               const DSV4QMat *m, int n)
{
    DSV4PMJob job = { m, x, y, n };
    return pm(p, layer, &job, 1);
}

int dsv4_prefill_pm_stats(DSV4Prefill *p, float *y, const float *x,
                          const DSV4QMat *m, int n, double *timer,
                          double *flops)
{
    const double t0 = dsv4_now_s();
    const int rc = pm1(p, 0, y, x, m, n);
    *timer += dsv4_now_s() - t0;
    if (flops) pf_flops(flops, m, n);
    return rc;
}

int dsv4_prefill_pm_multi_stats(DSV4Prefill *p, const DSV4PMJob *jobs, int nj,
                                double *timer, double *flops)
{
    const double t0 = dsv4_now_s();
    const int rc = pm(p, 0, jobs, nj);
    *timer += dsv4_now_s() - t0;
    if (flops)
        for (int j = 0; j < nj; j++) pf_flops(flops, jobs[j].m, jobs[j].n);
    return rc;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_pm.c - the one batched-matmul seam. Every dense projection goes
 * through these so none can silently fall out of the GPU/CPU dispatch. */
#include "dsv4.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"
#include "dsv4_prefill_vk.h"

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

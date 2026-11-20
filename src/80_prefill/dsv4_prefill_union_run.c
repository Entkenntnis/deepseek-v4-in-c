/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_union_run.c - drain one expert batch and do the k-order sum. */
#include <stdio.h>

#include "dsv4.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

static int union_run_group(DSV4Prefill *p, int L, const DSV4ExpertW **ex,
                           const int *order, int ng)
{
    int done[DSV4_MAX_PREFETCH];
    int q[DSV4_MAX_PREFETCH];
    int qh = 0, qt = 0;
    int remaining = ng;
    for (int j = 0; j < ng; j++) done[j] = 0;

    DSV4PFillCtx fill = {
        .p = p, .ex = ex, .order = order, .ng = ng, .done = done,
        .remaining = &remaining, .q = q, .qh = &qh, .qt = &qt,
        .io_done = 0,
    };

    /* Prime the ready queue with everything the readers have already
     * published. This is non-blocking and gives a GPU bucket's first fill
     * window real CPU work immediately; wait_any() is only needed when the
     * queue runs dry. */
    {
        int k;
        while (dsv4_cache_try(p->cache, &k)) {
            if (k >= 0 && k < ng && !done[k] && ex[k]) q[qt++] = k;
        }
    }

    int cancelled = 0;
    for (;;) {
        if (dsv4_prefill_cancelled(p)) { cancelled = 1; break; }
        if (remaining <= 0) break;

        /* 1. Run one ready bucket through the GPU-first policy. A GPU bucket
         *    fills its waits with CPU work; if none is ready, the CPU takes
         *    the smallest available bucket. */
        if (dsv4_prefill_union_run_one_ready(p, &fill)) continue;

        /* 2. Nothing ready: take the next published bucket from the source. */
        int k = -1;
        const int got = dsv4_cache_wait(p->cache, &k);
        if (!got) break;
        if (k >= 0 && k < ng && !done[k] && ex[k]) q[qt++] = k;
        while (dsv4_cache_try(p->cache, &k)) {
            if (k >= 0 && k < ng && !done[k] && ex[k]) q[qt++] = k;
        }
    }

    dsv4_cache_end(p->cache, ex);

    if (cancelled) return dsv4_prefill_cancel_code(p);

    /* Hits (and anything wait_any never handed out) arrive through end(). Put
     * them back through the same GPU-first policy: resident hits may not
     * bypass the split. End the I/O phase first so the fill callback stops
     * polling try_any(). */
    fill.io_done = 1;
    qh = qt = 0;
    for (int j = 0; j < ng; j++) {
        if (done[j]) continue;
        if (!ex[j]) {
            fprintf(stderr, "prefill: layer %d could not obtain routed "
                            "expert %d; refusing to emit a token routed "
                            "through fewer experts\n",
                    L, order[j]);
            return -1;
        }
        q[qt++] = j;
    }
    while (remaining > 0 && dsv4_prefill_union_run_one_ready(p, &fill))
        ;
    for (int j = 0; j < ng; j++) {
        if (done[j]) continue;
        /* Should be unreachable: every remaining bucket was queued above. */
        const int e = order[j];
        dsv4_prefill_expert_bucket(p, ex[j], e, p->ex_count[e],
                                   p->ex_tok + p->ex_off[e],
                                   p->ex_k + p->ex_off[e],
                                   DSV4_EXPERT_ROUTE_CPU, NULL, NULL);
        done[j] = 1;
    }
    return 0;
}

int dsv4_prefill_moe_union(DSV4Prefill *p, int L, int n)
{
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, topk = c->topk;

    /* Direct callers may not have prefetched yet; do the decide pass now. */
    if (!p->pf_active || p->pf_layer != L) {
        const int prc = dsv4_prefill_moe_prefetch(p, L, n);
        if (prc) return prc;
    }
    if (!p->pf_active) return -1;

    const int rc = union_run_group(p, L, p->pf_ex, p->ex_order, p->pf_n);
    p->pf_active = 0;
    if (rc) return rc;

    if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);

    for (int t = 0; t < n; t++) {
        if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);
        float *o = p->out + (size_t)t * d;
        const float *tw = p->topk_w + (size_t)t * topk;
        for (int i = 0; i < d; i++) o[i] = 0.0f;
        for (int k = 0; k < topk; k++) {
            const float wk = tw[k];
            const float *acc = p->scratch[t].expert_acc +
                               (size_t)k * p->scratch[t].expert_acc_stride;
            for (int i = 0; i < d; i++) o[i] += wk * acc[i];
        }
        for (int i = 0; i < d; i++) o[i] += p->sh_out[(size_t)t * d + i];
    }
    return 0;
}

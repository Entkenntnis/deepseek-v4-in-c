/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_union.c - layer-level expert request set, then expert-major. */
#include <stdio.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_prof.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

/* Request every distinct expert this layer's block needs, in first-use order,
 * so the reader pool sees the whole layer instead of six requests at a time.
 * The per-token end() barrier exposed a large share of prefill as expert I/O;
 * here there is one barrier per group and each expert's bucket is computed as
 * soon as its read lands. `make e2e --detailed` shows the I/O share; the
 * per-token k-order sum is unchanged. */
int dsv4_prefill_moe_union(DSV4Prefill *p, int L, int n)
{
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, topk = c->topk, ne = c->n_experts;
    const double tu0 = dsv4_now_s();
    DSV4PrefillLayerInstr *li =
        (p->instr && L >= 0 && L < c->n_layers) ? &p->linstr[L] : NULL;
    p->cur_layer = L;
    if (li) {
        memset(li, 0, sizeof *li);
        li->requests = n * topk;
    }
    DSV4ExpertStats es0, es1;
    const int have_estats = p->instr && p->src->stats
                         && p->src->stats(p->src->ctx, &es0) == 0;

    memset(p->ex_count, 0, (size_t)ne * sizeof *p->ex_count);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < topk; k++)
            p->ex_count[p->topk_idx[(size_t)t * topk + k]]++;

    p->ex_off[0] = 0;
    for (int e = 0; e < ne; e++)
        p->ex_off[e + 1] = p->ex_off[e] + p->ex_count[e];

    memset(p->ex_fill, 0, (size_t)ne * sizeof *p->ex_fill);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < topk; k++) {
            const int e = p->topk_idx[(size_t)t * topk + k];
            const int pos = p->ex_off[e] + p->ex_fill[e]++;
            p->ex_tok[pos] = t;
            p->ex_k[pos] = k;
        }

    memset(p->ex_seen, 0, (size_t)ne * sizeof *p->ex_seen);
    int nord = 0;
    for (int t = 0; t < n; t++)
        for (int k = 0; k < topk; k++) {
            const int e = p->topk_idx[(size_t)t * topk + k];
            if (!p->ex_seen[e]) {
                p->ex_seen[e] = 1;
                p->ex_order[nord++] = e;
            }
        }
    if (li) li->experts = nord;

    for (int base = 0; base < nord; base += DSV4_MAX_PREFETCH) {
        if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);
        int ng = nord - base;
        if (ng > DSV4_MAX_PREFETCH) ng = DSV4_MAX_PREFETCH;

        const DSV4ExpertW *ex[DSV4_MAX_PREFETCH];
        int done[DSV4_MAX_PREFETCH];
        for (int j = 0; j < ng; j++) { ex[j] = NULL; done[j] = 0; }

        double t0 = dsv4_now_s();
        const int rc = p->src->begin(p->src->ctx, L, p->ex_order + base,
                                     ng, ex);
        double td = dsv4_now_s() - t0;
        p->t_io += td;
        p->st.expert_io += td;
        if (p->instr) {
            p->st.io_begin_s += td;
            p->st.io_groups++;
            if (li) li->begin_s += td;
        }
        if (rc != 0) {
            p->src->end(p->src->ctx, ex);
            return -1;
        }

        /* Once a group's reads are in flight, its end() barrier still has to
         * run. Stop computing buckets, but drain the group before returning:
         * returning with reads outstanding would race the reset that follows a
         * cancelled prefill. */
        int cancelled = 0;
        const double tc0 = dsv4_now_s();
        for (int j = 0; j < ng; j++) {
            if (dsv4_prefill_cancelled(p)) cancelled = 1;
            if (cancelled || !ex[j]) continue;
            const int e = p->ex_order[base + j];
            dsv4_prefill_expert_bucket(p, ex[j], p->ex_count[e],
                                       p->ex_tok + p->ex_off[e],
                                       p->ex_k + p->ex_off[e]);
            done[j] = 1;
        }
        if (p->instr && li) li->compute_s += dsv4_now_s() - tc0;

        if (p->src->wait_any) {
            for (;;) {
                int j = -1;
                t0 = dsv4_now_s();
                const int got = p->src->wait_any(p->src->ctx, &j);
                td = dsv4_now_s() - t0;
                p->t_io += td;
                p->st.expert_io += td;
                if (p->instr) {
                    p->st.io_wait_s += td;
                    if (li) li->wait_s += td;
                }
                if (!got) break;
                if (j < 0 || j >= ng || done[j] || !ex[j]) continue;
                if (dsv4_prefill_cancelled(p)) cancelled = 1;
                if (cancelled) continue;
                const double tb0 = dsv4_now_s();
                const int e = p->ex_order[base + j];
                dsv4_prefill_expert_bucket(p, ex[j], p->ex_count[e],
                                           p->ex_tok + p->ex_off[e],
                                           p->ex_k + p->ex_off[e]);
                if (p->instr && li) li->compute_s += dsv4_now_s() - tb0;
                done[j] = 1;
            }
        }

        t0 = dsv4_now_s();
        p->src->end(p->src->ctx, ex);
        td = dsv4_now_s() - t0;
        p->t_io += td;
        p->st.expert_io += td;
        if (p->instr) {
            p->st.io_end_s += td;
            if (li) li->end_s += td;
        }

        if (cancelled) return dsv4_prefill_cancel_code(p);

        const double tt0 = dsv4_now_s();
        for (int j = 0; j < ng; j++) {
            if (done[j]) continue;
            if (!ex[j]) {
                fprintf(stderr, "prefill: layer %d could not obtain routed "
                                "expert %d; refusing to emit a token routed "
                                "through fewer experts\n",
                        L, p->ex_order[base + j]);
                return -1;
            }
            const int e = p->ex_order[base + j];
            dsv4_prefill_expert_bucket(p, ex[j], p->ex_count[e],
                                       p->ex_tok + p->ex_off[e],
                                       p->ex_k + p->ex_off[e]);
            done[j] = 1;
        }
        if (p->instr && li) li->compute_s += dsv4_now_s() - tt0;
    }

    if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);

    const double tsum = dsv4_now_s();
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
    p->st.expert_sum += dsv4_now_s() - tsum;
    p->t_moe += dsv4_now_s() - tsum;

    if (p->instr) {
        if (li) {
            li->union_s = dsv4_now_s() - tu0;
            if (have_estats
                && p->src->stats(p->src->ctx, &es1) == 0)
                li->bytes = es1.bytes_read - es0.bytes_read;
        }
    }
    return 0;
}

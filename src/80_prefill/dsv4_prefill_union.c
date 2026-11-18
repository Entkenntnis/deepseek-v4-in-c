/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_union.c - layer-level expert request set, then expert-major.
 *
 * The batch is split in two:
 *
 *   prefetch()  builds the layer's distinct-expert set, sorts it into
 *               checkpoint order and calls begin() once for the whole layer.
 *               Callers run it as soon as routing is known: after the gate for
 *               scored layers, at the top of the layer for hash layers. The
 *               reader pool keeps filling slots while the caller runs the
 *               shared expert (and, for hash layers, attention).
 *
 *   union()     waits for the reads, runs each expert's bucket as soon as its
 *               data lands, ends the batch and does the k-order weighted sum.
 *
 * One begin() per layer means the readers never wait at a 64-expert group
 * boundary, and begin() queues the whole layer instead of six requests at a
 * time. The per-token k-order sum is unchanged. */
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_prof.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

static DSV4PrefillLayerInstr *union_instr(DSV4Prefill *p, int L)
{
    return (p->instr && L >= 0 && L < p->c->n_layers)
         ? &p->linstr[L] : NULL;
}



/* Build the distinct-expert set in first-use order, then hand it to the
 * source's optional checkpoint-order sort. ex_tok/ex_k are indexed by expert
 * id and stay valid across the sort; only the processing order changes. */
static int union_prepare(DSV4Prefill *p, int L, int n)
{
    const DSV4Cfg *c = p->c;
    const int topk = c->topk;
    const int ne = c->n_experts;

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
    if (p->src->sort) p->src->sort(p->src->ctx, L, p->ex_order, nord);
    return nord;
}

/* Ready-bucket queue shared by the scheduler and the GPU fill callback.
 * Entries are indices into ex/order and are only appended by wait_any(), so a
 * pointer a reader is still publishing is never dereferenced here. Resident
 * hits are not in this queue; end() delivers them and the trailing pass runs
 * them after the batch has synchronized. */
typedef struct {
    DSV4Prefill *p;
    const DSV4ExpertW **ex;
    const int *order;
    int ng;
    int *done;
    int *remaining;
    int *q, *qh, *qt;
    DSV4PrefillLayerInstr *li;
} DSV4PFillCtx;

/* Run one ready CPU bucket while the GPU fence is pending. The smallest count
 * wins: a 1-token bucket fits a short GPU tile, while a 5-7 bucket may not.
 * Returns 1 when a bucket ran, 0 when the ready queue has no CPU work. */
static int union_fill_cpu(void *ctx)
{
    DSV4PFillCtx *f = (DSV4PFillCtx *)ctx;
    DSV4Prefill *p = f->p;
    if (dsv4_prefill_cancelled(p)) return 0;

    /* Readers keep publishing while the GPU tile runs. Pull in whatever has
     * landed since the last window; this is non-blocking and is what lets the
     * later w3/w2 windows find CPU buckets that were not ready for w1. */
    if (p->src->try_any) {
        int k;
        while (p->src->try_any(p->src->ctx, &k)) {
            if (k >= 0 && k < f->ng && !f->done[k] && f->ex[k])
                f->q[(*f->qt)++] = k;
        }
    }

    int best = -1, best_count = INT_MAX;
    for (int i = *f->qh; i < *f->qt; i++) {
        const int j = f->q[i];
        if (j < 0 || f->done[j] || !f->ex[j]) continue;
        const int e = f->order[j];
        const int count = p->ex_count[e];
        if (count >= 8 || count > DSV4_PREFILL_FILL_MAX) continue;
        if (count < best_count) { best = i; best_count = count; }
    }
    if (best < 0) return 0;

    const int j = f->q[best];
    f->q[best] = -1;
    const int e = f->order[j];
    const double tb0 = dsv4_now_s();
    dsv4_prefill_expert_bucket_fill(p, f->ex[j], p->ex_count[e],
                                    p->ex_tok + p->ex_off[e],
                                    p->ex_k + p->ex_off[e]);
    if (p->instr && f->li) f->li->compute_s += dsv4_now_s() - tb0;
    f->done[j] = 1;
    (*f->remaining)--;
    return 1;
}

/* Consume one batch. Ready GPU buckets start first; while each GPU tile is in
 * flight, fill_cpu() runs one ready CPU bucket. If no ready bucket remains, the
 * scheduler asks wait_any() for the next one, so the batch's I/O tail still
 * overlaps compute. end() drains whatever wait_any() did not consume, and the
 * trailing pass handles resident hits after that synchronization point.
 *
 * The batch's slot claims are released by end(); on a cancel we still drain it
 * before returning so the reset that follows cannot race a reader. */
static int union_run_group(DSV4Prefill *p, int L, const DSV4ExpertW **ex,
                           const int *order, int ng, DSV4PrefillLayerInstr *li)
{
    int done[DSV4_MAX_PREFETCH];
    int q[DSV4_MAX_PREFETCH];
    int qh = 0, qt = 0;
    int remaining = ng;
    for (int j = 0; j < ng; j++) done[j] = 0;

    DSV4PFillCtx fill = {
        .p = p, .ex = ex, .order = order, .ng = ng, .done = done,
        .remaining = &remaining, .q = q, .qh = &qh, .qt = &qt, .li = li,
    };

    /* Prime the ready queue with everything the readers have already
     * published. This is non-blocking and gives a GPU bucket's first fill
     * window real CPU work immediately; wait_any() is only needed when the
     * queue runs dry. */
    if (p->src->try_any) {
        int k;
        while (p->src->try_any(p->src->ctx, &k)) {
            if (k >= 0 && k < ng && !done[k] && ex[k]) q[qt++] = k;
        }
    }

    int cancelled = 0;
    for (;;) {
        if (dsv4_prefill_cancelled(p)) { cancelled = 1; break; }
        if (remaining <= 0) break;

        /* 1. A ready GPU bucket: start it and fill its waits with CPU work. */
        int pick = -1;
        for (int i = qh; i < qt; i++) {
            const int j = q[i];
            if (j < 0 || done[j] || !ex[j]) continue;
            if (dsv4_prefill_expert_is_gpu(p, p->ex_count[order[j]])) {
                pick = i;
                break;
            }
        }
        if (pick >= 0) {
            const int j = q[pick];
            q[pick] = -1;
            const int e = order[j];
            const double tb0 = dsv4_now_s();
            dsv4_prefill_expert_bucket(p, ex[j], p->ex_count[e],
                                       p->ex_tok + p->ex_off[e],
                                       p->ex_k + p->ex_off[e],
                                       union_fill_cpu, &fill);
            if (p->instr && li) li->compute_s += dsv4_now_s() - tb0;
            done[j] = 1;
            remaining--;
            continue;
        }

        /* 2. No GPU bucket ready: do one ready CPU bucket. */
        pick = -1;
        for (int i = qh; i < qt; i++) {
            const int j = q[i];
            if (j < 0 || done[j] || !ex[j]) continue;
            pick = i;
            break;
        }
        if (pick >= 0) {
            const int j = q[pick];
            q[pick] = -1;
            const int e = order[j];
            const double tb0 = dsv4_now_s();
            dsv4_prefill_expert_bucket(p, ex[j], p->ex_count[e],
                                       p->ex_tok + p->ex_off[e],
                                       p->ex_k + p->ex_off[e], NULL, NULL);
            if (p->instr && li) li->compute_s += dsv4_now_s() - tb0;
            done[j] = 1;
            remaining--;
            continue;
        }

        /* 3. Nothing ready: take the next published bucket from the source. */
        if (!p->src->wait_any) break;
        int k = -1;
        double t0 = dsv4_now_s();
        const int got = p->src->wait_any(p->src->ctx, &k);
        const double td = dsv4_now_s() - t0;
        p->t_io += td;
        p->st.expert_io += td;
        if (p->instr) {
            p->st.io_wait_s += td;
            if (li) li->wait_s += td;
        }
        if (!got) break;
        if (k >= 0 && k < ng && !done[k] && ex[k]) q[qt++] = k;
        if (p->src->try_any) {
            while (p->src->try_any(p->src->ctx, &k)) {
                if (k >= 0 && k < ng && !done[k] && ex[k]) q[qt++] = k;
            }
        }
    }

    double t0 = dsv4_now_s();
    p->src->end(p->src->ctx, ex);
    double td = dsv4_now_s() - t0;
    p->t_io += td;
    p->st.expert_io += td;
    if (p->instr) {
        p->st.io_end_s += td;
        if (li) li->end_s += td;
    }

    if (cancelled) return dsv4_prefill_cancel_code(p);

    /* Hits (and anything wait_any never handed out) arrive through end(). The
     * trailing pass runs after that lock, so these pointers are safe too. */
    const double tt0 = dsv4_now_s();
    for (int j = 0; j < ng; j++) {
        if (done[j]) continue;
        if (!ex[j]) {
            fprintf(stderr, "prefill: layer %d could not obtain routed "
                            "expert %d; refusing to emit a token routed "
                            "through fewer experts\n",
                    L, order[j]);
            return -1;
        }
        const int e = order[j];
        dsv4_prefill_expert_bucket(p, ex[j], p->ex_count[e],
                                   p->ex_tok + p->ex_off[e],
                                   p->ex_k + p->ex_off[e], NULL, NULL);
        done[j] = 1;
    }
    if (p->instr && li) li->compute_s += dsv4_now_s() - tt0;
    return 0;
}

int dsv4_prefill_moe_prefetch(DSV4Prefill *p, int L, int n)
{
    const DSV4Cfg *c = p->c;
    const int topk = c->topk;

    if (p->pf_active) {
        if (p->pf_layer == L) return 0;      /* already started for this layer */
        dsv4_prefill_moe_abort(p);
    }

    DSV4PrefillLayerInstr *li = union_instr(p, L);
    p->cur_layer = L;
    p->pf_t0 = dsv4_now_s();
    p->pf_compute_t0 = p->pf_t0;
    p->pf_layer = L;
    p->pf_active = 0;
    p->pf_prepared = 0;
    p->pf_estats = 0;
    p->pf_bytes0 = 0;
    if (li) {
        memset(li, 0, sizeof *li);
        li->requests = n * topk;
    }

    const int nord = union_prepare(p, L, n);
    p->pf_n = nord;
    if (li) li->experts = nord;

    if (p->instr && p->src->stats) {
        DSV4ExpertStats es0;
        if (p->src->stats(p->src->ctx, &es0) == 0) {
            p->pf_estats = 1;
            p->pf_bytes0 = es0.bytes_read;
        }
    }

    /* The cache batch is fixed-size. One Flash layer has at most 256 distinct
     * experts, so this normally fires for the whole layer; a larger model falls
     * back to the grouped path in union(). */
    p->pf_prepared = 1;
    if (nord > DSV4_MAX_PREFETCH) return 0;

    for (int j = 0; j < nord; j++) p->pf_ex[j] = NULL;

    const double t0 = dsv4_now_s();
    const int rc = p->src->begin(p->src->ctx, L, p->ex_order, nord, p->pf_ex);
    const double td = dsv4_now_s() - t0;
    p->t_io += td;
    p->st.expert_io += td;
    if (p->instr) {
        p->st.io_begin_s += td;
        p->st.io_groups++;
        if (li) li->begin_s += td;
    }
    if (rc != 0) {
        p->src->end(p->src->ctx, p->pf_ex);
        p->pf_prepared = 0;
        return -1;
    }
    p->pf_active = 1;
    return 0;
}

void dsv4_prefill_moe_abort(DSV4Prefill *p)
{
    if (!p || !p->pf_active) return;
    p->src->end(p->src->ctx, p->pf_ex);
    p->pf_active = 0;
    p->pf_prepared = 0;
}

int dsv4_prefill_moe_union(DSV4Prefill *p, int L, int n)
{
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, topk = c->topk;
    DSV4PrefillLayerInstr *li = union_instr(p, L);

    /* Direct callers (and the rare layer too large for one batch) may not have
     * prefetched yet; do the decide pass now. */
    if (!p->pf_prepared || p->pf_layer != L) {
        const int prc = dsv4_prefill_moe_prefetch(p, L, n);
        if (prc) return prc;
    }

    p->pf_compute_t0 = dsv4_now_s();
    if (li) li->lead_s = p->pf_compute_t0 - p->pf_t0;

    int rc = 0;
    if (p->pf_active) {
        rc = union_run_group(p, L, p->pf_ex, p->ex_order, p->pf_n, li);
        p->pf_active = 0;
    } else {
        /* Fallback: more distinct experts than one cache batch can hold. */
        const int nord = p->pf_n;
        for (int base = 0; base < nord; base += DSV4_MAX_PREFETCH) {
            if (dsv4_prefill_cancelled(p)) {
                rc = dsv4_prefill_cancel_code(p);
                break;
            }
            int ng = nord - base;
            if (ng > DSV4_MAX_PREFETCH) ng = DSV4_MAX_PREFETCH;

            const DSV4ExpertW *ex[DSV4_MAX_PREFETCH];
            for (int j = 0; j < ng; j++) ex[j] = NULL;

            const double t0 = dsv4_now_s();
            const int brc = p->src->begin(p->src->ctx, L,
                                          p->ex_order + base, ng, ex);
            const double td = dsv4_now_s() - t0;
            p->t_io += td;
            p->st.expert_io += td;
            if (p->instr) {
                p->st.io_begin_s += td;
                p->st.io_groups++;
                if (li) li->begin_s += td;
            }
            if (brc != 0) {
                p->src->end(p->src->ctx, ex);
                rc = -1;
                break;
            }
            rc = union_run_group(p, L, ex, p->ex_order + base, ng, li);
            if (rc) break;
        }
    }
    p->pf_prepared = 0;
    if (rc) return rc;

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
    p->t_moe_wall += dsv4_now_s() - p->pf_compute_t0;

    if (p->instr) {
        if (li) {
            li->union_s = dsv4_now_s() - p->pf_t0;
            if (p->pf_estats) {
                DSV4ExpertStats es1;
                if (p->src->stats(p->src->ctx, &es1) == 0)
                    li->bytes = es1.bytes_read - p->pf_bytes0;
            }
        }
    }
    return 0;
}

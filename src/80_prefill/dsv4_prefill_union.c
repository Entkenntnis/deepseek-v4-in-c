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
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

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
    dsv4_cache_sort_experts(p->cache, L, p->ex_order, nord);
    return nord;
}

/* Ready-bucket queue shared by the scheduler and the GPU fill callback.
 * Entries are indices into ex/order and are only appended by wait_any(), so a
 * pointer a reader is still publishing is never dereferenced here. Resident
 * hits are not in this queue; end() delivers them and the trailing pass runs
 * them after the batch has synchronized. */
int dsv4_prefill_moe_prefetch(DSV4Prefill *p, int L, int n)
{
    if (p->pf_active) {
        if (p->pf_layer == L) return 0;      /* already started for this layer */
        dsv4_prefill_moe_abort(p);
    }

    p->pf_layer = L;
    p->pf_active = 0;
    p->pf_prepared = 0;

    const int nord = union_prepare(p, L, n);
    p->pf_n = nord;

    /* The cache batch is fixed-size. One Flash layer has at most 256 distinct
     * experts, so this normally fires for the whole layer; a larger model falls
     * back to the grouped path in union(). */
    p->pf_prepared = 1;
    if (nord > DSV4_MAX_PREFETCH) return 0;

    for (int j = 0; j < nord; j++) p->pf_ex[j] = NULL;

    const int rc = dsv4_cache_begin(p->cache, L, p->ex_order, nord, p->pf_ex);
    if (rc != 0) {
        dsv4_cache_end(p->cache, p->pf_ex);
        p->pf_prepared = 0;
        return -1;
    }
    p->pf_active = 1;
    return 0;
}

void dsv4_prefill_moe_abort(DSV4Prefill *p)
{
    if (!p || !p->pf_active) return;
    dsv4_cache_end(p->cache, p->pf_ex);
    p->pf_active = 0;
    p->pf_prepared = 0;
}

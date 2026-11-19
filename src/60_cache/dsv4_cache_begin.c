/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_begin.c - the serial decide pass for one expert batch. */
#include <string.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"

int dsv4_cache_get_many_begin(DSV4Cache *c, int layer, const int *experts,
                              int n, const DSV4ExpertW **out,
                              DSV4CacheBatch *b)
{
    if (n > DSV4_MAX_PREFETCH) { b->rc = -1; return -1; }

    int claimed[DSV4_MAX_PREFETCH], nclaim = 0;
    b->layer = layer;
    b->n = n;
    b->ntodo = 0;
    b->ndup = 0;
    b->nwait = 0;
    b->rc = 0;
    b->rdone = 0;
    b->is_prefetch = 0;
    b->streaming = (c->stream_cap > 0 && c->nrth > 0) ? 1 : 0;
    b->next_j = 0;
    b->out = out;
    for (int i = 0; i < DSV4_MAX_PREFETCH; i++) {
        b->published[i] = 0;
        b->taken[i] = 0;
    }

    pthread_mutex_lock(&c->rmx);
    for (int k = 0; k < n; k++) {
        const int e = experts[k];
        out[k] = NULL;
        c->clock++;

        /* A duplicate request must not take a second slot. */
        int dup = -1;
        for (int j = 0; j < b->ntodo; j++)
            if (b->expert_of[j] == e) { dup = j; break; }
        if (dup >= 0) {
            c->slot[b->slot_of[dup]].stamp = ++c->clock;
            c->hits++;
            b->dup_k[b->ndup] = k;
            b->dup_j[b->ndup] = dup;
            b->ndup++;
            continue;
        }

        int found = -1;
        for (int i = 0; i < c->limit; i++)
            if (c->slot[i].layer == layer && c->slot[i].expert == e) {
                found = i; break;
            }
        if (found >= 0 && c->slot[found].pending) {
            /* The hash-layer prefetch is already reading it. Do NOT block
             * begin(): register the slot and let wait_any()/end() hand it
             * over the moment its reader publishes. Adding it to claimed[]
             * keeps a later claim in this same decide pass off it. */
            claimed[nclaim++] = found;
            b->wait_slot[b->nwait]   = found;
            b->wait_k[b->nwait]      = k;
            b->wait_expert[b->nwait] = e;
            b->wait_done[b->nwait]   = 0;
            b->nwait++;
            continue;
        }
        if (found >= 0) {
            c->slot[found].stamp = ++c->clock;
            c->hits++;
            out[k] = &c->slot[found].w;      /* resident: safe to use NOW */
            claimed[nclaim++] = found;
            continue;
        }

        DSV4Loc t[6];
        if (dsv4_cache_resolve(c, layer, e, t) != 0) { b->rc = -1; continue; }

        const int victim = dsv4_cache_claim_slot(c, layer, e, claimed, nclaim);
        if (victim < 0) { b->rc = -1; continue; }   /* fewer slots than top-k */
        claimed[nclaim++] = victim;

        b->slot_of[b->ntodo]   = victim;
        b->k_of[b->ntodo]      = k;
        b->layer_of[b->ntodo]  = layer;
        b->expert_of[b->ntodo] = e;
        memcpy(b->t_of[b->ntodo], t, sizeof t);
        b->failed[b->ntodo] = 0;
        b->got[b->ntodo]    = 0;
        b->ntodo++;
    }

    if (b->ntodo > 0 && c->nrth > 0) {
        int first = b->ntodo;
        if (b->streaming && c->stream_cap < first) first = c->stream_cap;
        b->next_j = first;
        for (int j = 0; j < first; j++) {
            if (dsv4_cache_rq_push(c, b, j) != 0) {
                DSV4Slot *sl = &c->slot[b->slot_of[j]];
                sl->layer = sl->expert = -1;
                sl->pending = 0;
                b->ntodo = j;
                b->next_j = j;
                b->rc = -1;
                break;
            }
        }
        pthread_cond_broadcast(&c->rwork);
    }
    pthread_mutex_unlock(&c->rmx);
    return b->rc;
}

int dsv4_cache_try_any(DSV4Cache *c, DSV4CacheBatch *b, int *k)
{
    if (!c || !b || !k || !b->streaming) return 0;
    int expert = -1;
    pthread_mutex_lock(&c->rmx);
    const int got = dsv4_cache_take_ready(c, b, k, &expert);
    pthread_mutex_unlock(&c->rmx);
    return got;
}

int dsv4_cache_wait_any(DSV4Cache *c, DSV4CacheBatch *b, int *k)
{
    if (!c || !b || !k) return 0;
    if (!b->streaming) return 0;

    pthread_mutex_lock(&c->rmx);
    for (;;) {
        int expert = -1;
        if (dsv4_cache_take_ready(c, b, k, &expert)) {
            pthread_mutex_unlock(&c->rmx);
            return 1;
        }
        if (b->rdone >= b->ntodo && !dsv4_cache_waits_left(b)) {
            pthread_mutex_unlock(&c->rmx);
            return 0;                      /* everything consumed */
        }
        pthread_cond_wait(&c->rdone, &c->rmx);
    }
}

/* ---- engine-facing wrappers: one batch owned by the cache ---- */

int dsv4_cache_begin(DSV4Cache *c, int layer, const int *experts, int n,
                     const DSV4ExpertW **out)
{
    return dsv4_cache_get_many_begin(c, layer, experts, n, out, c->batch);
}

int dsv4_cache_end(DSV4Cache *c, const DSV4ExpertW **out)
{
    return dsv4_cache_get_many_end(c, out, c->batch);
}

int dsv4_cache_wait(DSV4Cache *c, int *k)
{
    return dsv4_cache_wait_any(c, c->batch, k);
}

int dsv4_cache_try(DSV4Cache *c, int *k)
{
    return dsv4_cache_try_any(c, c->batch, k);
}

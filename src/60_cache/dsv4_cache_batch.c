/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_batch.c - slot claiming and the begin()/end() decide pass. */
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"

/* Reserve a slot for (layer, expert). Caller holds rmx. Prefers an empty slot,
 * then LRU, and never takes a slot another request in this call owns. */
int dsv4_cache_claim_slot(DSV4Cache *c, int layer, int expert,
                          const int *claimed, int nclaim)
{
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < c->nslot; i++) {
        DSV4Slot *s = &c->slot[i];
        if (s->pending) continue;
        int taken = 0;
        for (int j = 0; j < nclaim; j++)
            if (claimed[j] == i) { taken = 1; break; }
        if (taken) continue;
        if (s->layer < 0) { victim = i; break; }
        if (s->stamp < oldest) { oldest = s->stamp; victim = i; }
    }
    if (victim < 0) return -1;
    DSV4Slot *s = &c->slot[victim];
    s->layer = layer;
    s->expert = expert;
    s->origin = 0;
    s->pending = 1;
    s->stamp = ++c->clock;
    return victim;
}

/* Caller holds rmx. Hand out one k that has become usable: either a read this
 * batch queued (published by its reader for a streaming batch) or a prefetch
 * slot that begin() registered. Sets b->out[k] for the prefetch case. Returns
 * 0 when nothing new is ready. */
int dsv4_cache_take_ready(DSV4Cache *c, DSV4CacheBatch *b, int *k, int *expert)
{
    for (int j = 0; j < b->ntodo; j++) {
        if (!b->published[j] || b->taken[j]) continue;
        b->taken[j] = 1;
        *k = b->k_of[j];
        *expert = b->expert_of[j];
        return 1;
    }
    for (int i = 0; i < b->nwait; i++) {
        if (b->wait_done[i]) continue;
        DSV4Slot *s = &c->slot[b->wait_slot[i]];
        if (s->pending) continue;
        b->wait_done[i] = 1;
        if (s->layer != b->layer || s->expert != b->wait_expert[i]) {
            b->out[b->wait_k[i]] = NULL;   /* prefetch failed; caller aborts */
            b->rc = -1;
            continue;
        }
        c->hits++;
        s->stamp = ++c->clock;
        b->out[b->wait_k[i]] = &s->w;
        *k = b->wait_k[i];
        *expert = b->wait_expert[i];
        return 1;
    }
    return 0;
}

int dsv4_cache_waits_left(const DSV4CacheBatch *b)
{
    for (int i = 0; i < b->nwait; i++)
        if (!b->wait_done[i]) return 1;
    return 0;
}

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
        for (int i = 0; i < c->nslot; i++)
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

        const DSV4Tensor *t[6];
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

int dsv4_cache_get_many_end(DSV4Cache *c, const DSV4ExpertW **out,
                            DSV4CacheBatch *b)
{
    if (b->ntodo > 0 && c->nrth == 0) {
        /* No pool. Still concurrent, just not overlapped with the caller --
         * exactly what this function did before the pool existed. */
#pragma omp parallel for schedule(dynamic, 1) if (b->ntodo > 1)
        for (int j = 0; j < b->ntodo; j++) {
            int bi = omp_get_thread_num();
            if (bi >= c->nbounce) bi = 0;
            b->failed[j] = dsv4_cache_load(c, &c->slot[b->slot_of[j]],
                                           b->t_of[j], c->bounce[bi],
                                           &b->got[j]) != 0;
        }
        b->rdone = b->ntodo;
    }

    pthread_mutex_lock(&c->rmx);
    for (;;) {
        int k, expert;
        /* Jobs and prefetch slots that have already published are handed out
         * here, so a batch whose caller never calls wait_any() (stream cap 0,
         * or the tail after the last wake) still gets them all. */
        while (dsv4_cache_take_ready(c, b, &k, &expert)) { }
        if (b->rdone >= b->ntodo && !dsv4_cache_waits_left(b)) break;
        pthread_cond_wait(&c->rdone, &c->rmx);
    }

    if (b->streaming) {
        /* The reader already published and accounted every miss. Anything the
         * caller did not consume through wait_any() is handed out here, so
         * begin()/end() users keep the old contract. */
        for (int j = 0; j < b->ntodo; j++) {
            if (b->taken[j]) continue;
            b->taken[j] = 1;
            out[b->k_of[j]] = b->failed[j] ? NULL
                                           : &c->slot[b->slot_of[j]].w;
        }
    } else {
        for (int j = 0; j < b->ntodo; j++) {
            DSV4Slot *sl = &c->slot[b->slot_of[j]];
            c->bytes_read += b->got[j];
            sl->pending = 0;
            if (b->failed[j]) {
                sl->layer = sl->expert = -1; /* no half-read slot goes live */
                b->rc = -1;
                continue;
            }
            sl->layer  = b->layer_of[j];
            sl->expert = b->expert_of[j];
            sl->origin = 2;
            c->misses++;
            out[b->k_of[j]] = &sl->w;
        }
    }
    for (int d = 0; d < b->ndup; d++) {
        const int j = b->dup_j[d];
        if (!b->failed[j]) out[b->dup_k[d]] = &c->slot[b->slot_of[j]].w;
    }
    pthread_mutex_unlock(&c->rmx);
    return b->rc;
}

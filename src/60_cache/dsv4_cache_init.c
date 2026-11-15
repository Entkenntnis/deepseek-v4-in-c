/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_init.c - allocate the slot array, staging buffers and reader pool. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"
#include "dsv4_threads.h"

int dsv4_cache_init(DSV4Cache *c, const DSV4St *st, const DSV4Cfg *cfg,
                    int64_t budget_bytes)
{
    memset(c, 0, sizeof *c);
    c->st = st; c->cfg = cfg;
    dsv4_cache_expert_geometry(c, cfg);

    /* A cache smaller than topk is a WARNING here, not an error: the constraint
     * belongs to get_many, which will not evict a slot it already claimed. A
     * single-expert caller is fine with two slots, and the cache gate relies on
     * that to test eviction order. The budget planner reserves topk experts,
     * and moe() is the last line of defence. */
    if ((int64_t)cfg->topk * c->expert_bytes > budget_bytes)
        fprintf(stderr,
                "dsv4_cache: %lld slots cannot hold one layer's %d experts at "
                "once;\n  dsv4_cache_get_many will fail. Single-expert use is "
                "fine.\n",
                (long long)(budget_bytes / c->expert_bytes), cfg->topk);

    if (budget_bytes < c->expert_bytes) {
        fprintf(stderr, "dsv4_cache: budget %lld bytes cannot hold one expert "
                        "(%lld bytes). Every access would miss and the same\n"
                        "  bytes would be read six times per layer.\n",
                (long long)budget_bytes, (long long)c->expert_bytes);
        return -1;
    }

    /* The largest coalesced run is the three weights; the widened window can
     * overhang by up to one alignment unit at each end. */
    int64_t widest_run = 0;
    for (int i = 0; i < 3; i++) widest_run += c->len_w[i];
    c->bounce_cap = widest_run + 2 * 4096;
    /* The reader width is a property of the DRIVE, not of the compute team:
     * taking it from omp_get_max_threads() made every pinning experiment change
     * two things at once. */
    int readers = dsv4_threads_readers();
    if (readers < 1) readers = 1;
    if (readers > DSV4_MAX_TOPK) readers = DSV4_MAX_TOPK;
    c->nbounce = omp_get_max_threads();
    /* +1: bounce[0] belongs to synchronous get(), readers use 1..nrth. */
    if (c->nbounce < readers + 1) c->nbounce = readers + 1;
    if (c->nbounce < 2) c->nbounce = 2;
    c->bounce = (unsigned char **)calloc((size_t)c->nbounce, sizeof *c->bounce);
    if (!c->bounce) return -1;
    for (int b = 0; b < c->nbounce; b++)
        if (posix_memalign((void **)&c->bounce[b], 4096,
                           (size_t)c->bounce_cap) != 0) {
            fprintf(stderr, "dsv4_cache: could not allocate a %lld byte staging "
                            "buffer\n", (long long)c->bounce_cap);
            return -1;
        }
    c->direct = (st->dfd && st->nshard > 0 && st->dfd[0] >= 0);
    if (!c->direct)
        fprintf(stderr, "dsv4_cache: O_DIRECT unavailable; expert reads go "
                        "through the page cache, which at a 148 GB working set "
                        "evicts more than it saves\n");

    /* Charged against slot_bytes, not expert_bytes: alignment slack is real
     * memory and the budget is a promise about peak RSS. */
    c->nslot = (int)(budget_bytes / c->slot_bytes);

    /* SAY SO WHEN THE CACHE CANNOT POSSIBLY WORK. Under LRU a hit needs the
     * entry to survive until the next layer visit; below a full pass's working
     * set the hit rate is exactly zero (README.md). This happened: 160 slots
     * against 258 returned 0 hits on 2,580 requests while reading 32 GB. */
    const int64_t ws = (int64_t)cfg->n_layers * cfg->topk;
    if ((int64_t)c->nslot < ws)
        fprintf(stderr,
                "dsv4_cache: %d slots (%.2f GB) is below one forward pass's "
                "working set of\n  %lld experts (%.2f GB). LRU cannot hit at "
                "this size -- every entry is evicted\n  before its layer comes "
                "round again. Raise --budget by at least %.1f GB.\n",
                c->nslot, (double)budget_bytes / 1073741824.0,
                (long long)ws, (double)(ws * c->expert_bytes) / 1073741824.0,
                (double)(ws * c->expert_bytes) / 1073741824.0);

    c->slot = (DSV4Slot *)calloc((size_t)c->nslot, sizeof(DSV4Slot));
    if (!c->slot) return -1;

    for (int i = 0; i < c->nslot; i++) {
        c->slot[i].layer = -1;
        c->slot[i].expert = -1;
        c->slot[i].pending = 0;
        /* 4096-aligned so a widened O_DIRECT window can be read straight in. */
        if (posix_memalign((void **)&c->slot[i].bytes, 4096,
                           (size_t)c->slot_bytes) != 0) {
            c->slot[i].bytes = NULL;
            dsv4_cache_free(c); return -1;
        }
        /* Until the first load, the compact layout is the honest description
         * of an empty slot -- bind_slot needs SOMETHING coherent to point at. */
        for (int k = 0; k < 3; k++) {
            c->slot[i].toff[k]     = c->off_s[k];
            c->slot[i].toff[3 + k] = c->off_w[k];
        }
        dsv4_cache_bind_slot(c, &c->slot[i]);
    }

    /* One FIFO for every reader job: normal begin() batches and the optional
     * hash prefetch. Overflow is a hard error, not a stall. */
    c->rq_cap = c->nslot + 64;
    if (c->rq_cap < 256) c->rq_cap = 256;
    c->rq = (struct DSV4CacheJob *)calloc((size_t)c->rq_cap, sizeof *c->rq);
    if (!c->rq) { dsv4_cache_free(c); return -1; }
    c->rq_head = c->rq_tail = c->rq_n = 0;

    dsv4_cache_pool_start(c);
    /* Hash prefetch is automatic; pf_has_room still decides whether this cache
     * is large enough. */
    (void)dsv4_cache_enable_prefetch(c);
    return 0;
}

void dsv4_cache_free(DSV4Cache *c)
{
    if (!c) return;
    /* Readers first, and before anything they point at is freed. */
    if (c->slot) dsv4_cache_pool_stop(c);
    for (int i = 0; i < c->nslot; i++) free(c->slot[i].bytes);
    free(c->slot);
    if (c->bounce) {
        for (int b = 0; b < c->nbounce; b++) free(c->bounce[b]);
        free(c->bounce);
    }
    free(c->rq);
    free(c->pf_batch);
    memset(c, 0, sizeof *c);
}

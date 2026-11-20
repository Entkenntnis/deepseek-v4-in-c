/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_evict.c - shrink the resident slot window for a block prefill.
 *
 * The arena is allocated once, so invalidating slot keys does not lower RSS.
 * The guard therefore waits for readers, empties the tail slots and asks the
 * kernel to drop their pages. It then caps claims to the remaining prefix for
 * the duration of the prefill: otherwise the prefill itself would refill the
 * whole cache while its block buffers are still alive.
 *
 * The caller supplies an extra margin. Vulkan imports the cache in 1 GiB
 * host-pointer chunks and importing a chunk that still holds a used slot
 * re-commits every page in that chunk, including pages the guard just dropped.
 * Evicting one extra chunk keeps that re-commit from exceeding the pre-guard
 * cache footprint.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"

static void drop_pages(unsigned char *base, size_t bytes)
{
#ifdef MADV_DONTNEED
    if (bytes > 0 && madvise(base, bytes, MADV_DONTNEED) != 0)
        fprintf(stderr, "cache: madvise(DONTNEED) failed: %s\n",
                strerror(errno));
#else
    (void)base;
    (void)bytes;
#endif
}

void dsv4_cache_guard_begin(DSV4Cache *c, int64_t reserve_bytes,
                            int64_t extra_bytes, DSV4CacheGuard *g)
{
    if (!c || !g) return;
    g->old_limit = c->limit;

    /* No batch may be live at the turn boundary. A hash prefetch may still be
     * draining, so wait for every pending slot and queued reader job. */
    int old_limit;
    int keep;
    pthread_mutex_lock(&c->rmx);
    for (;;) {
        int pending = 0, queued = c->rq_n;
        for (int i = 0; i < c->nslot; i++)
            if (c->slot[i].pending) { pending = 1; break; }
        if (!pending && queued == 0) break;
        pthread_cond_wait(&c->rdone, &c->rmx);
    }

    old_limit = c->limit;
    const int64_t reserve_total = reserve_bytes + (extra_bytes > 0 ? extra_bytes : 0);
    const int64_t slots = reserve_total > 0
        ? (reserve_total + c->slot_bytes - 1) / c->slot_bytes : 0;
    keep = old_limit - (int)slots;
    if (keep < 0) keep = 0;

    /* The prefill batch claims one slot per distinct expert in a layer. Below
     * that the block path cannot run at all. Do not invent a slower fallback:
     * if this still does not fit, the normal allocation is allowed to fail. */
    int min_keep = c->cfg ? c->cfg->n_experts : 0;
    if (min_keep > DSV4_MAX_PREFETCH) min_keep = DSV4_MAX_PREFETCH;
    if (keep < min_keep) keep = min_keep;
    if (keep > old_limit) keep = old_limit;

    for (int i = keep; i < old_limit; i++) {
        DSV4Slot *s = &c->slot[i];
        s->layer = s->expert = -1;
        s->pending = 0;
        s->origin = 0;
        s->stamp = 0;
    }
    c->limit = keep;
    pthread_mutex_unlock(&c->rmx);

    if (keep >= old_limit) return;

    if (c->slot_arena)
        drop_pages(c->slot[keep].bytes,
                   (size_t)(old_limit - keep) * (size_t)c->slot_bytes);
    else
        for (int i = keep; i < old_limit; i++)
            drop_pages(c->slot[i].bytes, (size_t)c->slot_bytes);

    fprintf(stderr, "cache: %d -> %d slots for %.1f MB prefill",
            old_limit, keep, (double)reserve_bytes / 1048576.0);
    if (extra_bytes > 0)
        fprintf(stderr, " + %.1f MB import slack",
                (double)extra_bytes / 1048576.0);
    fputc('\n', stderr);
}

void dsv4_cache_guard_end(DSV4Cache *c, DSV4CacheGuard *g)
{
    if (!c || !g) return;
    pthread_mutex_lock(&c->rmx);
    c->limit = g->old_limit;
    pthread_mutex_unlock(&c->rmx);
    g->old_limit = -1;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_get.c - the synchronous single-expert path. */
#include <stdio.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"
#include "dsv4_prof.h"

const DSV4ExpertW *dsv4_cache_get(DSV4Cache *c, int layer, int expert)
{
    if (c->route_log) fprintf(c->route_log, "%d %d\n", layer, expert);

    const DSV4Tensor *t[6];
    pthread_mutex_lock(&c->rmx);
    c->clock++;

    int found = -1;
    for (int i = 0; i < c->nslot; i++)
        if (c->slot[i].layer == layer && c->slot[i].expert == expert) {
            found = i; break;
        }
    if (found >= 0 && c->slot[found].pending) {
        const double t0 = dsv4_now_s();
        while (c->slot[found].pending)
            pthread_cond_wait(&c->rdone, &c->rmx);
        c->pf_wait_s += dsv4_now_s() - t0;
        if (c->slot[found].layer != layer || c->slot[found].expert != expert)
            found = -1;
    }
    if (found >= 0) {
        c->slot[found].stamp = ++c->clock;
        c->hits++;
        const DSV4ExpertW *w = &c->slot[found].w;
        pthread_mutex_unlock(&c->rmx);
        return w;
    }

    /* Resolve FIRST, so an unsatisfiable request never disturbs the cache. */
    if (dsv4_cache_resolve(c, layer, expert, t) != 0) {
        pthread_mutex_unlock(&c->rmx);
        return NULL;
    }

    const int victim = dsv4_cache_claim_slot(c, layer, expert, NULL, 0);
    if (victim < 0) {
        pthread_mutex_unlock(&c->rmx);
        return NULL;
    }
    DSV4Slot *s = &c->slot[victim];
    pthread_mutex_unlock(&c->rmx);

    uint64_t got = 0;
    const int bad = dsv4_cache_load(c, s, t, c->bounce[0], &got) != 0;

    pthread_mutex_lock(&c->rmx);
    s->pending = 0;
    if (bad) {
        s->layer = s->expert = -1;       /* do not leave a half-read slot live */
    } else {
        c->misses++;
        c->bytes_read += got;
    }
    pthread_mutex_unlock(&c->rmx);
    return bad ? NULL : &s->w;
}

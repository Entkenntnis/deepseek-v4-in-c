/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_stream.c - get_many, the stream cap and wait_any(). */
#include "dsv4_cache.h"
#include "dsv4_cache_int.h"

int dsv4_cache_get_many(DSV4Cache *c, int layer, const int *experts, int n,
                        const DSV4ExpertW **out)
{
    DSV4CacheBatch b;
    const int rc  = dsv4_cache_get_many_begin(c, layer, experts, n, out, &b);
    const int rc2 = dsv4_cache_get_many_end(c, out, &b);
    return rc != 0 ? rc : rc2;
}

void dsv4_cache_set_stream_cap(DSV4Cache *c, int cap)
{
    if (!c) return;
    if (cap < 0) cap = 0;
    if (cap > DSV4_MAX_TOPK) cap = DSV4_MAX_TOPK;
    c->stream_cap = cap;
}

int dsv4_cache_wait_any(DSV4Cache *c, DSV4CacheBatch *b, int *k)
{
    if (!c || !b || !k) return 0;
    if (!b->streaming) return 0;

    pthread_mutex_lock(&c->rmx);
    for (;;) {
        for (int j = 0; j < b->ntodo; j++) {
            if (!b->published[j] || b->taken[j]) continue;
            b->taken[j] = 1;
            *k = b->k_of[j];
            pthread_mutex_unlock(&c->rmx);
            return 1;
        }
        if (b->rdone >= b->ntodo) {
            pthread_mutex_unlock(&c->rmx);
            return 0;                      /* everything consumed */
        }
        pthread_cond_wait(&c->rdone, &c->rmx);
    }
}

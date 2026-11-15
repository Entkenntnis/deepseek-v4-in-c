/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_int.h - slot layout and seams between the cache files. */
#ifndef DSV4_CACHE_INT_H
#define DSV4_CACHE_INT_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_cache.h"

typedef struct DSV4Slot {
    /* Key. layer < 0 and pending == 0 means empty. While a reader owns the
     * slot the key is already set and pending is 1, so a later request for the
     * same expert can wait for this load instead of issuing a duplicate read. */
    int      layer, expert;
    int      pending;
    uint8_t  origin;           /* 0 empty, 1 hash prefetch, 2 a miss read */
    uint64_t stamp;            /* for LRU */
    DSV4ExpertW w;             /* points into `bytes` */
    unsigned char *bytes;      /* 4096-aligned; O_DIRECT reads land here */
    /* Where each of the six tensors ended up, in the order
     * (w1.s, w2.s, w3.s, w1.w, w2.w, w3.w). Recomputed on every load. */
    int64_t  toff[6];
} DSV4Slot;

typedef struct {
    DSV4Cache *c;
    int        bi;          /* which bounce buffer this reader owns */
} ReaderArg;

struct DSV4CacheJob {
    DSV4CacheBatch *b;
    int             j;
};

int  dsv4_cache_expert_geometry(DSV4Cache *c, const DSV4Cfg *cfg);
int  dsv4_cache_enable_prefetch(DSV4Cache *c);
void dsv4_cache_bind_slot(DSV4Cache *c, DSV4Slot *s);
int  dsv4_cache_rq_push(DSV4Cache *c, DSV4CacheBatch *b, int j);
int  dsv4_cache_resolve(DSV4Cache *c, int layer, int expert,
                        const DSV4Tensor *out[6]);
int  dsv4_cache_load(DSV4Cache *c, DSV4Slot *s, const DSV4Tensor *t[6],
                     unsigned char *bounce, uint64_t *bytes_read);
int  dsv4_cache_claim_slot(DSV4Cache *c, int layer, int expert,
                           const int *claimed, int nclaim);
void dsv4_cache_pool_start(DSV4Cache *c);
void dsv4_cache_pool_stop(DSV4Cache *c);

#endif /* DSV4_CACHE_INT_H */

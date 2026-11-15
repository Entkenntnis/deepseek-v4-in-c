/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache.h - LRU cache for the streamed routed experts.
 * The sizing measurements, the alignment argument and the overlap contract are
 * in README.md. */
#ifndef DSV4_CACHE_H
#define DSV4_CACHE_H

#include <pthread.h>
#include <stdio.h>

#include "dsv4.h"
#include "dsv4_st.h"

/* Slack so each coalesced run can sit at the 4096 residue its FILE offset
 * demands, which is what lets O_DIRECT land straight in the slot. Two runs in
 * practice; four runs' worth is bought so placement never has to think. */
#define DSV4_SLOT_SLACK (6 * 4096)

/* Upper bound on one hash-layer prefetch batch: num_hash_layers * topk.
 * 18 on both released models; 64 leaves room and keeps the batch fixed-size. */
#define DSV4_MAX_PREFETCH 64

struct DSV4CacheBatch;
struct DSV4CacheJob;
struct DSV4Slot;

typedef struct {
    const DSV4St  *st;
    const DSV4Cfg *cfg;
    struct DSV4Slot *slot;
    int            nslot;
    int64_t        expert_bytes;
    uint64_t       clock;

    /* Offsets of the six tensors inside a slot, computed once. */
    int64_t  off_w[3], off_s[3];
    int64_t  len_w[3], len_s[3];
    /* Bytes allocated per slot: the payload plus DSV4_SLOT_SLACK. expert_bytes
     * stays the true payload size so "GB read" keeps its meaning. */
    int64_t  slot_bytes;

    /* One 4096-aligned staging buffer per reader, plus index 0 for the
     * synchronous get(). Only a fallback: a run that does not fit its residue
     * placement lands here and is copied. */
    unsigned char **bounce;
    int             nbounce;
    int64_t         bounce_cap;
    int            direct;      /* 1 when the shards opened O_DIRECT */

    /* Optional routing log: every (layer, expert) request in issue order. */
    FILE *route_log;

    /* READER POOL: threads that exist only to sit inside pread(). They are
     * pthreads, not an OpenMP team, because the caller is expected to run its
     * own full-width matmul meanwhile. See README.md. */
    pthread_t      *rth;
    int             nrth;
    pthread_mutex_t rmx;
    pthread_cond_t  rwork, rdone;
    struct DSV4CacheJob *rq;         /* FIFO of reader jobs */
    int             rq_cap, rq_head, rq_tail, rq_n;
    void           *rargs;           /* per-reader {cache, bounce index} */
    int             rshutdown;

    /* stats */
    uint64_t hits, misses, evictions;
    uint64_t bytes_read;

    /* Publish each miss as its read lands and queue at most this many reads at
     * once. 0 keeps the historical end() barrier. This changes WHEN a finished
     * expert becomes usable, never WHICH slot was chosen. */
    int             stream_cap;

    /* HASH-LAYER PREFETCH: uses the same slots and reader pool. */
    int             pf_enabled;
    struct DSV4CacheBatch *pf_batch;
    uint64_t pf_requests, pf_hits, pf_loads, pf_fail, pf_bytes;
    double   pf_wait_s;
} DSV4Cache;

/* Bytes one routed expert occupies, from config alone. The budget planner
 * reserves room for topk of them BEFORE giving the trunk its share. */
int64_t dsv4_cache_expert_bytes(const DSV4Cfg *cfg);

/* What one slot costs: payload plus alignment slack. This, not expert_bytes, is
 * what a budget is divided by -- the slack is real resident memory. */
int64_t dsv4_cache_slot_bytes(const DSV4Cfg *cfg);

/* budget_bytes decides the slot count. Returns 0 on success. A budget below one
 * expert is an error. */
int  dsv4_cache_init(DSV4Cache *c, const DSV4St *st, const DSV4Cfg *cfg,
                     int64_t budget_bytes);
void dsv4_cache_free(DSV4Cache *c);

/* ---- hash-layer prefetch (automatic; init enables it) -------------------- */
int dsv4_cache_enable_prefetch(DSV4Cache *c);
int dsv4_cache_prefetch(DSV4Cache *c, const int *layers, const int *experts,
                        int n);
int dsv4_cache_prefetch_hash_layers(DSV4Cache *c, int token_id);

/* Fetch one expert, loading it if absent. NULL means a read failure, which
 * callers must treat as fatal: routing a token through fewer experts than the
 * model specifies is a silently different model. */
const DSV4ExpertW *dsv4_cache_get(DSV4Cache *c, int layer, int expert);

/* Fetch n experts of one layer at once, reading the misses concurrently.
 * out[k] is NULL for any expert that failed to load. Returns 0 if all loaded. */
int dsv4_cache_get_many(DSV4Cache *c, int layer, const int *experts, int n,
                        const DSV4ExpertW **out);

/* ---- overlapped fetching ------------------------------------------------
 * begin() does the whole serial decide pass, fills out[k] for every HIT, leaves
 * NULL for misses and hands them to the pool. end() waits and fills the rest.
 * Between the two, the caller owns the CPU. Cache state is decided in begin()
 * in request order, before any I/O, so output does not depend on read order.
 * The caller must still accumulate each k into its own buffer and sum in k
 * order. One batch per cache at a time. */
typedef struct DSV4CacheBatch {
    int layer, n, ntodo;
    int is_prefetch;                    /* persistent hash-prefetch batch */
    int streaming;                      /* publish misses as they land     */
    int next_j;
    int published[DSV4_MAX_PREFETCH];
    int taken[DSV4_MAX_PREFETCH];
    const DSV4ExpertW **out;
    int rdone;
    int slot_of[DSV4_MAX_PREFETCH];
    int k_of[DSV4_MAX_PREFETCH];
    const DSV4Tensor *t_of[DSV4_MAX_PREFETCH][6];
    int expert_of[DSV4_MAX_PREFETCH];
    int layer_of[DSV4_MAX_PREFETCH];
    int failed[DSV4_MAX_PREFETCH];
    uint64_t got[DSV4_MAX_PREFETCH];
    /* Requests for an expert another request in this batch is already
     * reading; they cannot be answered until that read lands. */
    int ndup, dup_k[DSV4_MAX_PREFETCH], dup_j[DSV4_MAX_PREFETCH];
    int rc;
} DSV4CacheBatch;

int dsv4_cache_get_many_begin(DSV4Cache *c, int layer, const int *experts,
                              int n, const DSV4ExpertW **out,
                              DSV4CacheBatch *b);
int dsv4_cache_get_many_end(DSV4Cache *c, const DSV4ExpertW **out,
                            DSV4CacheBatch *b);

/* Set how many miss reads one batch keeps in flight. 0 restores the barrier. */
void dsv4_cache_set_stream_cap(DSV4Cache *c, int cap);

/* Block until one miss has landed; *k indexes the caller's out array. Returns 1
 * when one was handed out, 0 when every miss has been consumed. */
int dsv4_cache_wait_any(DSV4Cache *c, DSV4CacheBatch *b, int *k);

void dsv4_cache_reset_stats(DSV4Cache *c);
void dsv4_cache_report(const DSV4Cache *c, const char *label);

#endif /* DSV4_CACHE_H */

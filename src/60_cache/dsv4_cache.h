/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache.h - LRU cache for the streamed routed experts. */
#ifndef DSV4_CACHE_H
#define DSV4_CACHE_H

#include <pthread.h>

#include "dsv4.h"
#include "dsv4_st.h"

/* Slack so each coalesced run can sit at the 4096 residue its FILE offset
 * demands, which is what lets O_DIRECT land straight in the slot. Two runs in
 * practice; four runs' worth is bought so placement never has to think. */
#define DSV4_SLOT_SLACK (6 * 4096)

/* Upper bound on one prefetch batch. Hash-layer prefetch needs
 * num_hash_layers * topk (18 on both released models); a whole-layer prefill
 * batch needs up to n_routed_experts (256 for Flash). Keeping both in one
 * fixed-size batch means begin() can queue the entire layer in one pass. */
#define DSV4_MAX_PREFETCH 256

struct DSV4CacheBatch;
struct DSV4CacheJob;
struct DSV4Slot;

typedef struct DSV4Cache {
    const DSV4St  *st;
    struct DSV4Trunk *tr;
    const DSV4Cfg *cfg;
    struct DSV4CacheBatch *batch;
    struct DSV4Slot *slot;
    int            nslot;
    /* Active window for claims. A prefill guard lowers this so the block
     * buffers fit next to the resident expert slots; the slots above the
     * window are empty and their pages were returned to the kernel. */
    int            limit;
    unsigned char *slot_arena;   /* one allocation for every slot, or NULL */
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

    /* READER POOL: threads that exist only to sit inside pread(). They are
     * pthreads, not an OpenMP team, because the caller is expected to run its
     * own full-width matmul meanwhile. */
    pthread_t      *rth;
    int             nrth;
    pthread_mutex_t rmx;
    pthread_cond_t  rwork, rdone;
    struct DSV4CacheJob *rq;         /* FIFO of reader jobs */
    int             rq_cap, rq_head, rq_tail, rq_n;
    void           *rargs;           /* per-reader {cache, bounce index} */
    int             rshutdown;

    /* Counters that feed the default TUI/e2e/prefill hit-rate and read-GB
     * readouts. Everything else about a read is deliberately not tracked. */
    uint64_t hits, misses;
    uint64_t bytes_read;
    uint64_t pf_bytes;      /* prefetch bytes are part of displayed read GB */

    /* Publish each miss as its read lands, keeping this many reads in flight.
     * Fixed at 6; this changes WHEN a finished expert becomes usable, never
     * WHICH slot was chosen. */
    int             stream_cap;

    /* HASH-LAYER PREFETCH: uses the same slots and reader pool. */
    int             pf_enabled;
    struct DSV4CacheBatch *pf_batch;
} DSV4Cache;

/* What one slot costs: payload plus alignment slack. This, not expert_bytes, is
 * what a budget is divided by -- the slack is real resident memory. */
int64_t dsv4_cache_slot_bytes(const DSV4Cfg *cfg);

/* budget_bytes decides the slot count. Returns 0 on success. A budget below one
 * expert is an error. */
int  dsv4_cache_init(DSV4Cache *c, const DSV4St *st, struct DSV4Trunk *tr,
                     const DSV4Cfg *cfg, int64_t budget_bytes);
void dsv4_cache_free(DSV4Cache *c);

/* ---- prefill memory guard ------------------------------------------------
 * Before a block prefill, drop enough LRU slots to make `reserve_bytes` of
 * resident memory available and cap claims to the remaining slots. The slots
 * above the new limit are invalidated and their pages are advised away with
 * MADV_DONTNEED; clearing keys alone would not lower RSS because the arena is
 * allocated up front. end() restores the full window after the block buffers
 * have been released. The guard must be taken between turns, when no batch is
 * in flight; it drains an in-flight hash prefetch itself. */
typedef struct {
    int old_limit;
} DSV4CacheGuard;

void dsv4_cache_guard_begin(DSV4Cache *c, int64_t reserve_bytes,
                            DSV4CacheGuard *g);
void dsv4_cache_guard_end(DSV4Cache *c, DSV4CacheGuard *g);

/* ---- hash-layer prefetch (automatic; init enables it) -------------------- */
int dsv4_cache_prefetch_hash_layers(DSV4Cache *c, int token_id);

/* ---- overlapped fetching ------------------------------------------------
 * begin() does the whole serial decide pass, fills out[k] for every HIT, leaves
 * NULL for misses and hands them to the pool. A slot the hash prefetch is
 * still reading is registered, not waited for: wait_any()/end() deliver it
 * when it publishes, exactly like a miss. end() waits and fills the rest.
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
    DSV4Loc t_of[DSV4_MAX_PREFETCH][6];
    int expert_of[DSV4_MAX_PREFETCH];
    int layer_of[DSV4_MAX_PREFETCH];
    int failed[DSV4_MAX_PREFETCH];
    uint64_t got[DSV4_MAX_PREFETCH];
    /* Slots the hash-layer prefetch was already reading when begin() ran.
     * They are NOT jobs of this batch: begin() neither waits for them nor
     * re-reads them. wait_any()/end() hand each one out the moment its
     * prefetch reader publishes, so the caller streams them like a miss. */
    int nwait;
    int wait_slot[DSV4_MAX_PREFETCH];
    int wait_k[DSV4_MAX_PREFETCH];
    int wait_expert[DSV4_MAX_PREFETCH];
    int wait_done[DSV4_MAX_PREFETCH];
    /* Requests for an expert another request in this batch is already
     * reading; they cannot be answered until that read lands. */
    int ndup, dup_k[DSV4_MAX_PREFETCH], dup_j[DSV4_MAX_PREFETCH];
    int rc;
} DSV4CacheBatch;

int dsv4_cache_get_many_begin(DSV4Cache *c, int layer, const int *experts,
                              int n, const DSV4ExpertW **out,
                              DSV4CacheBatch *b);

/* Engine-facing wrappers: one batch is owned by the cache. */
int dsv4_cache_begin(DSV4Cache *c, int layer, const int *experts, int n,
                     const DSV4ExpertW **out);
int dsv4_cache_end(DSV4Cache *c, const DSV4ExpertW **out);
int dsv4_cache_wait(DSV4Cache *c, int *k);
int dsv4_cache_try(DSV4Cache *c, int *k);
int dsv4_cache_get_many_end(DSV4Cache *c, const DSV4ExpertW **out,
                            DSV4CacheBatch *b);

/* Reorder `experts` into checkpoint order (shard, then w1.weight byte
 * offset), so the reader pool consumes one layer's extents as one increasing
 * stream instead of first-use order. Numerically neutral: the caller still
 * accumulates each request in k order. */
void dsv4_cache_sort_experts(const DSV4Cache *c, int layer, int *experts, int n);

/* Block until one miss has landed; *k indexes the caller's out array. Returns 1
 * when one was handed out, 0 when every miss has been consumed. A prefetch
 * slot that was still reading at begin() counts as one of these. */
int dsv4_cache_wait_any(DSV4Cache *c, DSV4CacheBatch *b, int *k);

/* Non-blocking variant: hand out one already-published miss, or return 0. */
int dsv4_cache_try_any(DSV4Cache *c, DSV4CacheBatch *b, int *k);

#endif /* DSV4_CACHE_H */

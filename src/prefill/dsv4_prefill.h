/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill.h - batched dense prompt prefill, CPU only.
 *
 * WHY THIS IS A SEPARATE PATH
 *   Decode is one token by one token: a weight matrix is read once and used for
 *   that token. Prefill has the opposite shape -- the same weight is needed by
 *   every prompt token -- so the right unit of work is a batch of tokens
 *   through one layer. Keeping the batch code here means the decode path keeps
 *   its bit-exact reference loop, its gates and its profile, and a change to
 *   one cannot silently move the other.
 *
 * WHAT IS BATCHED
 *   The dense matrices: q/kv/output projections, the shared expert and the
 *   router. Their weights are read once per chunk and used for every token in
 *   it. The routed experts stay on the per-token path -- at a 32-token prompt
 *   top-6 over 256 experts gives only ~2.9x reuse, and the expert union was
 *   measured to eat the cache hits that made the per-token path cheap.
 *
 * WHAT IS NOT BATCHED, DELIBERATELY
 *   mHC, RoPE, the compressor's sequential state, the CSA indexer, sparse
 *   attention and the routed experts. They either carry a position-ordered
 *   dependency or have no weight-reuse payoff. Only the projections are GEMMs.
 */
#ifndef DSV4_PREFILL_H
#define DSV4_PREFILL_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_prefill_gemm.h"
#include "dsv4_bind.h"
#include "dsv4_cache.h"
#include "dsv4_layer.h"
#include "dsv4_trunk.h"

/* Progress is layer-major: a chunk of tokens is walked through layer 0..N-1,
 * so a per-token "progress" is an estimate until the chunk completes. The
 * fields are deliberately explicit rather than a single percentage: a caller
 * can show the stage, the chunk, the layer and a bar without guessing. */
typedef struct {
    const char *stage;      /* start | chunk | layer | chunk_done | done */
    int chunk, chunks;      /* 1-based chunk index and total */
    int layer, layers;      /* completed layers in this chunk / total layers */
    int tokens_done;        /* tokens whose work is complete or estimated */
    int tokens_total;
    int chunk_tokens;
    int batch;
    double elapsed;         /* seconds since this prefill call began */
    double progress;        /* 0..1, estimated across the whole prompt */
} DSV4PrefillProgress;

typedef void (*DSV4PrefillProgressFn)(void *ctx,
                                      const DSV4PrefillProgress *p);

typedef struct {
    int batch;      /* tokens per chunk, 1..DSV4_MAX_BATCH. 1 = disabled */
    int max_pos;    /* same bound the decode scratch/state were built with */
    DSV4PrefillProgressFn on_progress; /* optional; NULL in one-shot use */
    void *progress_ctx;
} DSV4PrefillOpts;

typedef struct DSV4Prefill DSV4Prefill;

int  dsv4_prefill_init(DSV4Prefill **out, const DSV4Cfg *c, DSV4Trunk *tr,
                       DSV4LayerState *lst, const DSV4ExpertSrc *src,
                       const DSV4ModelW *mw, const DSV4PrefillOpts *opt);
void dsv4_prefill_free(DSV4Prefill *p);

/* Run tokens ids[0..n-1] through every layer, in chunks, at absolute
 * positions pos0..pos0+n-1. On return `h_out` is the last token's residual
 * stream, ready for the output head.
 *
 * `pos0` is not cosmetic. A resident engine (dsv4-serve) appends a new turn to
 * an existing KV cache, so the prefill must start where the previous turn
 * stopped. Passing ids from the start of the sequence together with pos0 != 0
 * is a caller bug: `ids` is the slice of NEW tokens. */
int dsv4_prefill_range(DSV4Prefill *p, const int32_t *ids, int n, int pos0,
                       const float *cs_d, const float *sn_d,
                       const float *cs_c, const float *sn_c,
                       float *h_out);

/* The one-shot case: a fresh sequence, positions 0..n-1. */
int dsv4_prefill_prompt(DSV4Prefill *p, const int32_t *ids, int n,
                        const float *cs_d, const float *sn_d,
                        const float *cs_c, const float *sn_c,
                        float *h_out);

#endif /* DSV4_PREFILL_H */

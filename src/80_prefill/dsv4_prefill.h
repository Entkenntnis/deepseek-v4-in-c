/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill.h - batched dense prompt prefill, CPU and optional Vulkan.
 *
 * WHY THIS IS A SEPARATE PATH
 *   Decode is one token by one token: a weight matrix is read once and used for
 *   that token. Prefill has the opposite shape -- the same weight is needed by
 *   every prompt token -- so the right unit of work is a batch of tokens
 *   through one layer. Keeping the batch code here means the decode path keeps
 *   its reference loop and gates, and a change to one cannot silently move the
 *   other. The model-level numerical contract is L2 relative error,
 *   not bit identity; `make test` holds the matmul gates.
 *
 * WHAT IS BATCHED
 *   The dense matrices: q/kv/output projections, the shared expert and the
 *   router, read once per block and used for every token in it. The routed
 *   experts are batched too, by layer-level union: every distinct expert the
 *   block routes to is requested once, and each expert's whole token bucket is
 *   run through one CPU FP4 batch kernel, or the FP4 Vulkan GEMM when that
 *   backend is available and the bucket is large enough.
 *
 * WHAT IS NOT BATCHED, DELIBERATELY
 *   mHC, RoPE, the compressor's sequential state, the CSA indexer and sparse
 *   attention. They either carry a position-ordered dependency or have no
 *   weight-reuse payoff. Only the projections are GEMMs.
 *
 * SCHEDULING
 *   The prompt is one block and is walked layer-major: every token of layer L
 *   runs before L+1. That keeps L's dense weights and routed-expert working set
 *   resident for the whole prompt. The CPU matmul entry point splits the block
 *   into fixed kernel tiles internally; the Vulkan backend tiles it too, with
 *   its own tile size. Either way a very long block does not turn the dense
 *   GEMMs into a DRAM stream. Prefill always initializes the Vulkan backend and
 *   uses it when available; the CPU kernels remain the fallback when the
 *   device is unavailable or a geometry is unsupported. `make e2e` exercises
 *   the selected path end to end.
 */
#ifndef DSV4_PREFILL_H
#define DSV4_PREFILL_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_matmul.h"
#include "dsv4_bind.h"
#include "dsv4_cache.h"
#include "dsv4_layer.h"
#include "dsv4_trunk.h"

/* Progress. The prefill block is the whole prompt: every layer walks every
 * token before the next layer starts. `layer` is the layer currently running
 * (1-based); `tokens_done` is an estimate while the block is mid-layer. */
typedef struct {
    const char *stage;      /* start | layer | done */
    int layer, layers;      /* current layer (1-based) / total layers */
    int tokens_done;        /* estimated from completed layers */
    int tokens_total;       /* tokens in this block */
    double elapsed;         /* seconds since this prefill call began */
    double progress;        /* 0..1, estimated across the block */
} DSV4PrefillProgress;

typedef void (*DSV4PrefillProgressFn)(void *ctx,
                                      const DSV4PrefillProgress *p);

/* Polled at layer and token boundaries. Returning nonzero asks the block to
 * stop; the callback must be cheap and must not touch the prefill state. */
typedef int (*DSV4PrefillCancelFn)(void *ctx);

typedef struct {
    int max_pos;    /* same bound the decode scratch/state were built with */
    DSV4PrefillProgressFn on_progress; /* optional; NULL in one-shot use */
    void *progress_ctx;
    DSV4PrefillCancelFn is_cancelled;  /* optional; NULL never cancels */
    void *cancel_ctx;
} DSV4PrefillOpts;

typedef struct DSV4Prefill DSV4Prefill;

int  dsv4_prefill_init(DSV4Prefill **out, const DSV4Cfg *c, DSV4Trunk *tr,
                       DSV4LayerState *lst, const DSV4ExpertSrc *src,
                       const DSV4ModelW *mw, const DSV4PrefillOpts *opt);
void dsv4_prefill_free(DSV4Prefill *p);

/* Allocate the block buffers for `n_tokens` before any model state is touched.
 * dsv4_prefill_range() calls this too, but a caller that wants a safe fallback
 * to the per-token path on allocation failure should call it first: after the
 * range starts writing KV/compressor state there is no way back. */
int  dsv4_prefill_reserve(DSV4Prefill *p, int n_tokens);

/* Release the block-sized buffers but keep the prefill engine usable. Call once
 * a range has finished: keeping them pinned turns the longest prompt ever seen
 * into a permanent memory high-water mark. */
void dsv4_prefill_release(DSV4Prefill *p);

/* Return codes for dsv4_prefill_range(), in addition to -1 for an error. */
#define DSV4_PREFILL_CANCELLED 1   /* stopped before model state was changed */
#define DSV4_PREFILL_ABORTED   2   /* stopped after KV/compressor state changed */

/* Run tokens ids[0..n-1] through every layer as one block, at absolute
 * positions pos0..pos0+n-1. On return `h_out` is the last token's residual
 * stream, ready for the output head.
 *
 * The optional is_cancelled callback is polled between layers and between the
 * tokens of the sequential stages. A cancelled block is not a shorter block:
 * the layer-major schedule means a mid-block stop can leave some layers' KV and
 * compressor rows written and others untouched. DSV4_PREFILL_CANCELLED means
 * no model state was touched (the caller can drop the new tokens and keep its
 * old context); DSV4_PREFILL_ABORTED means it was, and the caller must restore
 * a snapshot or reset the model state before reusing the engine.
 *
 * `pos0` is not cosmetic. A resident engine (dsv4-serve) appends a new turn to
 * an existing KV cache, so the prefill must start where the previous turn
 * stopped. Passing ids from the start of the sequence together with pos0 != 0
 * is a caller bug: `ids` is the slice of NEW tokens. */
int dsv4_prefill_range(DSV4Prefill *p, const int32_t *ids, int n, int pos0,
                       const float *cs_d, const float *sn_d,
                       const float *cs_c, const float *sn_c,
                       float *h_out);

#endif /* DSV4_PREFILL_H */

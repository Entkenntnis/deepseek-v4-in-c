/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer.h - one decoder layer, decode path. */
#ifndef DSV4_LAYER_H
#define DSV4_LAYER_H

#include "dsv4.h"

typedef struct DSV4Cache DSV4Cache;

/* Per-layer scratch, allocated once and reused. A decode step touches every
 * buffer here once per layer per token, so churning them would dominate.
 *
 * expert_acc holds ONE accumulator per k and must never alias the MoE input:
 * experts may finish in any order, but the weighted sum runs in k order.
 * shared_acc is separate because the streamed experts overwrite expert_acc.
 * qr is reused for the layer output and must hold max(q_lora, hidden). */
typedef struct {
    float *x1, *resid, *post, *comb, *mixes;
    float *qr, *q, *kv, *o, *ogrp;
    float *gate_scores, *gate_orig, *expert_gate, *expert_up, *expert_out;
    float *expert_acc;   /* topk buffers of `hidden`, contiguous */
    int    expert_acc_stride;
    float *shared_acc;
    float *comp_kv_in, *comp_sc_in;    /* compressor projections, one token */
    float *idx_q, *idx_w, *idx_scores; /* CSA indexer working set          */
    int    idx_cap;                    /* how many scores idx_scores holds  */
    float *attn_scratch;               /* [n_heads][sliding_window+index_topk] */
    int   *idxs;
    int    topk_idx[DSV4_MAX_TOPK];
    float  topk_w[DSV4_MAX_TOPK];
    void  *arena;
} DSV4Scratch;

/* max_pos must be the same bound passed to dsv4_state_init: the indexer's row
 * count is a function of it, so a scratch sized differently is sized by
 * guesswork. */
int  dsv4_scratch_init(DSV4Scratch *s, const DSV4Cfg *c, int max_pos);
void dsv4_scratch_free(DSV4Scratch *s);

/* Per-layer state that persists across tokens. Distinct from DSV4Scratch,
 * which is reused by every layer within one token. kv_cache is
 * [sliding_window + max_compressed][head_dim]: the window ring first, then the
 * compressed rows, which is why indexer offsets start at sliding_window.
 * comp_score must start at -INFINITY so unfilled slots contribute nothing. */
typedef struct {
    float *kv_cache;
    float *comp_kv, *comp_score;
    int    n_compressed;      /* rows written so far */
    void  *arena;
} DSV4LayerState;

int  dsv4_state_init(DSV4LayerState *st, const DSV4Cfg *c, int layer,
                     int max_pos);
void dsv4_state_free(DSV4LayerState *st);

/* Rollback point for an in-flight prefill.
 *
 * Only the mutable part of each layer's context is saved: the sliding-window
 * KV ring and the compressor's working arrays. Compressed KV rows are
 * append-only and n_compressed bounds every read, so restoring n_compressed
 * hides rows a cancelled block appended without copying those rows. The size
 * is fixed by the model (window + compressor geometry), not by the prompt, so
 * one snapshot can be saved again before every turn and restored on a
 * cancellation. */
typedef struct DSV4StateSnapshot DSV4StateSnapshot;

DSV4StateSnapshot *dsv4_state_snapshot_new(const DSV4Cfg *c,
                                           DSV4LayerState *lst);
void dsv4_state_snapshot_free(DSV4StateSnapshot *s);
int  dsv4_state_snapshot_save(DSV4StateSnapshot *s);
void dsv4_state_snapshot_restore(DSV4StateSnapshot *s);

/* `h` is [hc_mult][hidden] in and out. cs/sn are the layer's RoPE table; a
 * compressed layer takes the compress table, ratio 0 the plain one. */
void dsv4_layer_forward(float *h, const DSV4LayerW *w, const DSV4Cfg *c,
                        DSV4Scratch *s, DSV4Cache *cache,
                        DSV4LayerState *st, const float *cs, const float *sn,
                        int pos, int token_id);


/* Seams inside the decoder implementation. */
void dsv4_layer_attention(float *out, const float *x, const DSV4LayerW *w,
                          const DSV4Cfg *c, DSV4Scratch *s, DSV4LayerState *st,
                          const float *cs, const float *sn, int pos, int n_idx);

#endif /* DSV4_LAYER_H */

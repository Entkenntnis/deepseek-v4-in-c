/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_int.h - the block state and seams between prefill files. */
#ifndef DSV4_PREFILL_INT_H
#define DSV4_PREFILL_INT_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_vk.h"

/* Routed-expert scheduling classes. The scheduler owns the split, not the
 * bucket function: 1..4 always run on the CPU, 9+ always on the GPU, and 5..8
 * (the contested band) go to whichever side has capacity. A CPU fill bucket
 * can therefore be as large as the whole contested band. */
#define DSV4_PF_CPU_ALWAYS_MAX 4
#define DSV4_PF_GPU_ALWAYS_MIN 9
/* Start offloading 5..8 to the CPU once at least this many 9+ buckets are
 * still pending. With 2 pending high buckets the CPU may take a 5; each
 * further high bucket moves the CPU limit up by one, until it reaches 8. */
#define DSV4_PF_MID_BACKLOG    2
#define DSV4_PREFILL_FILL_MAX  (DSV4_PF_GPU_ALWAYS_MIN - 1)

/* One matrix over `n` token rows: the batched-matmul seam. It uses the Vulkan
 * backend when available and falls back to the CPU kernels otherwise. */
typedef struct {
    const DSV4QMat *m;
    const float    *x;
    float          *y;
    int             n;
} DSV4PMJob;

/* Per-token state the sequential prefill stages still need.
 *
 * Deliberately NOT a DSV4Scratch: that type carries the whole decode working
 * set (mHC, q/kv/o, router, shared expert, ...), none of which the block path
 * uses. Only expert_acc has to be per token -- it stays live from routing until
 * the weighted sum after the expert-union pass. Everything else here is consumed
 * inside the per-token attention loop, and the count==1 expert path borrows the
 * block-wide ex_* rows instead of owning more buffers. */
typedef struct {
    float *kv;              /* [head_dim] compressor output                  */
    float *idx_scores;      /* [max_pos/4 + 1] compressed-row scores         */
    int    idx_cap;
    int   *idxs;            /* [sliding_window + index_topk]                  */
    float *expert_acc;      /* [topk][hidden]                                 */
    int    expert_acc_stride;
    void  *arena;           /* one allocation backing the float buffers       */
} DSV4PrefillTok;

struct DSV4Prefill {
    const DSV4Cfg   *c;
    DSV4Trunk       *tr;
    DSV4LayerState  *lst;
    const DSV4ExpertSrc *src;
    const DSV4ModelW *mw;
    struct DSV4PrefillVk *vk;   /* NULL = no GPU backend, use CPU kernels */
    DSV4PrefillOpts  opt;
    int              cap;          /* allocated token rows in the block buffers */
    int              scratch_cap;  /* allocated DSV4PrefillTok entries */
    int              n;            /* tokens in the block being run */
    int              state_dirty;  /* a layer may have written KV/compressor state */

    float *H;          /* [B][hc*hidden] */
    float *x1;         /* [B][hidden] */
    float *resid;      /* [B][hc*hidden] */
    float *post;       /* [B][hc] */
    float *comb;       /* [B][hc*hc] */
    float *mixes;      /* [B][(2+hc)*hc] */
    float *qr;         /* [B][q_lora or hidden] */
    float *q;          /* [B][heads*head_dim] */
    float *kv;         /* [B][head_dim] */
    float *o;          /* [B][heads*head_dim] */
    float *ogrp;       /* [B][o_groups*o_lora] */
    float *xg, *yg;    /* per-group packing */
    float *gate;       /* [B][n_experts] */
    float *gate_orig;  /* [B][n_experts] */
    int   *topk_idx;   /* [B][topk] */
    float *topk_w;     /* [B][topk] */
    float *sh_gate;    /* [B][inter] */
    float *sh_up;      /* [B][inter] */
    float *sh_mid;     /* [B][inter] */
    float *sh_out;     /* [B][hidden] */
    float *out;        /* [B][hidden] */
    float *xh;         /* [hidden] embed scratch */
    DSV4PrefillTok *scratch; /* [B] per-token state for the sequential parts */
    /* dsv4_sparse_attn() is parallel over heads and needs one score row per
     * head: [n_heads][sliding_window + index_topk]. The token loop is serial,
     * so one block-wide buffer serves every token instead of one per token. */
    float *attn_scratch;

    /* Expert-union scheduling: one bucket per expert for the whole layer. */
    int   *ex_tok, *ex_k;    /* [B*topk] request -> token and k              */
    int   *ex_count;         /* [n_experts] requests per expert              */
    int   *ex_order;         /* [n_experts] distinct experts, first-use order*/
    int   *ex_off;           /* [n_experts+1] bucket start                   */
    int   *ex_fill;          /* [n_experts] prefix-sum fill cursor           */
    int   *ex_seen;          /* [n_experts] first-use dedup                  */
    float *ex_x;             /* [B][hidden] gathered activations             */
    float *ex_gate, *ex_up, *ex_mid; /* [B][moe_inter] expert working set   */
    float *ex_out;           /* [B][hidden] expert output before scatter     */

    /* A GPU bucket owns ex_* from its first gather until its last scatter.
     * The CPU buckets that fill the GPU wait use these separate rows, so the
     * two schedules can overlap without aliasing. */
    float *sm_x;
    float *sm_gate, *sm_up, *sm_mid;
    float *sm_out;

    /* Whole-layer expert prefetch. begin() is called as soon as routing is
     * known -- after the gate for scored layers, at the top of the layer for
     * hash layers -- and the batch stays live while the shared expert (and,
     * for hash layers, attention) runs. */
    const DSV4ExpertW *pf_ex[DSV4_MAX_PREFETCH];
    int      pf_n;           /* distinct experts prepared for this layer      */
    int      pf_layer;       /* layer the prepared batch belongs to           */
    int      pf_active;      /* begin() succeeded and end() is still owed     */
    int      pf_prepared;    /* counts/order are valid for pf_layer           */

    /* Per-token projections that carry no position dependency are GEMMs too.
     * The compressor and indexer weights are read once per block instead of
     * once per token: [B][2*head_dim] twice, then [B][index heads*dim] and
     * [B][index heads]. The sequential parts (compress_step, RoPE, scoring)
     * stay per token and read these rows by index. */
    float *comp_kv_b, *comp_sc_b;
    float *idx_qb, *idx_wb;
};

/* Cancellation polling. The caller can distinguish a stop before any layer
 * touched model state from one that left partially written KV/compressor rows;
 * only the latter needs a context reset. */
static inline int dsv4_prefill_cancelled(const DSV4Prefill *p)
{
    return p->opt.is_cancelled && p->opt.is_cancelled(p->opt.cancel_ctx);
}

static inline int dsv4_prefill_cancel_code(const DSV4Prefill *p)
{
    return p->state_dirty ? DSV4_PREFILL_ABORTED : DSV4_PREFILL_CANCELLED;
}

/* Batched-matmul seam. */
int dsv4_prefill_pm(DSV4Prefill *p, float *y, const float *x,
                    const DSV4QMat *m, int n);
int dsv4_prefill_pm_multi(DSV4Prefill *p, const DSV4PMJob *jobs, int nj);

/* Sequential stages that still run per token. */
int  dsv4_prefill_attention(DSV4Prefill *p, int L, const DSV4LayerW *w,
                            int n, int pos0, const float *cs, const float *sn);
int  dsv4_prefill_oproj(DSV4Prefill *p, const DSV4LayerW *w, int n);
int  dsv4_prefill_moe(DSV4Prefill *p, int L, const DSV4LayerW *w,
                      int n, const int32_t *ids);

void dsv4_prefill_expert(DSV4Prefill *p, float *out, const float *x,
                         const DSV4ExpertW *e, int hidden, int inter,
                         float limit);
/* Whether this bucket is in the always-GPU class. */
int  dsv4_prefill_expert_is_gpu(const DSV4Prefill *p, int count);
/* Explicit routing for the scheduler-visible bands. AUTO keeps the old
 * count-based decision (9+ -> GPU). CPU and GPU are used by the union
 * scheduler so it can place a contested 5..8 bucket on either backend. */
enum {
    DSV4_EXPERT_ROUTE_AUTO = -1,
    DSV4_EXPERT_ROUTE_CPU  =  0,
    DSV4_EXPERT_ROUTE_GPU  =  1
};
/* fill/fill_ctx are optional: when present and the bucket runs on the GPU,
 * they run CPU buckets between each submit and fence wait. */
void dsv4_prefill_expert_bucket(DSV4Prefill *p, const DSV4ExpertW *ew,
                                int expert, int count, const int *tok,
                                const int *ks, int route,
                                DSV4PrefillVkFillFn fill, void *fill_ctx);
/* CPU-only bucket using the sm_* workspace. Never submits GPU work. */
void dsv4_prefill_expert_bucket_fill(DSV4Prefill *p, const DSV4ExpertW *ew,
                                     int expert, int count, const int *tok,
                                     const int *ks);
/* Start a whole-layer expert batch early. Returns 0 when the batch is live or
 * when the layer is too large for one batch (the union then groups it); the
 * caller must still run dsv4_prefill_moe_union(), which ends it. */
int  dsv4_prefill_moe_prefetch(DSV4Prefill *p, int L, int n);
int  dsv4_prefill_moe_union(DSV4Prefill *p, int L, int n);
/* Drain any live prefetch batch on an error/cancel path. Safe when idle. */
void dsv4_prefill_moe_abort(DSV4Prefill *p);

/* Progress only. */
void dsv4_prefill_progress(DSV4Prefill *p, const char *stage, int layer,
                           int tokens_done, int tokens_total,
                           double elapsed, double progress);

#endif /* DSV4_PREFILL_INT_H */

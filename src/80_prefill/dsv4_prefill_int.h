/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_int.h - the block state and seams between prefill files. */
#ifndef DSV4_PREFILL_INT_H
#define DSV4_PREFILL_INT_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_prefill.h"

/* Fine-grained accounting for one dsv4_prefill_range call. The public profile
 * only knows broad buckets, which cannot tell a kernel problem from a
 * scheduling problem. f_* are GFLOP (2 x MAC), so effective GF/s is
 * f_family / seconds_in_family. Timers nest: do not add the columns up. */
typedef struct {
    double embed, hc_pre, hc_post, hc_post_moe, norm;
    double proj_qkv, q_norm, q_head, kv_norm;
    double comp_proj, comp_step, win_idx;
    double idx_proj, idx_rope, idx_score;
    double attn_core, attn_deroep;
    double oproj_pack, oproj, oproj_unpack;
    double gate_mm, route, shared_mm, shared_swiglu;
    double expert_io, expert_mm, expert_swiglu, expert_pack, expert_sum;
    double bind, trunk_io;

    double f_qkv, f_oproj, f_gate, f_shared, f_expert, f_comp, f_idx, f_attn;

    uint64_t e_hits, e_misses, e_bytes, trunk_bytes;
    int have_estats;
} DSV4PrefillStats;

/* One matrix over `n` token rows: the CPU seam the prefill module is built on. */
typedef struct {
    const DSV4QMat *m;
    const float    *x;
    float          *y;
    int             n;
} DSV4PMJob;

struct DSV4Prefill {
    const DSV4Cfg   *c;
    DSV4Trunk       *tr;
    DSV4LayerState  *lst;
    const DSV4ExpertSrc *src;
    const DSV4ModelW *mw;
    DSV4PrefillOpts  opt;
    int              cap;          /* allocated token rows in the block buffers */
    int              scratch_cap;  /* allocated DSV4Scratch entries */
    int              n;            /* tokens in the block being run */

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
    DSV4Scratch *scratch; /* [B] per-token scratch for the sequential parts */

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

    double t_total, t_attn, t_idx, t_gate, t_shared, t_moe;
    double t_io, t_trunk, t_bind, t_hc;

    DSV4PrefillStats st;   /* per-call sub-stage accounting */
};

/* Batched-matmul seam and its timed wrappers. */
int dsv4_prefill_pm(DSV4Prefill *p, int layer, const DSV4PMJob *jobs, int nj);
int dsv4_prefill_pm1(DSV4Prefill *p, int layer, float *y, const float *x,
                     const DSV4QMat *m, int n);
int dsv4_prefill_pm_stats(DSV4Prefill *p, float *y, const float *x,
                          const DSV4QMat *m, int n, double *timer,
                          double *flops);
int dsv4_prefill_pm_multi_stats(DSV4Prefill *p, const DSV4PMJob *jobs, int nj,
                                double *timer, double *flops);

/* Sequential stages that still run per token. */
int  dsv4_prefill_attention(DSV4Prefill *p, int L, const DSV4LayerW *w,
                            int n, int pos0, const float *cs, const float *sn);
int  dsv4_prefill_oproj(DSV4Prefill *p, const DSV4LayerW *w, int n);
int  dsv4_prefill_moe(DSV4Prefill *p, int L, const DSV4LayerW *w,
                      int n, const int32_t *ids);

void dsv4_prefill_expert(DSV4Prefill *p, float *out, const float *x,
                         const DSV4ExpertW *e, DSV4Scratch *s,
                         int hidden, int inter, float limit);
void dsv4_prefill_expert_bucket(DSV4Prefill *p, const DSV4ExpertW *ew,
                                int count, const int *tok, const int *ks);
int  dsv4_prefill_moe_union(DSV4Prefill *p, int L, int n);

/* Reporting. */
void dsv4_prefill_progress(DSV4Prefill *p, const char *stage, int layer,
                           int tokens_done, int tokens_total,
                           double elapsed, double progress);
void dsv4_prefill_report(DSV4Prefill *p, int n);

#endif /* DSV4_PREFILL_INT_H */

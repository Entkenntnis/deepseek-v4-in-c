/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_kernels.h - the compute kernels that are not the packed matmul.
 *
 * These entry points used to be declared by hand in dsv4_layer.c,
 * dsv4_prefill.c, dsv4_serve.c and the unit tests. That is one header per
 * consumer too many: a signature change had to be repeated in four files and
 * the compiler could not point at the drift until link time.
 *
 * This header is the one declaration. It contains no inline code and no
 * state; include dsv4.h for the types.
 */
#ifndef DSV4_KERNELS_H
#define DSV4_KERNELS_H

#include <stdint.h>

#include "dsv4.h"

/* elementwise + router */
void dsv4_rmsnorm(float *y, const float *x, const float *w, int n, float eps);
void dsv4_swiglu(float *y, const float *gate, const float *up, int n,
                 float limit);
void dsv4_route(int *idx, float *wts, float *scores, float *orig,
                const float *bias, const int64_t *tid2eid, int token_id,
                int n_experts, int topk, float route_scale);

/* RoPE */
void dsv4_rope_table(float *cs, float *sn, int dim, int seqlen,
                     int orig_seq_len, double base, double factor,
                     double beta_fast, double beta_slow);
void dsv4_rope_apply(float *v, int head_dim, int rd, const float *cs,
                     const float *sn, int pos, int table_half, int inverse);

/* sparse attention */
void dsv4_sparse_attn(float *o, const float *q, const float *kv,
                      const float *sink, const int *idxs, int h, int d,
                      int topk, float scale, float *scratch);
int  dsv4_window_idxs(int *idxs, int pos, int win);

/* HCA compressor */
int  dsv4_compress_step(float *out, const float *kv_in, const float *score_in,
                        const float *ape, float *kv_state, float *score_state,
                        int pos, int ratio, int head_dim);
int  dsv4_compress_rope_pos(int pos, int ratio);

/* CSA indexer */
void dsv4_indexer_score(float *out, const float *q, const float *kv,
                        const float *weights, int n_heads, int head_dim,
                        int n_pos);
int  dsv4_topk(int *idxs, const float *scores, int n, int k);
void dsv4_indexer_offset(int *idxs, int n, int offset);

/* mHC */
void dsv4_hc_split_sinkhorn(float *pre, float *post, float *comb,
                            const float *mixes, const float *hc_scale,
                            const float *hc_base, int hc, int iters,
                            float eps);
void dsv4_hc_pre(float *y, float *post, float *comb, float *mixes,
                 const float *x, const DSV4HcW *w, int hc, int hidden,
                 int iters, float norm_eps, float hc_eps);
void dsv4_hc_post(float *y, const float *x, const float *residual,
                  const float *post, const float *comb, int hc, int hidden);

#endif /* DSV4_KERNELS_H */

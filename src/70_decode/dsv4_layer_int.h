/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer_int.h - seams between the decode-files. Not a public header. */
#ifndef DSV4_LAYER_INT_H
#define DSV4_LAYER_INT_H

#include "dsv4.h"
#include "dsv4_layer.h"

/* Attention for one token at absolute position `pos`. kv_cache is the layer's
 * ring; idxs already holds the window slots and any compressed selections. */
void dsv4_layer_attention(float *out, const float *x, const DSV4LayerW *w,
                          const DSV4Cfg *c, DSV4Scratch *s, DSV4LayerState *st,
                          const float *cs, const float *sn, int pos, int n_idx);

/* MoE for one token. token_id is only read on hash-routed layers. */
void dsv4_layer_moe(float *out, const float *x, const DSV4LayerW *w,
                    const DSV4Cfg *c, DSV4Scratch *s, const DSV4ExpertSrc *src,
                    int layer, int token_id);

#endif /* DSV4_LAYER_INT_H */

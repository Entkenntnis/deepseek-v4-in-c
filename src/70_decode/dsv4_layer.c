/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer.c - one decoder layer, decode path. */
#include <string.h>

#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_layer.h"
#include "dsv4_layer_int.h"

/* One decoder layer. `h` is [hc_mult][hidden] in and out. */
void dsv4_layer_forward(float *h, const DSV4LayerW *w, const DSV4Cfg *c,
                        DSV4Scratch *s, const DSV4ExpertSrc *src,
                        DSV4LayerState *st, const float *cs, const float *sn,
                        int pos, int token_id)
{
    const int hc = c->hc_mult, d = c->hidden;
    const size_t wide = (size_t)hc * d;

    /* ---- attention half ---- */
    memcpy(s->resid, h, wide * sizeof(float));
    dsv4_hc_pre(s->x1, s->post, s->comb, s->mixes, h, &w->hc_attn,
                hc, d, c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
    dsv4_rmsnorm(s->x1, s->x1, w->attn_norm, d, c->rms_eps);

    const int n_idx = dsv4_window_idxs(s->idxs, pos, c->sliding_window);
    dsv4_layer_attention(s->qr, s->x1, w, c, s, st, cs, sn, pos, n_idx);

    dsv4_hc_post(h, s->qr, s->resid, s->post, s->comb, hc, d);

    /* ---- FFN half. The residual is taken HERE, not at the top. ---- */
    memcpy(s->resid, h, wide * sizeof(float));
    dsv4_hc_pre(s->x1, s->post, s->comb, s->mixes, h, &w->hc_ffn,
                hc, d, c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
    dsv4_rmsnorm(s->x1, s->x1, w->ffn_norm, d, c->rms_eps);

    dsv4_layer_moe(s->qr, s->x1, w, c, s, src, w->layer, token_id);

    dsv4_hc_post(h, s->qr, s->resid, s->post, s->comb, hc, d);
}

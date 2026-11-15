/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill.c - one block through every layer. See README.md. */
#include <stdio.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_prof.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

static int prefill_layer(DSV4Prefill *p, int L, const DSV4LayerW *w,
                         int n, int pos0, const int32_t *ids,
                         const float *cs, const float *sn)
{
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const int mix_hc = (2 + hc) * hc;
    const double thc = dsv4_now_s();

    for (int t = 0; t < n; t++) {
        double ts = dsv4_now_s();
        memcpy(p->resid + (size_t)t * hcd, p->H + (size_t)t * hcd,
               hcd * sizeof(float));
        dsv4_hc_pre(p->x1 + (size_t)t * d,
                    p->post + (size_t)t * hc,
                    p->comb + (size_t)t * hc * hc,
                    p->mixes + (size_t)t * mix_hc,
                    p->H + (size_t)t * hcd, &w->hc_attn, hc, d,
                    c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
        p->st.hc_pre += dsv4_now_s() - ts;
        ts = dsv4_now_s();
        dsv4_rmsnorm(p->x1 + (size_t)t * d, p->x1 + (size_t)t * d,
                     w->attn_norm, d, c->rms_eps);
        p->st.norm += dsv4_now_s() - ts;
    }
    p->t_hc += dsv4_now_s() - thc;

    if (dsv4_prefill_attention(p, L, w, n, pos0, cs, sn)) return -1;

    const double thc2 = dsv4_now_s();
    for (int t = 0; t < n; t++) {
        double ts = dsv4_now_s();
        memcpy(p->resid + (size_t)t * hcd, p->H + (size_t)t * hcd,
               hcd * sizeof(float));
        dsv4_hc_pre(p->x1 + (size_t)t * d,
                    p->post + (size_t)t * hc,
                    p->comb + (size_t)t * hc * hc,
                    p->mixes + (size_t)t * mix_hc,
                    p->H + (size_t)t * hcd, &w->hc_ffn, hc, d,
                    c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
        p->st.hc_pre += dsv4_now_s() - ts;
        ts = dsv4_now_s();
        dsv4_rmsnorm(p->x1 + (size_t)t * d, p->x1 + (size_t)t * d,
                     w->ffn_norm, d, c->rms_eps);
        p->st.norm += dsv4_now_s() - ts;
    }
    p->t_hc += dsv4_now_s() - thc2;

    if (dsv4_prefill_moe(p, L, w, n, ids)) return -1;
    return 0;
}

int dsv4_prefill_range(DSV4Prefill *p, const int32_t *ids, int n, int pos0,
                       const float *cs_d, const float *sn_d,
                       const float *cs_c, const float *sn_c,
                       float *h_out)
{
    if (!p || !ids || n <= 0 || !h_out || pos0 < 0) return -1;
    if (dsv4_prefill_reserve(p, n) != 0) return -1;
    p->n = n;
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const double t0 = dsv4_now_s();

    /* A resident engine calls this once per turn. Reset so this turn's profile
     * is not the previous turn's plus this one. */
    p->t_total = p->t_attn = p->t_idx = p->t_gate = p->t_shared = 0.0;
    p->t_moe = p->t_io = p->t_trunk = 0.0;
    p->t_bind = p->t_hc = 0.0;
    memset(&p->st, 0, sizeof p->st);

    DSV4ExpertStats es0, es1;
    memset(&es0, 0, sizeof es0);
    memset(&es1, 0, sizeof es1);
    if (p->src->stats && p->src->stats(p->src->ctx, &es0) == 0)
        p->st.have_estats = 1;
    const uint64_t trunk0 = p->tr ? p->tr->bytes_read : 0;

    dsv4_prefill_progress(p, "start", 0, 0, n, 0.0, 0.0);

    /* ONE BLOCK, LAYER-MAJOR. The whole prompt is the block: every token
     * passes layer L before any token enters L+1, which keeps the layer's dense
     * weights and routed experts resident for the entire prompt. */
    const double te = dsv4_now_s();
    for (int t = 0; t < n; t++) {
        dsv4_embed_row(p->xh, p->mw->embed, p->mw->wdt, ids[t], d);
        for (int j = 0; j < hc; j++)
            memcpy(p->H + (size_t)t * hcd + (size_t)j * d, p->xh,
                   (size_t)d * sizeof(float));
    }
    p->st.embed += dsv4_now_s() - te;

    for (int L = 0; L < c->n_layers; L++) {
        DSV4LayerBind lb;
        const double tb = dsv4_now_s();
        const double io0 = p->tr->load_seconds;
        if (dsv4_trunk_bind(p->tr, c, L, &lb) != 0) return -1;
        const double io = p->tr->load_seconds - io0;
        const double call = dsv4_now_s() - tb;
        p->t_io += io;
        p->t_trunk += io;
        p->t_bind += call > io ? call - io : 0.0;
        p->st.trunk_io += io;
        p->st.bind += call > io ? call - io : 0.0;

        const int comp = dsv4_compress_ratio(c, L) != 0;
        const float *cs = comp ? cs_c : cs_d;
        const float *sn = comp ? sn_c : sn_d;

        /* `layer` is 1-based and names the layer being entered; progress is
         * the completed fraction, so the bar moves when a layer finishes. */
        const double frac = (double)L / (double)c->n_layers;
        dsv4_prefill_progress(p, "layer", L + 1, (int)(frac * (double)n), n,
                              dsv4_now_s() - t0, frac);
        if (prefill_layer(p, L, &lb.w, n, pos0, ids, cs, sn)) return -1;
    }

    dsv4_prefill_progress(p, "done", c->n_layers, n, n, dsv4_now_s() - t0, 1.0);
    memcpy(h_out, p->H + (size_t)(n - 1) * hcd, hcd * sizeof(float));
    p->t_total = dsv4_now_s() - t0;

    if (p->st.have_estats) {
        if (p->src->stats(p->src->ctx, &es1) == 0) {
            p->st.e_hits = es1.hits - es0.hits;
            p->st.e_misses = es1.misses - es0.misses;
            p->st.e_bytes = es1.bytes_read - es0.bytes_read;
        }
    }
    if (p->tr) p->st.trunk_bytes = p->tr->bytes_read - trunk0;

    /* Fold the prompt's timers into the same report decode uses. t_io carries
     * both expert reads and the block's trunk reads; split them so the report
     * keeps its two different meanings. */
    dsv4_prof.attn       += p->t_attn;
    dsv4_prof.idxer      += p->t_idx;
    dsv4_prof.gate       += p->t_gate;
    dsv4_prof.expert_mm  += p->t_moe;
    dsv4_prof.shared     += p->t_shared;
    dsv4_prof.hc         += p->t_hc;
    dsv4_prof.trunk_io   += p->t_trunk;
    dsv4_prof.bind       += p->t_bind;
    dsv4_prof.expert_io  += p->t_io - p->t_trunk;

    fprintf(stderr,
            "prefill: %d tokens at pos %d, one block, %.2f s "
            "(%.3f s/token); attn %.2f moe %.2f io %.2f bind %.2f hc %.2f\n",
            n, pos0, p->t_total,
            p->t_total / (double)n, p->t_attn, p->t_moe, p->t_io,
            p->t_bind, p->t_hc);
    dsv4_prefill_report(p, n);
    return 0;
}

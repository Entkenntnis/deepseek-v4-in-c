/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill.c - one block through every layer, layer-major. */
#include <string.h>

#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"
#include "dsv4_util.h"

static int prefill_layer(DSV4Prefill *p, int L, const DSV4LayerW *w,
                         int n, int pos0, const int32_t *ids,
                         const float *cs, const float *sn)
{
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const int mix_hc = (2 + hc) * hc;

    /* Hash layers get their expert ids from the token id itself, so their
     * reads can start before any attention work. The batch stays live until
     * dsv4_prefill_moe() reaches the union; the readers fill slots while the
     * whole dense half runs. Scored layers start after the router gate. */
    if (w->moe.tid2eid) {
        const int topk = c->topk;
        for (int t = 0; t < n; t++) {
            const int64_t *row = w->moe.tid2eid + (int64_t)ids[t] * topk;
            for (int k = 0; k < topk; k++)
                p->topk_idx[(size_t)t * topk + k] = (int)row[k];
        }
        const int prc = dsv4_prefill_moe_prefetch(p, L, n);
        if (prc) return prc;
    }

    for (int t = 0; t < n; t++) {
        memcpy(p->resid + (size_t)t * hcd, p->H + (size_t)t * hcd,
               hcd * sizeof(float));
        dsv4_hc_pre(p->x1 + (size_t)t * d,
                    p->post + (size_t)t * hc,
                    p->comb + (size_t)t * hc * hc,
                    p->mixes + (size_t)t * mix_hc,
                    p->H + (size_t)t * hcd, &w->hc_attn, hc, d,
                    c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
        dsv4_rmsnorm(p->x1 + (size_t)t * d, p->x1 + (size_t)t * d,
                     w->attn_norm, d, c->rms_eps);
    }

    const int arc = dsv4_prefill_attention(p, L, w, n, pos0, cs, sn);
    if (arc) {
        dsv4_prefill_moe_abort(p);
        return arc;
    }

    for (int t = 0; t < n; t++) {
        memcpy(p->resid + (size_t)t * hcd, p->H + (size_t)t * hcd,
               hcd * sizeof(float));
        dsv4_hc_pre(p->x1 + (size_t)t * d,
                    p->post + (size_t)t * hc,
                    p->comb + (size_t)t * hc * hc,
                    p->mixes + (size_t)t * mix_hc,
                    p->H + (size_t)t * hcd, &w->hc_ffn, hc, d,
                    c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
        dsv4_rmsnorm(p->x1 + (size_t)t * d, p->x1 + (size_t)t * d,
                     w->ffn_norm, d, c->rms_eps);
    }

    const int mrc = dsv4_prefill_moe(p, L, w, n, ids);
    if (mrc) return mrc;
    return 0;
}

int dsv4_prefill_range(DSV4Prefill *p, const int32_t *ids, int n, int pos0,
                       const float *cs_d, const float *sn_d,
                       const float *cs_c, const float *sn_c,
                       float *h_out)
{
    if (!p || !ids || n <= 0 || !h_out || pos0 < 0) return -1;
    p->state_dirty = 0;
    /* A previous call must never leave a batch on the wire; drain it before
     * touching state again so its readers cannot race this call. */
    dsv4_prefill_moe_abort(p);
    if (dsv4_prefill_cancelled(p)) return DSV4_PREFILL_CANCELLED;
    if (dsv4_prefill_reserve(p, n) != 0) return -1;
    p->n = n;

    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const double t0 = dsv4_now_s();

    dsv4_prefill_progress(p, "start", 0, 0, n, 0.0, 0.0);

    /* ONE BLOCK, LAYER-MAJOR. The whole prompt is the block: every token
     * passes layer L before any token enters L+1, which keeps the layer's dense
     * weights and routed experts resident for the entire prompt. */
    for (int t = 0; t < n; t++) {
        if ((t & 63) == 0 && dsv4_prefill_cancelled(p))
            return dsv4_prefill_cancel_code(p);
        dsv4_embed_row(p->xh, p->mw->embed, p->mw->wdt, ids[t], d);
        for (int j = 0; j < hc; j++)
            memcpy(p->H + (size_t)t * hcd + (size_t)j * d, p->xh,
                   (size_t)d * sizeof(float));
    }

    for (int L = 0; L < c->n_layers; L++) {
        if (dsv4_prefill_cancelled(p))
            return dsv4_prefill_cancel_code(p);
        DSV4LayerW lw;
        if (dsv4_trunk_bind(p->tr, c, L, &lw) != 0) return -1;

        const int comp = dsv4_compress_ratio(c, L) != 0;
        const float *cs = comp ? cs_c : cs_d;
        const float *sn = comp ? sn_c : sn_d;

        /* `layer` is 1-based and names the layer being entered; progress is
         * the completed fraction, so the bar moves when a layer finishes. */
        const double frac = (double)L / (double)c->n_layers;
        dsv4_prefill_progress(p, "layer", L + 1, (int)(frac * (double)n), n,
                              dsv4_now_s() - t0, frac);
        /* From here on layer state can change; a stop now needs a reset. */
        p->state_dirty = 1;
        const int lrc = prefill_layer(p, L, &lw, n, pos0, ids, cs, sn);
        if (lrc) {
            dsv4_prefill_moe_abort(p);
            return lrc;
        }
    }

    dsv4_prefill_progress(p, "done", c->n_layers, n, n,
                          dsv4_now_s() - t0, 1.0);
    memcpy(h_out, p->H + (size_t)(n - 1) * hcd, hcd * sizeof(float));
    return 0;
}

int dsv4_prefill_pm_multi(DSV4Prefill *p, const DSV4PMJob *jobs, int nj)
{
    for (int j = 0; j < nj; j++) {
        if (jobs[j].n < 1) continue;
        if (p->vk && dsv4_prefill_vk_gemm(p->vk, jobs[j].y, jobs[j].x,
                                          jobs[j].m, jobs[j].n) == 0)
            continue;
        dsv4_mmq_n(jobs[j].y, jobs[j].x, jobs[j].m, jobs[j].n);
    }
    return 0;
}

int dsv4_prefill_pm(DSV4Prefill *p, float *y, const float *x,
                    const DSV4QMat *m, int n)
{
    DSV4PMJob job = { m, x, y, n };
    return dsv4_prefill_pm_multi(p, &job, 1);
}

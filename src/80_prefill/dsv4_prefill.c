/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill.c - one block through every layer, layer-major. */
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

    const int arc = dsv4_prefill_attention(p, L, w, n, pos0, cs, sn);
    if (arc) {
        dsv4_prefill_moe_abort(p);
        return arc;
    }

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
    if (p->instr) {
        p->instr_call++;
        memset(p->linstr, 0, sizeof p->linstr);
        memset(p->vk_stats, 0, sizeof p->vk_stats);
    }
    DSV4PrefillVkStats vk0[DSV4_VK_KIND_COUNT];
    if (p->instr) dsv4_prefill_vk_stats(p->vk, vk0);
    const DSV4Cfg *c = p->c;
    const int hc = c->hc_mult, d = c->hidden;
    const size_t hcd = (size_t)hc * d;
    const double t0 = dsv4_now_s();

    /* A resident engine calls this once per turn. Reset so this turn's profile
     * is not the previous turn's plus this one. */
    p->t_total = p->t_attn = p->t_idx = p->t_gate = p->t_shared = 0.0;
    p->t_moe = p->t_moe_wall = p->t_io = p->t_trunk = 0.0;
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
        if ((t & 63) == 0 && dsv4_prefill_cancelled(p))
            return dsv4_prefill_cancel_code(p);
        dsv4_embed_row(p->xh, p->mw->embed, p->mw->wdt, ids[t], d);
        for (int j = 0; j < hc; j++)
            memcpy(p->H + (size_t)t * hcd + (size_t)j * d, p->xh,
                   (size_t)d * sizeof(float));
    }
    p->st.embed += dsv4_now_s() - te;

    for (int L = 0; L < c->n_layers; L++) {
        if (dsv4_prefill_cancelled(p))
            return dsv4_prefill_cancel_code(p);
        const double tl0 = dsv4_now_s();
        const double ta0 = p->t_attn;
        const double tmw0 = p->t_moe_wall;
        const double tg0 = p->t_gate, ts0 = p->t_shared;
        const double th0 = p->t_hc, tio0 = p->t_io;
        DSV4PrefillVkStats lvk0[DSV4_VK_KIND_COUNT];
        if (p->instr >= 2) dsv4_prefill_vk_stats(p->vk, lvk0);
        p->cur_layer = L;
        DSV4LayerW lw;
        const double tb = dsv4_now_s();
        const double io0 = p->tr->load_seconds;
        if (dsv4_trunk_bind(p->tr, c, L, &lw) != 0) return -1;
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
        /* From here on layer state can change; a stop now needs a reset. */
        p->state_dirty = 1;
        const int lrc = prefill_layer(p, L, &lw, n, pos0, ids, cs, sn);
        if (lrc) {
            dsv4_prefill_moe_abort(p);
            return lrc;
        }
        if (p->instr >= 2) {
            DSV4PrefillLayerInstr *li = &p->linstr[L];
            li->total_s = dsv4_now_s() - tl0;
            li->attn_s = p->t_attn - ta0;
            li->moe_s = p->t_moe_wall - tmw0;
            li->gate_s = p->t_gate - tg0;
            li->shared_s = p->t_shared - ts0;
            li->hc_s = p->t_hc - th0;
            li->io_s = p->t_io - tio0;
            DSV4PrefillVkStats lvk1[DSV4_VK_KIND_COUNT];
            dsv4_prefill_vk_stats(p->vk, lvk1);
            for (int k = 0; k < DSV4_VK_KIND_COUNT; k++) {
                const DSV4PrefillVkStats *a = &lvk0[k], *b = &lvk1[k];
                DSV4PrefillVkStats *vd = &li->vk[k];
                vd->calls = b->calls - a->calls;
                vd->tiles = b->tiles - a->tiles;
                vd->xbytes = b->xbytes - a->xbytes;
                vd->ybytes = b->ybytes - a->ybytes;
                vd->desc_s = b->desc_s - a->desc_s;
                vd->copy_in_s = b->copy_in_s - a->copy_in_s;
                vd->record_s = b->record_s - a->record_s;
                vd->submit_s = b->submit_s - a->submit_s;
                vd->wait_s = b->wait_s - a->wait_s;
                vd->fill_s = b->fill_s - a->fill_s;
                vd->copy_out_s = b->copy_out_s - a->copy_out_s;
                vd->total_s = b->total_s - a->total_s;
            }
            dsv4_prefill_instr_layer(p, L);
        }
    }

    dsv4_prefill_progress(p, "done", c->n_layers, n, n, dsv4_now_s() - t0, 1.0);
    memcpy(h_out, p->H + (size_t)(n - 1) * hcd, hcd * sizeof(float));
    p->t_total = dsv4_now_s() - t0;

    if (p->st.have_estats) {
        if (p->src->stats(p->src->ctx, &es1) == 0) {
            p->st.e_hits = es1.hits - es0.hits;
            p->st.e_misses = es1.misses - es0.misses;
            p->st.e_bytes = es1.bytes_read - es0.bytes_read;
            p->st.io_read_n = es1.read_n - es0.read_n;
            p->st.io_read_s = es1.read_s - es0.read_s;
            p->st.io_read_max_s = es1.read_max_s;
            for (int i = 0; i < DSV4_PF_HIST_BINS; i++) {
                p->st.read_lat_n[i] = es1.lat_n[i] - es0.lat_n[i];
                p->st.read_lat_s[i] = es1.lat_s[i] - es0.lat_s[i];
            }
            p->st.cache_slots = es1.slots;
            p->st.cache_readers = es1.readers;
        }
    }
    if (p->tr) p->st.trunk_bytes = p->tr->bytes_read - trunk0;

    /* Fold the prompt's timers into the same report decode uses. t_io carries
     * both expert reads and the block's trunk reads; split them so the report
     * keeps its two different meanings. */
    dsv4_prof.attn       += p->t_attn;
    dsv4_prof.idxer      += p->t_idx;
    dsv4_prof.gate       += p->t_gate;
    dsv4_prof.expert_mm  += p->t_moe_wall;
    dsv4_prof.shared     += p->t_shared;
    dsv4_prof.hc         += p->t_hc;
    dsv4_prof.trunk_io   += p->t_trunk;
    dsv4_prof.bind       += p->t_bind;
    dsv4_prof.expert_io  += p->t_io - p->t_trunk;

    fprintf(stderr,
            "prefill: %d tokens at pos %d, one block, %.2f s "
            "(%.3f s/token); attn %.2f moe %.2f io %.2f bind %.2f hc %.2f\n",
            n, pos0, p->t_total,
            p->t_total / (double)n, p->t_attn, p->t_moe_wall, p->t_io,
            p->t_bind, p->t_hc);
    dsv4_prefill_report(p, n);

    if (p->instr) {
        DSV4PrefillVkStats vk1[DSV4_VK_KIND_COUNT];
        dsv4_prefill_vk_stats(p->vk, vk1);
        for (int k = 0; k < DSV4_VK_KIND_COUNT; k++) {
            const DSV4PrefillVkStats *a = &vk0[k], *b = &vk1[k];
            DSV4PrefillVkStats *vd = &p->vk_stats[k];
            vd->calls = b->calls - a->calls;
            vd->tiles = b->tiles - a->tiles;
            vd->xbytes = b->xbytes - a->xbytes;
            vd->ybytes = b->ybytes - a->ybytes;
            vd->desc_s = b->desc_s - a->desc_s;
            vd->copy_in_s = b->copy_in_s - a->copy_in_s;
            vd->record_s = b->record_s - a->record_s;
            vd->submit_s = b->submit_s - a->submit_s;
            vd->wait_s = b->wait_s - a->wait_s;
            vd->fill_s = b->fill_s - a->fill_s;
            vd->copy_out_s = b->copy_out_s - a->copy_out_s;
            vd->total_s = b->total_s - a->total_s;
        }
        dsv4_prefill_instr_summary(p, n, pos0);
    }
    return 0;
}

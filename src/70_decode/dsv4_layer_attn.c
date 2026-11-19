/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer_attn.c - attention for one token: CSA, HCA, grouped output proj. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_layer.h"
#include "dsv4_layer_int.h"

void dsv4_layer_attention(float *out, const float *x, const DSV4LayerW *w,
                          const DSV4Cfg *c, DSV4Scratch *s, DSV4LayerState *st,
                          const float *cs, const float *sn, int pos, int n_idx)
{
    const int hd = c->head_dim, rd = c->qk_rope, H = c->n_heads;
    float *kv_cache = st->kv_cache;

    /* q: learned norm on the latent, then the per-head unweighted norm. */
    dsv4_mmq(s->qr, x, &w->attn.wq_a);
    dsv4_rmsnorm(s->qr, s->qr, w->attn.q_norm, c->q_lora, c->rms_eps);
    dsv4_mmq(s->q, s->qr, &w->attn.wq_b);
    for (int h = 0; h < H; h++) {
        float *qh = s->q + (size_t)h * hd;
        double sq = 0.0;
        for (int i = 0; i < hd; i++) sq += (double)qh[i] * qh[i];
        const float inv =
            (float)(1.0 / sqrt(sq / (double)hd + (double)c->rms_eps));
        for (int i = 0; i < hd; i++) qh[i] *= inv;
        dsv4_rope_apply(qh, hd, rd, cs, sn, pos, rd / 2, 0);
    }

    /* kv: one head, written into the sliding ring. */
    dsv4_mmq(s->kv, x, &w->attn.wkv);
    dsv4_rmsnorm(s->kv, s->kv, w->attn.kv_norm, hd, c->rms_eps);
    dsv4_rope_apply(s->kv, hd, rd, cs, sn, pos, rd / 2, 0);
    memcpy(kv_cache + (size_t)(pos % c->sliding_window) * hd, s->kv,
           (size_t)hd * sizeof(float));

    /* HCA: pool `ratio` tokens into one compressed row, past the window. */
    if (w->has_comp) {
        const int ratio = w->compress_ratio;

        dsv4_mmq(s->comp_kv_in, x, &w->attn.comp.wkv);
        dsv4_mmq(s->comp_sc_in, x, &w->attn.comp.wgate);
        if (dsv4_compress_step(s->kv, s->comp_kv_in, s->comp_sc_in,
                               w->attn.comp.ape, st->comp_kv, st->comp_score,
                               pos, ratio, hd)) {
            dsv4_rmsnorm(s->kv, s->kv, w->attn.comp.norm, hd, c->rms_eps);
            /* Stamped with the FIRST position of the window it summarises. */
            dsv4_rope_apply(s->kv, hd, rd, cs, sn,
                            dsv4_compress_rope_pos(pos, ratio), rd / 2, 0);
            memcpy(kv_cache + (size_t)(c->sliding_window + st->n_compressed) * hd,
                   s->kv, (size_t)hd * sizeof(float));
            st->n_compressed++;
        }

        /* Which compressed rows may attention see? ratio 4 selects index_topk
         * with the CSA indexer; every other ratio takes them all, in order.
         * Both then offset by sliding_window, where the window sits. */
        const int avail = st->n_compressed;
        if (avail > s->idx_cap) {
            /* Impossible once scratch and state use the same max_pos, but a
             * wrong size must stop here rather than scribble past the arena. */
            fprintf(stderr, "dsv4_layer: %d compressed rows but idx_scores "
                            "holds %d; scratch and state disagree on max_pos\n",
                    avail, s->idx_cap);
            abort();
        }
        int take = avail;
        if (avail > 0) {
            if (w->has_idx) {
                dsv4_mmq(s->idx_q, s->qr, &w->attn.idx.wq_b);
                for (int h = 0; h < c->index_n_heads; h++)
                    dsv4_rope_apply(s->idx_q + (size_t)h * c->index_head_dim,
                                    c->index_head_dim, rd, cs, sn, pos, rd / 2, 0);
                dsv4_mmq(s->idx_w, x, &w->attn.idx.weights_proj);
                dsv4_indexer_score(s->idx_scores, s->idx_q,
                                   kv_cache + (size_t)c->sliding_window * hd,
                                   s->idx_w, c->index_n_heads,
                                   c->index_head_dim, avail);
                take = avail < c->index_topk ? avail : c->index_topk;
                dsv4_topk(s->idxs + n_idx, s->idx_scores, avail, take);
            } else {
                for (int i = 0; i < take; i++) s->idxs[n_idx + i] = i;
            }
            dsv4_indexer_offset(s->idxs + n_idx, take, c->sliding_window);
            n_idx += take;
        }
    }

    dsv4_sparse_attn(s->o, s->q, kv_cache, w->attn.sink, s->idxs,
                     H, hd, n_idx, 1.0f / sqrtf((float)hd), s->attn_scratch);

    /* De-rotate, then the grouped low-rank output projection. */
    for (int h = 0; h < H; h++)
        dsv4_rope_apply(s->o + (size_t)h * hd, hd, rd, cs, sn, pos, rd / 2, 1);

    const int gw = H * hd / c->o_groups;       /* channels per group */
    const int gr = c->o_lora;                  /* rank per group     */
    /* Stored bytes per element: 2 for BF16, 1 for FP8/packed FP4. Advancing by
     * the logical width instead read halfway into the wrong rows. */
    const size_t esz = (w->attn.wo_a.wdt == DSV4_WBF16) ? 2u : 1u;
    const size_t row_bytes = (w->attn.wo_a.wdt == DSV4_WFP4)
                           ? (size_t)gw / 2u : (size_t)gw * esz;
    for (int g = 0; g < c->o_groups; g++) {
        DSV4QMat slice = w->attn.wo_a;
        slice.rows = gr;
        slice.cols = gw;
        slice.w = (const unsigned char *)w->attn.wo_a.w
                + (size_t)g * gr * row_bytes;
        if (slice.s) {
            const size_t sc_row = (size_t)((gw + slice.blk_c - 1) / slice.blk_c);
            slice.s = (const unsigned char *)w->attn.wo_a.s
                    + (size_t)g * (size_t)(gr / (slice.blk_r ? slice.blk_r : 1))
                    * sc_row;
        }
        dsv4_mmq(s->ogrp + (size_t)g * gr, s->o + (size_t)g * gw, &slice);
    }
    dsv4_mmq(out, s->ogrp, &w->attn.wo_b);
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_attn.c - the attention half for a block of tokens. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

/* The projections run as GEMMs; q/kv norms, RoPE, compressor, indexer and
 * sparse attention stay per token because they carry a sequential position or
 * a tiny working set. */
int dsv4_prefill_attention(DSV4Prefill *p, int L, const DSV4LayerW *w,
                           int n, int pos0, const float *cs, const float *sn)
{
    const DSV4Cfg *c = p->c;
    const int d = c->hidden, hd = c->head_dim;
    const int H = c->n_heads, rd = c->qk_rope, ql = c->q_lora;
    const int comp = w->has_comp;
    DSV4LayerState *st = &p->lst[L];
    float *kvc = st->kv_cache;

    if (dsv4_prefill_pm(p, p->qr, p->x1, &w->attn.wq_a, n)) return -1;
    if (dsv4_prefill_pm(p, p->kv, p->x1, &w->attn.wkv, n)) return -1;
    for (int t = 0; t < n; t++)
        dsv4_rmsnorm(p->qr + (size_t)t * ql, p->qr + (size_t)t * ql,
                     w->attn.q_norm, ql, c->rms_eps);
    if (dsv4_prefill_pm(p, p->q, p->qr, &w->attn.wq_b, n)) return -1;

    /* COMPRESSOR PROJECTIONS ARE GEMMs, NOT GEMVs. They read x1 and carry no
     * position dependency, so the block's weights are read once and used for
     * every token instead of being streamed again per token. compress_step below
     * still runs per token and reads its row. */
    if (comp) {
        if (dsv4_prefill_pm(p, p->comp_kv_b, p->x1, &w->attn.comp.wkv, n))
            return -1;
        if (dsv4_prefill_pm(p, p->comp_sc_b, p->x1, &w->attn.comp.wgate, n))
            return -1;
    }

    /* The indexer projections are position independent too, but they are only
     * needed once the layer has compressed rows. avail grows with t, so the
     * first token that needs them batches the whole remaining suffix; earlier
     * blocks of a fresh prompt would otherwise pay for rows nobody reads. */
    int idx_batched = 0;

    for (int t = 0; t < n; t++) {
        if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);
        const int pos = pos0 + t;
        float *qt = p->q + (size_t)t * (size_t)H * hd;
        for (int h = 0; h < H; h++) {
            float *qh = qt + (size_t)h * hd;
            double sq = 0.0;
            for (int i = 0; i < hd; i++) sq += (double)qh[i] * qh[i];
            const float inv = (float)(1.0 / sqrt(sq / (double)hd +
                                                 (double)c->rms_eps));
            for (int i = 0; i < hd; i++) qh[i] *= inv;
            dsv4_rope_apply(qh, hd, rd, cs, sn, pos, rd / 2, 0);
        }

        float *kvt = p->kv + (size_t)t * hd;
        dsv4_rmsnorm(kvt, kvt, w->attn.kv_norm, hd, c->rms_eps);
        dsv4_rope_apply(kvt, hd, rd, cs, sn, pos, rd / 2, 0);
        memcpy(kvc + (size_t)(pos % c->sliding_window) * hd, kvt,
               (size_t)hd * sizeof(float));

        if (comp) {
            const int ratio = w->compress_ratio;
            const float *ckv = p->comp_kv_b + (size_t)t * 2 * hd;
            const float *csc = p->comp_sc_b + (size_t)t * 2 * hd;
            if (dsv4_compress_step(p->attn_kv,
                                   ckv, csc,
                                   w->attn.comp.ape, st->comp_kv,
                                   st->comp_score, pos, ratio, hd)) {
                dsv4_rmsnorm(p->attn_kv, p->attn_kv,
                             w->attn.comp.norm, hd, c->rms_eps);
                dsv4_rope_apply(p->attn_kv, hd, rd, cs, sn,
                                dsv4_compress_rope_pos(pos, ratio), rd / 2, 0);
                memcpy(kvc + (size_t)(c->sliding_window + st->n_compressed) * hd,
                       p->attn_kv, (size_t)hd * sizeof(float));
                st->n_compressed++;
            }
        }

        int n_idx = dsv4_window_idxs(p->attn_idxs, pos,
                                     c->sliding_window);
        const int avail = st->n_compressed;
        if (avail > 0) {
            if (avail > p->attn_idx_cap) {
                fprintf(stderr, "prefill: %d compressed rows but idx_scores "
                                "holds %d\n", avail, p->attn_idx_cap);
                return -1;
            }
            int take = avail;
            if (w->has_idx) {
                if (!idx_batched) {
                    const int rem = n - t;
                    if (dsv4_prefill_pm(
                            p, p->idx_qb + (size_t)t * c->index_n_heads *
                                           c->index_head_dim,
                            p->qr + (size_t)t * ql, &w->attn.idx.wq_b, rem))
                        return -1;
                    if (dsv4_prefill_pm(
                            p, p->idx_wb + (size_t)t * c->index_n_heads,
                            p->x1 + (size_t)t * d,
                            &w->attn.idx.weights_proj, rem))
                        return -1;
                    idx_batched = 1;
                }
                float *iq = p->idx_qb + (size_t)t * c->index_n_heads *
                                        c->index_head_dim;
                const float *iw = p->idx_wb + (size_t)t * c->index_n_heads;
                for (int h = 0; h < c->index_n_heads; h++)
                    dsv4_rope_apply(iq + (size_t)h * c->index_head_dim,
                                    c->index_head_dim, rd, cs, sn, pos,
                                    rd / 2, 0);
                dsv4_indexer_score(p->attn_idx_scores,
                                   iq,
                                   kvc + (size_t)c->sliding_window * hd,
                                   iw, c->index_n_heads,
                                   c->index_head_dim, avail);
                take = avail < c->index_topk ? avail : c->index_topk;
                dsv4_topk(p->attn_idxs + n_idx,
                          p->attn_idx_scores, avail, take);
            } else {
                for (int i = 0; i < take; i++)
                    p->attn_idxs[n_idx + i] = i;
            }
            dsv4_indexer_offset(p->attn_idxs + n_idx, take,
                                c->sliding_window);
            n_idx += take;
        }

        float *ot = p->o + (size_t)t * (size_t)H * hd;
        dsv4_sparse_attn(ot, qt, kvc, w->attn.sink, p->attn_idxs,
                         H, hd, n_idx, 1.0f / sqrtf((float)hd),
                         p->attn_scratch);

        for (int h = 0; h < H; h++)
            dsv4_rope_apply(ot + (size_t)h * hd, hd, rd, cs, sn, pos,
                            rd / 2, 1);
    }

    if (dsv4_prefill_cancelled(p)) return dsv4_prefill_cancel_code(p);
    if (dsv4_prefill_oproj(p, w, n)) return -1;
    return 0;
}

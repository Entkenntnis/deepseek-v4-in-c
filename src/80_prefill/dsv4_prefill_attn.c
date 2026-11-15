/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_attn.c - the attention half for a block of tokens. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_prof.h"
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
    const double t0 = dsv4_now_s();

    if (dsv4_prefill_pm_stats(p, p->qr, p->x1, &w->attn.wq_a, n,
                              &p->st.proj_qkv, &p->st.f_qkv)) return -1;
    if (dsv4_prefill_pm_stats(p, p->kv, p->x1, &w->attn.wkv, n,
                              &p->st.proj_qkv, &p->st.f_qkv)) return -1;
    {
        const double ts = dsv4_now_s();
        for (int t = 0; t < n; t++)
            dsv4_rmsnorm(p->qr + (size_t)t * ql, p->qr + (size_t)t * ql,
                         w->attn.q_norm, ql, c->rms_eps);
        p->st.q_norm += dsv4_now_s() - ts;
    }
    if (dsv4_prefill_pm_stats(p, p->q, p->qr, &w->attn.wq_b, n,
                              &p->st.proj_qkv, &p->st.f_qkv)) return -1;

    for (int t = 0; t < n; t++) {
        const int pos = pos0 + t;
        float *qt = p->q + (size_t)t * (size_t)H * hd;
        double ts = dsv4_now_s();
        for (int h = 0; h < H; h++) {
            float *qh = qt + (size_t)h * hd;
            double sq = 0.0;
            for (int i = 0; i < hd; i++) sq += (double)qh[i] * qh[i];
            const float inv = (float)(1.0 / sqrt(sq / (double)hd +
                                                 (double)c->rms_eps));
            for (int i = 0; i < hd; i++) qh[i] *= inv;
            dsv4_rope_apply(qh, hd, rd, cs, sn, pos, rd / 2, 0);
        }
        p->st.q_head += dsv4_now_s() - ts;

        float *kvt = p->kv + (size_t)t * hd;
        ts = dsv4_now_s();
        dsv4_rmsnorm(kvt, kvt, w->attn.kv_norm, hd, c->rms_eps);
        dsv4_rope_apply(kvt, hd, rd, cs, sn, pos, rd / 2, 0);
        memcpy(kvc + (size_t)(pos % c->sliding_window) * hd, kvt,
               (size_t)hd * sizeof(float));
        p->st.kv_norm += dsv4_now_s() - ts;

        if (comp) {
            const int ratio = w->compress_ratio;
            ts = dsv4_now_s();
            dsv4_mmq(p->scratch[t].comp_kv_in, p->x1 + (size_t)t * d,
                     &w->attn.comp.wkv);
            dsv4_mmq(p->scratch[t].comp_sc_in, p->x1 + (size_t)t * d,
                     &w->attn.comp.wgate);
            p->st.comp_proj += dsv4_now_s() - ts;
            p->st.f_comp += 2.0 * ((double)w->attn.comp.wkv.rows *
                                   (double)w->attn.comp.wkv.cols +
                                   (double)w->attn.comp.wgate.rows *
                                   (double)w->attn.comp.wgate.cols) * 1e-9;
            ts = dsv4_now_s();
            if (dsv4_compress_step(p->scratch[t].kv,
                                   p->scratch[t].comp_kv_in,
                                   p->scratch[t].comp_sc_in,
                                   w->attn.comp.ape, st->comp_kv,
                                   st->comp_score, pos, ratio, hd)) {
                dsv4_rmsnorm(p->scratch[t].kv, p->scratch[t].kv,
                             w->attn.comp.norm, hd, c->rms_eps);
                dsv4_rope_apply(p->scratch[t].kv, hd, rd, cs, sn,
                                dsv4_compress_rope_pos(pos, ratio), rd / 2, 0);
                memcpy(kvc + (size_t)(c->sliding_window + st->n_compressed) * hd,
                       p->scratch[t].kv, (size_t)hd * sizeof(float));
                st->n_compressed++;
            }
            p->st.comp_step += dsv4_now_s() - ts;
        }

        ts = dsv4_now_s();
        int n_idx = dsv4_window_idxs(p->scratch[t].idxs, pos,
                                     c->sliding_window);
        p->st.win_idx += dsv4_now_s() - ts;
        const int avail = st->n_compressed;
        if (avail > 0) {
            if (avail > p->scratch[t].idx_cap) {
                fprintf(stderr, "prefill: %d compressed rows but idx_scores "
                                "holds %d\n", avail, p->scratch[t].idx_cap);
                return -1;
            }
            int take = avail;
            if (w->has_idx) {
                const double ti = dsv4_now_s();
                ts = dsv4_now_s();
                dsv4_mmq(p->scratch[t].idx_q, p->qr + (size_t)t * ql,
                         &w->attn.idx.wq_b);
                p->st.idx_proj += dsv4_now_s() - ts;
                ts = dsv4_now_s();
                for (int h = 0; h < c->index_n_heads; h++)
                    dsv4_rope_apply(p->scratch[t].idx_q +
                                        (size_t)h * c->index_head_dim,
                                    c->index_head_dim, rd, cs, sn, pos,
                                    rd / 2, 0);
                p->st.idx_rope += dsv4_now_s() - ts;
                ts = dsv4_now_s();
                dsv4_mmq(p->scratch[t].idx_w, p->x1 + (size_t)t * d,
                         &w->attn.idx.weights_proj);
                p->st.idx_proj += dsv4_now_s() - ts;
                p->st.f_idx += 2.0 * ((double)w->attn.idx.wq_b.rows *
                                      (double)w->attn.idx.wq_b.cols +
                                      (double)w->attn.idx.weights_proj.rows *
                                      (double)w->attn.idx.weights_proj.cols)
                             * 1e-9;
                ts = dsv4_now_s();
                dsv4_indexer_score(p->scratch[t].idx_scores,
                                   p->scratch[t].idx_q,
                                   kvc + (size_t)c->sliding_window * hd,
                                   p->scratch[t].idx_w, c->index_n_heads,
                                   c->index_head_dim, avail);
                take = avail < c->index_topk ? avail : c->index_topk;
                dsv4_topk(p->scratch[t].idxs + n_idx,
                          p->scratch[t].idx_scores, avail, take);
                p->st.idx_score += dsv4_now_s() - ts;
                p->t_idx += dsv4_now_s() - ti;
            } else {
                ts = dsv4_now_s();
                for (int i = 0; i < take; i++)
                    p->scratch[t].idxs[n_idx + i] = i;
                p->st.idx_score += dsv4_now_s() - ts;
            }
            dsv4_indexer_offset(p->scratch[t].idxs + n_idx, take,
                                c->sliding_window);
            n_idx += take;
        }

        float *ot = p->o + (size_t)t * (size_t)H * hd;
        ts = dsv4_now_s();
        dsv4_sparse_attn(ot, qt, kvc, w->attn.sink, p->scratch[t].idxs,
                         H, hd, n_idx, 1.0f / sqrtf((float)hd),
                         p->scratch[t].attn_scratch);
        p->st.attn_core += dsv4_now_s() - ts;
        p->st.f_attn += 4.0 * (double)H * (double)hd * (double)n_idx * 1e-9;

        ts = dsv4_now_s();
        for (int h = 0; h < H; h++)
            dsv4_rope_apply(ot + (size_t)h * hd, hd, rd, cs, sn, pos,
                            rd / 2, 1);
        p->st.attn_deroep += dsv4_now_s() - ts;
    }

    if (dsv4_prefill_oproj(p, w, n)) return -1;

    p->t_attn += dsv4_now_s() - t0;
    return 0;
}

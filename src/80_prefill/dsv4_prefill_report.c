/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_report.c - progress callbacks and the stage report. */
#include <stdio.h>

#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

void dsv4_prefill_progress(DSV4Prefill *p, const char *stage, int layer,
                           int tokens_done, int tokens_total,
                           double elapsed, double progress)
{
    if (!p->opt.on_progress) return;
    DSV4PrefillProgress pr;
    pr.stage = stage;
    pr.layer = layer;
    pr.layers = p->c->n_layers;
    pr.tokens_done = tokens_done;
    pr.tokens_total = tokens_total;
    pr.elapsed = elapsed;
    pr.progress = progress < 0.0 ? 0.0 : (progress > 1.0 ? 1.0 : progress);
    p->opt.on_progress(p->opt.progress_ctx, &pr);
}

static void pf_row(const char *name, double sec, double total, int n,
                   double flop)
{
    if (sec <= 0.0 && flop <= 0.0) return;
    const double pct = total > 0.0 ? 100.0 * sec / total : 0.0;
    const double mst = n > 0 ? 1e3 * sec / (double)n : 0.0;
    if (flop > 0.0) {
        const double gfs = sec > 0.0 ? flop / sec : 0.0;
        fprintf(stderr, "    %-26s %8.3f s %6.1f%% %9.2f ms %9.2f GF %7.1f GF/s\n",
                name, sec, pct, mst, flop, gfs);
    } else {
        fprintf(stderr, "    %-26s %8.3f s %6.1f%% %9.2f ms\n",
                name, sec, pct, mst);
    }
}

void dsv4_prefill_report(DSV4Prefill *p, int n)
{
    const DSV4PrefillStats *s = &p->st;
    const double T = p->t_total;
    const double dense_sec = s->proj_qkv + s->oproj + s->gate_mm + s->shared_mm;
    const double dense_gf  = s->f_qkv + s->f_oproj + s->f_gate + s->f_shared;

    fprintf(stderr,
            "  prefill stages: %d tokens in one block, %.3f s, "
            "%.3f s/token\n", n, T, n > 0 ? T / (double)n : 0.0);
    fprintf(stderr, "    %-26s %8s %7s %10s %10s %8s\n",
            "stage", "sec", "%", "ms/token", "GFLOP", "GF/s");

    fprintf(stderr, "  [embed / mHC / norms]\n");
    pf_row("embed", s->embed, T, n, 0.0);
    pf_row("hc_pre (attn+ffn)", s->hc_pre, T, n, 0.0);
    pf_row("rmsnorm (attn+ffn)", s->norm, T, n, 0.0);
    pf_row("hc_post (attention)", s->hc_post, T, n, 0.0);
    pf_row("hc_post (ffn)", s->hc_post_moe, T, n, 0.0);

    fprintf(stderr, "  [attention]\n");
    pf_row("qkv projections", s->proj_qkv, T, n, s->f_qkv);
    pf_row("q norm (q_lora)", s->q_norm, T, n, 0.0);
    pf_row("q head norm + rope", s->q_head, T, n, 0.0);
    pf_row("kv norm + rope + store", s->kv_norm, T, n, 0.0);
    pf_row("compressor projections", s->comp_proj, T, n, s->f_comp);
    pf_row("compress_step", s->comp_step, T, n, 0.0);
    pf_row("window index build", s->win_idx, T, n, 0.0);
    pf_row("indexer projections", s->idx_proj, T, n, s->f_idx);
    pf_row("indexer rope", s->idx_rope, T, n, 0.0);
    pf_row("indexer score + top-k", s->idx_score, T, n, 0.0);
    pf_row("sparse attention", s->attn_core, T, n, s->f_attn);
    pf_row("attention output de-rope", s->attn_deroep, T, n, 0.0);
    pf_row("o-proj pack", s->oproj_pack, T, n, 0.0);
    pf_row("o-proj group GEMMs", s->oproj, T, n, s->f_oproj);
    pf_row("o-proj unpack", s->oproj_unpack, T, n, 0.0);

    fprintf(stderr, "  [MoE]\n");
    pf_row("router matmul", s->gate_mm, T, n, s->f_gate);
    pf_row("router top-k", s->route, T, n, 0.0);
    pf_row("shared expert matmuls", s->shared_mm, T, n, s->f_shared);
    pf_row("shared expert swiglu", s->shared_swiglu, T, n, 0.0);
    pf_row("routed expert I/O", s->expert_io, T, n, 0.0);
    pf_row("routed expert matmuls", s->expert_mm, T, n, s->f_expert);
    pf_row("routed expert gather/scatter", s->expert_pack, T, n, 0.0);
    pf_row("routed expert swiglu", s->expert_swiglu, T, n, 0.0);
    pf_row("routed weighted sum", s->expert_sum, T, n, 0.0);

    fprintf(stderr, "  [trunk / bind]\n");
    pf_row("trunk read", s->trunk_io, T, n, 0.0);
    pf_row("trunk bind", s->bind, T, n, 0.0);

    fprintf(stderr, "  [kernel summary]\n");
    pf_row("dense projections", dense_sec, T, n, dense_gf);
    pf_row("routed experts (FP4)", s->expert_mm, T, n, s->f_expert);
    pf_row("sparse attention core", s->attn_core, T, n, s->f_attn);
    pf_row("compressor projections", s->comp_proj, T, n, s->f_comp);
    pf_row("indexer projections", s->idx_proj, T, n, s->f_idx);

    /* Top-level buckets. These do not overlap: t_hc is the two hc_pre/norm
     * passes, t_attn the whole attention half, t_gate/t_shared/t_moe the MoE
     * halves, and the two I/O buckets are the blocking calls. */
    const double covered = s->embed + p->t_hc + p->t_attn + p->t_gate +
                           p->t_shared + p->t_moe + s->expert_io +
                           s->trunk_io + s->bind + s->hc_post_moe;
    fprintf(stderr,
            "    top-level: hc %.3f  attn %.3f  router %.3f  shared %.3f  "
            "routed %.3f  exp-io %.3f  trunk %.3f  bind %.3f  other %.3f s\n",
            p->t_hc, p->t_attn, p->t_gate, p->t_shared, p->t_moe,
            s->expert_io, s->trunk_io, s->bind, T - covered);

    if (s->have_estats) {
        const uint64_t req = s->e_hits + s->e_misses;
        const double gb = (double)s->e_bytes / 1073741824.0;
        fprintf(stderr,
                "    expert cache: %llu requests, %llu hits (%.1f%%), "
                "%llu misses, %.2f GB read, %.2f GB/s over prefill%s\n",
                (unsigned long long)req, (unsigned long long)s->e_hits,
                req ? 100.0 * (double)s->e_hits / (double)req : 0.0,
                (unsigned long long)s->e_misses, gb,
                T > 0.0 ? gb / T : 0.0,
                (p->src->begin && p->src->end)
                    ? "  [union: duplicate requests are deduped before the cache]"
                    : "");
    }
    if (s->trunk_bytes) {
        const double gb = (double)s->trunk_bytes / 1073741824.0;
        fprintf(stderr, "    trunk data: %.2f GB read over prefill\n", gb);
    }
}

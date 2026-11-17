/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_report.c - progress callbacks, the stage report and the
 * opt-in deep instrumentation. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ---- deep instrumentation ------------------------------------------------ */

static void pf_instr_emit(DSV4Prefill *p, const char *fmt, ...)
{
    if (!p || p->instr <= 0) return;
    FILE *f = p->instr_fp ? p->instr_fp : stderr;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    fputc('\n', f);
    fflush(f);
    va_end(ap);
}

void dsv4_prefill_instr_init(DSV4Prefill *p)
{
    if (!p) return;
    const char *e = getenv("DSV4_PREFILL_INSTRUMENT");
    const char *layers = getenv("DSV4_PREFILL_INSTRUMENT_LAYERS");
    p->instr_file = getenv("DSV4_PREFILL_INSTRUMENT_FILE");

    if (e && *e && strcmp(e, "0") != 0) {
        const int v = atoi(e);
        p->instr = v >= 2 ? 2 : 1;
    }
    if (layers && *layers && strcmp(layers, "0") != 0) p->instr = 2;
    if (p->instr_file && *p->instr_file) {
        if (p->instr < 1) p->instr = 1;
        p->instr_fp = fopen(p->instr_file, "a");
        if (!p->instr_fp)
            fprintf(stderr, "prefill: cannot open instrumentation file %s\n",
                    p->instr_file);
    }
}

static void pf_instr_hist(DSV4Prefill *p, const char *tag, int L,
                          const int *hist, const int *tokens)
{
    char buf[512];
    int n = snprintf(buf, sizeof buf, "PF_%s call=%d L=%d", tag,
                     p->instr_call, L);
    for (int i = 0; i < DSV4_PF_HIST_BINS && n < (int)sizeof buf; i++)
        n += snprintf(buf + n, sizeof buf - (size_t)n, " h%d=%d", i, hist[i]);
    for (int i = 0; i < DSV4_PF_HIST_BINS && n < (int)sizeof buf; i++)
        n += snprintf(buf + n, sizeof buf - (size_t)n, " t%d=%d", i, tokens[i]);
    pf_instr_emit(p, "%s", buf);
}

void dsv4_prefill_instr_layer(DSV4Prefill *p, int L)
{
    if (!p || p->instr < 2) return;
    const DSV4PrefillLayerInstr *l = &p->linstr[L];
    pf_instr_emit(p,
        "PF_LAYER call=%d L=%d total_s=%.6f attn_s=%.6f moe_s=%.6f "
        "hc_s=%.6f io_s=%.6f gate_s=%.6f shared_s=%.6f experts=%d "
        "requests=%d cpu_buckets=%d gpu_buckets=%d fallbacks=%d "
        "cpu_s=%.6f gpu_s=%.6f union_s=%.6f lead_s=%.6f begin_s=%.6f "
        "wait_s=%.6f end_s=%.6f compute_s=%.6f bytes=%llu",
        p->instr_call, L, l->total_s, l->attn_s, l->moe_s, l->hc_s,
        l->io_s, l->gate_s, l->shared_s, l->experts, l->requests,
        l->buckets_cpu, l->buckets_gpu, l->gpu_fallbacks,
        l->cpu_mm, l->gpu_mm, l->union_s, l->lead_s, l->begin_s,
        l->wait_s, l->end_s, l->compute_s, (unsigned long long)l->bytes);
    pf_instr_hist(p, "LAYER_HIST", L, l->hist, l->hist_tokens);

    const DSV4PrefillVkStats *d = &l->vk[DSV4_VK_KIND_DENSE];
    const DSV4PrefillVkStats *e = &l->vk[DSV4_VK_KIND_EXPERT];
    pf_instr_emit(p,
        "PF_LAYER_VK call=%d L=%d "
        "dense_calls=%llu dense_tiles=%llu dense_desc_s=%.6f "
        "dense_copy_in_s=%.6f dense_record_s=%.6f dense_submit_s=%.6f "
        "dense_copy_out_s=%.6f expert_calls=%llu expert_tiles=%llu "
        "expert_desc_s=%.6f expert_copy_in_s=%.6f expert_record_s=%.6f "
        "expert_submit_s=%.6f expert_copy_out_s=%.6f",
        p->instr_call, L,
        (unsigned long long)d->calls, (unsigned long long)d->tiles,
        d->desc_s, d->copy_in_s, d->record_s, d->submit_s, d->copy_out_s,
        (unsigned long long)e->calls, (unsigned long long)e->tiles,
        e->desc_s, e->copy_in_s, e->record_s, e->submit_s, e->copy_out_s);
}

void dsv4_prefill_instr_summary(DSV4Prefill *p, int n, int pos0)
{
    if (!p || p->instr < 1) return;
    const DSV4PrefillStats *s = &p->st;
    const double routed_s = p->t_moe;
    const double expert_phase_s = routed_s + s->expert_io;
    const double read_avg_ms = s->io_read_n
        ? 1000.0 * s->io_read_s / (double)s->io_read_n : 0.0;
    const double read_busy = (s->cache_readers > 0 && p->t_total > 0.0)
        ? 100.0 * s->io_read_s / ((double)s->cache_readers * p->t_total)
        : 0.0;
    double union_s = 0.0;
    if (p->instr >= 2)
        for (int L = 0; L < p->c->n_layers; L++) union_s += p->linstr[L].union_s;
    const double read_busy_union = (s->cache_readers > 0 && union_s > 0.0)
        ? 100.0 * s->io_read_s / ((double)s->cache_readers * union_s)
        : 0.0;
    const double union_gbs = union_s > 0.0
        ? (double)s->e_bytes / 1e9 / union_s : 0.0;

    pf_instr_emit(p,
        "PF_CALL call=%d n=%d pos=%d total_s=%.6f attn_s=%.6f gate_s=%.6f "
        "shared_s=%.6f routed_s=%.6f io_block_s=%.6f expert_phase_s=%.6f "
        "hc_s=%.6f trunk_s=%.6f bind_s=%.6f union_s=%.6f",
        p->instr_call, n, pos0, p->t_total, p->t_attn, p->t_gate,
        p->t_shared, routed_s, s->expert_io, expert_phase_s, p->t_hc,
        s->trunk_io, s->bind, union_s);
    pf_instr_emit(p,
        "PF_EXPERT call=%d buckets=%d cpu_buckets=%d gpu_buckets=%d "
        "fallbacks=%d cpu_s=%.6f gpu_s=%.6f mm_s=%.6f swiglu_s=%.6f "
        "pack_s=%.6f sum_s=%.6f",
        p->instr_call, s->buckets_total, s->buckets_cpu, s->buckets_gpu,
        s->gpu_fallbacks, s->expert_mm_cpu, s->expert_mm_gpu,
        s->expert_mm, s->expert_swiglu, s->expert_pack, s->expert_sum);
    pf_instr_hist(p, "HIST", -1, s->bucket_hist, s->bucket_tokens);
    {
        char buf[512];
        int pos = snprintf(buf, sizeof buf, "PF_HIST_TIME call=%d",
                           p->instr_call);
        for (int i = 0; i < DSV4_PF_HIST_BINS && pos < (int)sizeof buf; i++)
            pos += snprintf(buf + pos, sizeof buf - (size_t)pos,
                            " s%d=%.6f", i, s->bucket_s[i]);
        for (int i = 0; i < DSV4_PF_HIST_BINS && pos < (int)sizeof buf; i++)
            pos += snprintf(buf + pos, sizeof buf - (size_t)pos,
                            " m%d=%.6f", i, s->bucket_mm_s[i]);
        pf_instr_emit(p, "%s", buf);
    }
    pf_instr_emit(p,
        "PF_IO call=%d groups=%d hits=%llu misses=%llu bytes=%llu reads=%llu "
        "read_sum_s=%.6f read_avg_ms=%.6f read_max_ms=%.6f busy_pct=%.2f "
        "busy_union_pct=%.2f union_gbs=%.3f slots=%d readers=%d "
        "begin_s=%.6f wait_s=%.6f end_s=%.6f",
        p->instr_call, s->io_groups,
        (unsigned long long)s->e_hits, (unsigned long long)s->e_misses,
        (unsigned long long)s->e_bytes, (unsigned long long)s->io_read_n,
        s->io_read_s, read_avg_ms, 1000.0 * s->io_read_max_s, read_busy,
        read_busy_union, union_gbs, s->cache_slots, s->cache_readers,
        s->io_begin_s, s->io_wait_s, s->io_end_s);
    {
        char buf[512];
        int pos = snprintf(buf, sizeof buf, "PF_LAT call=%d", p->instr_call);
        for (int i = 0; i < DSV4_PF_HIST_BINS && pos < (int)sizeof buf; i++)
            pos += snprintf(buf + pos, sizeof buf - (size_t)pos,
                            " n%d=%llu", i,
                            (unsigned long long)s->read_lat_n[i]);
        for (int i = 0; i < DSV4_PF_HIST_BINS && pos < (int)sizeof buf; i++)
            pos += snprintf(buf + pos, sizeof buf - (size_t)pos,
                            " s%d=%.6f", i, s->read_lat_s[i]);
        pf_instr_emit(p, "%s", buf);
    }

    for (int k = 0; k < DSV4_VK_KIND_COUNT; k++) {
        const DSV4PrefillVkStats *v = &p->vk_stats[k];
        pf_instr_emit(p,
            "PF_VK call=%d kind=%s calls=%llu tiles=%llu xbytes=%llu "
            "ybytes=%llu desc_s=%.6f copy_in_s=%.6f record_s=%.6f "
            "submit_s=%.6f copy_out_s=%.6f total_s=%.6f",
            p->instr_call, k == DSV4_VK_KIND_DENSE ? "dense" : "expert",
            (unsigned long long)v->calls, (unsigned long long)v->tiles,
            (unsigned long long)v->xbytes, (unsigned long long)v->ybytes,
            v->desc_s, v->copy_in_s, v->record_s, v->submit_s,
            v->copy_out_s, v->total_s);
    }
}


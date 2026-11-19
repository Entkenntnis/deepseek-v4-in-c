/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_emit.c - the line-delimited events.
 *
 * Shape: `<event> key=value key=value ...`. Only fields consumed by a default
 * TUI/e2e/prefill view are emitted. Strings are hex through
 * dsv4_emit_hex_field(); names and numbers are bare ASCII.
 */
#include <stdio.h>

#include "dsv4_serve_int.h"
#include "dsv4_util.h"

/* Static fields never change after init, so they are sent only on ready.
 * Dynamic fields are sampled at the TUI's cadence instead of on every event. */
#define DSV4_STATS_INTERVAL_S 0.5

static void write_static(const Engine *e)
{
    const DSV4Cache *c = &e->cache;
    const DSV4Trunk *t = &e->tr;
    printf(" cache_slots=%d cache_resident_gb=%.3f cache_slot_mb=%.2f"
           " trunk_layers=%d trunk_pinned=%d trunk_resident_gb=%.3f",
           c->nslot,
           (double)c->slot_bytes * (double)c->nslot / 1073741824.0,
           (double)c->slot_bytes / 1048576.0,
           t->n_layers, t->npin,
           (double)dsv4_trunk_resident_bytes(t) / 1073741824.0);
}

static void write_dynamic(const Engine *e)
{
    const DSV4Cache *c = &e->cache;
    const uint64_t n = c->hits + c->misses;
    printf(" sys_rss_mb=%.1f sys_io_read_gb=%.4f"
           " cache_hit=%llu cache_miss=%llu cache_hit_rate=%.4f"
           " cache_read_gb=%.4f",
           dsv4_proc_rss_mb(), dsv4_proc_read_gb(),
           (unsigned long long)c->hits, (unsigned long long)c->misses,
           n ? (double)c->hits / (double)n : 0.0,
           (double)(c->bytes_read + c->pf_bytes) / 1073741824.0);
}

static void write_stats(Engine *e, int force)
{
    const double now = dsv4_now_s();
    if (!force && e->last_stats_s > 0.0 &&
        now - e->last_stats_s < DSV4_STATS_INTERVAL_S)
        return;
    e->last_stats_s = now;
    write_dynamic(e);
}

void dsv4_emit_ready(Engine *e, const char *event)
{
    printf("%s config_hidden=%d config_layers=%d config_experts=%d "
           "config_topk=%d context_used=%d context_cap=%d",
           event, e->c.hidden, e->c.n_layers, e->c.n_experts, e->c.topk,
           e->n_ids, e->cap_ids);
    write_static(e);
    write_stats(e, 1);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_start(Engine *e, int start_pos, int prompt_tokens,
                     int gen_max, int prefill_tokens)
{
    printf("start pos=%d prompt_tokens=%d gen_max=%d prefill_tokens=%d "
           "context_cap=%d",
           start_pos, prompt_tokens, gen_max, prefill_tokens, e->cap_ids);
    write_stats(e, 1);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_layer(const Engine *e, const char *phase, int layer)
{
    (void)e;
    printf("layer layer=%d phase=%s\n", layer, phase);
    fflush(stdout);
}

void dsv4_emit_token(Engine *e, int32_t id, int index,
                     double t_turn, double ttft)
{
    printf("token id=%d index=%d t=%.4f ttft=%.4f "
           "prefill_s=%.4f prefill_tokens=%d",
           id, index, t_turn, ttft, e->prefill_s, e->prefill_tokens);
    write_stats(e, 0);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_reset(const Engine *e, const char *reason)
{
    (void)e;
    printf("reset reason=%s\n", reason);
    fflush(stdout);
}

void dsv4_emit_done(Engine *e, int prompt_tokens, int generated,
                    double seconds, double ttft, const char *stop)
{
    printf("done prompt_tokens=%d generated=%d seconds=%.4f "
           "s_per_token=%.4f ttft=%.4f stop=%s context_used=%d head_s=%.4f",
           prompt_tokens, generated, seconds,
           generated ? seconds / (double)generated : 0.0,
           ttft, stop, e->n_ids, e->head_s);
    write_stats(e, 1);
    putchar('\n');
    fflush(stdout);
}

/* The block is the whole prompt: every layer walks all of it before the next
 * layer starts, so tokens_done is an estimate while a layer is mid-flight. */
void dsv4_emit_prefill_progress(Engine *e, const DSV4PrefillProgress *p)
{
    printf("prefill stage=%s layer=%d layers=%d tokens_done=%d "
           "tokens_total=%d elapsed=%.4f progress=%.4f",
           p->stage, p->layer, p->layers, p->tokens_done, p->tokens_total,
           p->elapsed, p->progress);
    write_stats(e, 0);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_kv_built(int tokens, double seconds, long long bytes)
{
    printf("kv_built tokens=%d seconds=%.4f bytes=%lld\n",
           tokens, seconds, bytes);
    fflush(stdout);
}

void dsv4_emit_kv_loaded(int tokens, int context_cap, double seconds)
{
    printf("kv_loaded tokens=%d used=%d context_cap=%d seconds=%.4f\n",
           tokens, tokens, context_cap, seconds);
    fflush(stdout);
}


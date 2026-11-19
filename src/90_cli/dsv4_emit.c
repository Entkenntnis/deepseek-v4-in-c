/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_emit.c - the line-delimited events.
 *
 * Shape: `<event> key=value key=value ...`. Only fields consumed by a default
 * TUI/e2e/prefill view are emitted. Strings are hex through
 * dsv4_emit_hex_field(); names and numbers are bare ASCII.
 */
#include <stdio.h>

#include "dsv4_serve_int.h"

void dsv4_emit_ready(const Engine *e, const char *event)
{
    printf("%s config_hidden=%d config_layers=%d config_experts=%d "
           "config_topk=%d context_used=%d context_cap=%d",
           event, e->c.hidden, e->c.n_layers, e->c.n_experts, e->c.topk,
           e->n_ids, e->cap_ids);
    dsv4_write_sys(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_start(const Engine *e, int start_pos, int prompt_tokens,
                     int gen_max, int prefill_tokens)
{
    printf("start pos=%d prompt_tokens=%d gen_max=%d prefill_tokens=%d "
           "context_cap=%d",
           start_pos, prompt_tokens, gen_max, prefill_tokens, e->cap_ids);
    dsv4_write_sys(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_layer(const Engine *e, const char *phase, int layer)
{
    (void)e;
    printf("layer layer=%d phase=%s\n", layer, phase);
    fflush(stdout);
}

void dsv4_emit_token(const Engine *e, int32_t id, int index,
                     double t_turn, double ttft)
{
    printf("token id=%d index=%d t=%.4f ttft=%.4f "
           "prefill_s=%.4f prefill_tokens=%d",
           id, index, t_turn, ttft, e->prefill_s, e->prefill_tokens);
    dsv4_write_sys(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_reset(const Engine *e, const char *reason)
{
    (void)e;
    printf("reset reason=%s\n", reason);
    fflush(stdout);
}

void dsv4_emit_done(const Engine *e, int prompt_tokens, int generated,
                    double seconds, double ttft, const char *stop)
{
    printf("done prompt_tokens=%d generated=%d seconds=%.4f "
           "s_per_token=%.4f ttft=%.4f stop=%s context_used=%d head_s=%.4f",
           prompt_tokens, generated, seconds,
           generated ? seconds / (double)generated : 0.0,
           ttft, stop, e->n_ids, e->head_s);
    dsv4_write_sys(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
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
    dsv4_write_sys(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
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

/* A cache miss is not an engine error: the caller can fall back to a normal
 * prefill. Keeping it on its own event lets the TUI say so without painting the
 * transcript red. */
void dsv4_emit_kv_error(const char *message)
{
    fputs("kv_error ", stdout);
    dsv4_emit_hex_field("message", message);
    putchar('\n');
    fflush(stdout);
}

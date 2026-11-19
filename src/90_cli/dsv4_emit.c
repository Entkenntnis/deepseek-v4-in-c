/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_emit.c - the line-delimited events.
 *
 * Shape: `<event> key=value key=value ...`. Names and numbers are bare ASCII;
 * every string is written by dsv4_emit_hex_field() as `key=hex:...`, which is
 * why this file needs no quoting or Unicode handling.
 */
#include <stdio.h>
#include <string.h>

#include "dsv4_serve_int.h"
#include "dsv4_threads.h"

void dsv4_emit_ready(const Engine *e, const char *event)
{
    printf("%s", event);
    putchar(' ');
    dsv4_emit_hex_field("model", e->model_dir);
    putchar(' ');
    dsv4_emit_hex_field("threads", dsv4_threads_desc());
    printf(" config_hidden=%d config_layers=%d config_vocab=%d "
           "config_heads=%d config_head_dim=%d config_experts=%d "
           "config_topk=%d config_moe_inter=%d config_sliding_window=%d "
           "config_index_topk=%d config_hc_mult=%d config_max_pos=%d "
           "config_compress_ratios=",
           e->c.hidden, e->c.n_layers, e->c.vocab,
           e->c.n_heads, e->c.head_dim, e->c.n_experts, e->c.topk,
           e->c.moe_inter, e->c.sliding_window, e->c.index_topk,
           e->c.hc_mult, e->c.max_pos);
    for (int L = 0; L < e->c.n_layers; L++)
        printf("%s%d", L ? "," : "", e->c.compress_ratios[L]);
    printf(" context_used=%d context_cap=%d gen_default=%d think_default=%d",
           e->n_ids, e->cap_ids, e->gen_default, e->think_default);
    dsv4_write_sys_full(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_start(const Engine *e, int start_pos, int prompt_tokens,
                     int gen_max, const char *mode, int prefill_tokens)
{
    printf("start pos=%d prompt_tokens=%d gen_max=%d mode=%s "
           "prefill_tokens=%d context_cap=%d",
           start_pos, prompt_tokens, gen_max, mode, prefill_tokens,
           e->cap_ids);
    dsv4_write_sys_full(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    dsv4_write_prof(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_layer(const Engine *e, int pos, const char *phase, int layer,
                     double t_turn)
{
    printf("layer pos=%d phase=%s layer=%d layers=%d t=%.4f",
           pos, phase, layer, e->c.n_layers, t_turn);
    dsv4_write_sys_full(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    dsv4_write_prof(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_token(const Engine *e, int32_t id, int index, int pos,
                     double t_turn, double ttft, const TopK *top)
{
    printf("token id=%d index=%d pos=%d t=%.4f ttft=%.4f "
           "prefill_s=%.4f prefill_tokens=%d top=",
           id, index, pos, t_turn, ttft,
           e->prefill_s, e->prefill_tokens);
    for (int k = 0; k < top->n; k++)
        printf("%s%d:%.6f:%.6f", k ? "," : "", top->id[k],
               top->prob[k], (double)top->logit[k]);
    dsv4_write_sys_full(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    dsv4_write_prof(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_reset(const Engine *e, const char *reason)
{
    printf("reset reason=%s", reason);
    dsv4_write_sys_full(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_done(const Engine *e, int prompt_tokens, int generated,
                    double seconds, double ttft, const char *stop)
{
    const double tps = seconds > 0.0 ? (double)generated / seconds : 0.0;
    printf("done prompt_tokens=%d generated=%d seconds=%.4f tps=%.4f "
           "s_per_token=%.4f ttft=%.4f stop=%s context_used=%d",
           prompt_tokens, generated, seconds, tps,
           generated ? seconds / (double)generated : 0.0,
           ttft, stop, e->n_ids);
    dsv4_write_sys_full(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    dsv4_write_prof(e);
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
    dsv4_write_sys_full(e);
    dsv4_write_cache(e);
    dsv4_write_trunk(e);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_kv_built(const char *path, int tokens, double seconds,
                        long long bytes)
{
    printf("kv_built tokens=%d seconds=%.4f bytes=%lld ", tokens, seconds,
           bytes);
    dsv4_emit_hex_field("path", path);
    putchar('\n');
    fflush(stdout);
}

void dsv4_emit_kv_loaded(const char *path, int tokens, int context_cap,
                         double seconds)
{
    printf("kv_loaded tokens=%d used=%d context_cap=%d seconds=%.4f ",
           tokens, tokens, context_cap, seconds);
    dsv4_emit_hex_field("path", path);
    putchar('\n');
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

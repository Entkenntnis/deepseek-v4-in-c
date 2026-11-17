/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_emit.c - the line-delimited JSON events. */
#include <stdio.h>
#include <string.h>

#include "dsv4_serve_int.h"
#include "dsv4_threads.h"

void dsv4_emit_ready(const Engine *e, const char *event)
{
    printf("{\"event\":\"%s\",\"model\":", event);
    dsv4_json_escape(e->model_dir);
    printf(",\"threads\":");
    dsv4_json_escape(dsv4_threads_desc());
    printf(",\"config\":{\"hidden\":%d,\"layers\":%d,\"vocab\":%d,"
           "\"heads\":%d,\"head_dim\":%d,\"experts\":%d,\"topk\":%d,"
           "\"moe_inter\":%d,\"sliding_window\":%d,\"index_topk\":%d,"
           "\"hc_mult\":%d,\"max_pos\":%d,\"compress_ratios\":[",
           e->c.hidden, e->c.n_layers, e->c.vocab,
           e->c.n_heads, e->c.head_dim, e->c.n_experts, e->c.topk,
           e->c.moe_inter, e->c.sliding_window, e->c.index_topk,
           e->c.hc_mult, e->c.max_pos);
    for (int L = 0; L < e->c.n_layers; L++)
        printf("%s%d", L ? "," : "", e->c.compress_ratios[L]);
    printf("]},\"context\":{\"used\":%d,\"cap\":%d},\"gen_default\":%d,"
           "\"think_default\":%d,", e->n_ids, e->cap_ids,
           e->gen_default, e->think_default);
    dsv4_write_sys_full(e);
    putchar(',');
    dsv4_write_cache(e);
    putchar(',');
    dsv4_write_trunk(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

void dsv4_emit_start(const Engine *e, int start_pos, int prompt_tokens,
                     int gen_max, const char *mode, int prefill_tokens)
{
    printf("{\"event\":\"start\",\"pos\":%d,\"prompt_tokens\":%d,"
           "\"gen_max\":%d,\"mode\":\"%s\",\"prefill_tokens\":%d,"
           "\"context_cap\":%d,",
           start_pos, prompt_tokens, gen_max, mode, prefill_tokens,
           e->cap_ids);
    dsv4_write_sys_full(e);
    putchar(',');
    dsv4_write_cache(e);
    putchar(',');
    dsv4_write_trunk(e);
    putchar(',');
    dsv4_write_prof(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

void dsv4_emit_layer(const Engine *e, int pos, const char *phase, int layer,
                     double t_turn)
{
    printf("{\"event\":\"layer\",\"pos\":%d,\"phase\":\"%s\",\"layer\":%d,"
           "\"layers\":%d,\"t\":%.4f,", pos, phase, layer, e->c.n_layers,
           t_turn);
    dsv4_write_sys_full(e);
    putchar(',');
    dsv4_write_cache(e);
    putchar(',');
    dsv4_write_trunk(e);
    putchar(',');
    dsv4_write_prof(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

void dsv4_emit_token(const Engine *e, int32_t id, int index, int pos,
                     double t_turn, double ttft, const TopK *top)
{
    int nb = dsv4_tok_decode(&e->tok, id, e->tok_bytes, e->tok_cap);
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < nb; i++) {
        e->hexbuf[2 * i]     = hx[(e->tok_bytes[i] >> 4) & 15];
        e->hexbuf[2 * i + 1] = hx[e->tok_bytes[i] & 15];
    }
    e->hexbuf[2 * nb] = 0;

    printf("{\"event\":\"token\",\"id\":%d,\"hex\":\"%s\",\"index\":%d,"
           "\"pos\":%d,\"t\":%.4f,\"ttft\":%.4f,"
           "\"prefill_s\":%.4f,\"prefill_tokens\":%d,\"top\":[",
           id, e->hexbuf, index, pos, t_turn, ttft,
           e->prefill_s, e->prefill_tokens);
    for (int k = 0; k < top->n; k++)
        printf("%s[%d,%.6f,%.6f]", k ? "," : "", top->id[k],
               top->prob[k], (double)top->logit[k]);
    putchar(']');
    putchar(',');
    dsv4_write_sys_full(e);
    putchar(',');
    dsv4_write_cache(e);
    putchar(',');
    dsv4_write_trunk(e);
    putchar(',');
    dsv4_write_prof(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

void dsv4_emit_reset(const Engine *e, const char *reason)
{
    printf("{\"event\":\"reset\",\"reason\":\"%s\",", reason);
    dsv4_write_sys_full(e);
    putchar(',');
    dsv4_write_cache(e);
    putchar(',');
    dsv4_write_trunk(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

void dsv4_emit_done(const Engine *e, int prompt_tokens, int generated,
                    double seconds, double ttft, const char *stop)
{
    const double tps = seconds > 0.0 ? (double)generated / seconds : 0.0;
    printf("{\"event\":\"done\",\"prompt_tokens\":%d,\"generated\":%d,"
           "\"seconds\":%.4f,\"tps\":%.4f,\"s_per_token\":%.4f,"
           "\"ttft\":%.4f,\"stop\":\"%s\",\"context_used\":%d,",
           prompt_tokens, generated, seconds, tps,
           generated ? seconds / (double)generated : 0.0,
           ttft, stop, e->n_ids);
    dsv4_write_sys_full(e);
    putchar(',');
    dsv4_write_cache(e);
    putchar(',');
    dsv4_write_trunk(e);
    putchar(',');
    dsv4_write_prof(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

/* The block is the whole prompt: every layer walks all of it before the next
 * layer starts, so tokens_done is an estimate while a layer is mid-flight. */
void dsv4_emit_prefill_progress(Engine *e, const DSV4PrefillProgress *p)
{
    printf("{\"event\":\"prefill\",\"stage\":\"%s\",\"layer\":%d,"
           "\"layers\":%d,\"tokens_done\":%d,\"tokens_total\":%d,"
           "\"elapsed\":%.4f,\"progress\":%.4f,",
           p->stage, p->layer, p->layers, p->tokens_done, p->tokens_total,
           p->elapsed, p->progress);
    dsv4_write_sys_full(e);
    putchar(',');
    dsv4_write_cache(e);
    putchar(',');
    dsv4_write_trunk(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

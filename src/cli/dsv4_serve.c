/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_serve.c - a JSON-lines front end for the TUI, and nothing else.
 *
 * WHY THIS EXISTS
 *   dsv4_run.c is deliberately one-shot: it loads the model, prints ids, reports
 *   totals, and exits. That is the right shape for a gate or a benchmark, and the
 *   wrong shape for a conversation, where the KV state must survive from one turn
 *   to the next and a 1.6 s/token model needs to show what it is doing WHILE it
 *   does it.
 *
 *   This binary keeps the engine resident and speaks one JSON object per line on
 *   stdout. Python reads those lines and paints a TUI. Nothing here is used by
 *   the correctness gates; this file is additive.
 *
 * PROTOCOL (line-delimited JSON)
 *   stdin  commands:
 *     {"cmd":"generate","prompt":"...","gen":128,"think":false,"layers":true}
 *     {"cmd":"reset"}
 *     {"cmd":"system","text":"..."}
 *     {"cmd":"stats"}
 *     {"cmd":"quit"}
 *   SIGUSR1:
 *     cancel the running turn at the next token boundary; the engine emits
 *     done with "stop":"cancel" and keeps the context usable for the next turn.
 *   stdout events:
 *     ready, start, layer, token, done, reset, stats, error
 *
 *   Token text is emitted as raw ORIGINAL BYTES in hex ("hex"). A BPE token can
 *   end in the middle of a UTF-8 character, so a JSON string would be invalid;
 *   the client buffers the bytes and decodes incrementally.
 *
 *   All timing fields are seconds. "t" is relative to the turn's "start" event.
 *   "sys" is process-level truth (RSS, peak RSS, CPU, bytes read from disk),
 *   while "cache" and "trunk" are the engine's own accounting.
 *
 * THE PENDING-TOKEN RULE
 *   When generation stops on EOS (or on the gen limit), the last token has been
 *   appended to the sequence but NOT forwarded: its KV entry is not in the layer
 *   state yet. The next turn must forward it before appending "<User>", or the
 *   attention window would have a hole at that position. The engine tracks this
 *   with `pending_forward` and pays the cost at the start of the NEXT turn
 *   instead of stalling the end of the previous one.
 */
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "dsv4_cfg.h"
#include "dsv4_bind.h"
#include "dsv4_mem.h"
#include "dsv4_trunk.h"
#include "dsv4_cache.h"
#include "dsv4_layer.h"
#include "dsv4_tok.h"
#include "dsv4_threads.h"
#include "dsv4_prefill.h"
#include "json.h"

/* Kernel entry points that intentionally live outside a public header. */
void dsv4_rmsnorm(float *, const float *, const float *, int, float);
void dsv4_mmq(float *, const float *, const DSV4QMat *);
void dsv4_rope_table(float *, float *, int, int, int, double, double, double, double);

/* The TUI sends SIGUSR1 while a turn is running. Generation is one long
 * synchronous call, so a command on stdin cannot be read until it returns;
 * a signal flag lets the decode loop stop at the next token boundary instead. */
static volatile sig_atomic_t cancel_requested;

static void handle_cancel(int sig)
{
    (void)sig;
    cancel_requested = 1;
}

/* Tokenizer entry points. dsv4_tok.h documents decode() but, like dsv4_run.c,
 * leaves the rest to each CLI so the header stays about the algorithm. */
int  dsv4_tok_load(DSV4Tok *, const char *);
void dsv4_tok_free(DSV4Tok *);
int  dsv4_tok_encode(const DSV4Tok *, const char *, int, int32_t *, int);
int32_t dsv4_tok_find(const DSV4Tok *, const char *, int);

typedef struct {
    const char      *model_dir;
    DSV4Cfg          c;
    int              cr[DSV4_MAX_LAYERS];

    DSV4St           st;
    DSV4Tok          tok;
    DSV4Trunk        tr;
    DSV4Cache        cache;
    DSV4ModelBind    mb;
    DSV4Scratch      scratch;
    DSV4LayerState  *lst;
    DSV4ExpertSrc    src;      /* one source, shared by decode and prefill */
    DSV4Prefill     *pf;       /* NULL when batched prefill is disabled */
    int              prefill_batch;

    float           *cs_d, *sn_d, *cs_c, *sn_c;
    int              maxpos;

    float           *h, *xh, *logits;

    int32_t         *ids;
    int              n_ids, cap_ids;
    int32_t          eos_id;
    int              pending_forward;

    /* Last turn's prompt timing, so the TUI can print an honest prefill line
     * without subtracting the output head from ttft. */
    double           prefill_s;
    int              prefill_tokens;

    char            *system_prompt;
    int              think_default;
    int              gen_default;
    int              emit_layers_default;

    int32_t         *enc_tmp;
    int              enc_cap;
    unsigned char   *tok_bytes;
    int              tok_cap;
    char            *hexbuf;
    int              hex_cap;

    double           t_start;
    DSV4Prof         prof_base;
} Engine;

/* --------------------------------------------------------------- timing --- */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double proc_cpu_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void proc_mem_mb(double *rss_mb, double *peak_mb)
{
    *rss_mb = 0.0;
    *peak_mb = 0.0;
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            double kb;
            if (sscanf(line, "VmRSS: %lf kB", &kb) == 1) *rss_mb = kb / 1024.0;
            else if (sscanf(line, "VmHWM: %lf kB", &kb) == 1) *peak_mb = kb / 1024.0;
        }
        fclose(f);
    }
    if (*peak_mb <= 0.0) {
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0)
            *peak_mb = (double)ru.ru_maxrss / 1024.0;   /* Linux: KiB */
    }
}

static double proc_read_gb(void)
{
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return 0.0;
    char line[256];
    unsigned long long bytes = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "read_bytes: %llu", &bytes) == 1) break;
    }
    fclose(f);
    return (double)bytes / 1073741824.0;
}

/* ----------------------------------------------------------- JSON bits ---- */

static void json_escape(const char *s)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        switch (*p) {
        case '"':  fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\b': fputs("\\b", stdout); break;
        case '\f': fputs("\\f", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if (*p < 0x20) printf("\\u%04x", *p);
            else putchar(*p);
        }
    }
    putchar('"');
}

static void emit_error(const char *fmt, ...)
{
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fputs("{\"event\":\"error\",\"message\":", stdout);
    json_escape(msg);
    fputs("}\n", stdout);
    fflush(stdout);
}

static jval *jget(jval *o, const char *key) { return json_get(o, key); }

static const char *jstr(jval *o, const char *key)
{
    jval *v = jget(o, key);
    return v && v->t == J_STR ? v->str : NULL;
}

static double jnum(jval *o, const char *key, double dflt)
{
    jval *v = jget(o, key);
    return v && v->t == J_NUM ? v->num : dflt;
}

static int jbool(jval *o, const char *key, int dflt)
{
    jval *v = jget(o, key);
    if (!v) return dflt;
    if (v->t == J_BOOL) return v->boolean ? 1 : 0;
    if (v->t == J_NUM)  return v->num != 0.0;
    return dflt;
}

/* json.h deliberately has no free: it was written for parse-once-at-startup
 * files. A command stream parses repeatedly, so this walks the tree and frees
 * every j_dup'd string too. */
static void jfree(jval *v)
{
    if (!v) return;
    if (v->t == J_STR) free(v->str);
    for (int i = 0; i < v->len; i++) {
        if (v->kids) jfree(v->kids[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->kids);
    free(v->keys);
    free(v);
}

/* ------------------------------------------------------------- statistics -- */

/* Process truth. wall_s is uptime since the engine finished loading. */
static void write_sys_full(const Engine *e)
{
    double rss, peak;
    proc_mem_mb(&rss, &peak);
    printf("\"sys\":{\"rss_mb\":%.1f,\"peak_mb\":%.1f,\"io_read_gb\":%.4f,"
           "\"cpu_s\":%.3f,\"wall_s\":%.3f}",
           rss, peak, proc_read_gb(), proc_cpu_s(), now_s() - e->t_start);
}

static void write_cache(const Engine *e)
{
    const DSV4Cache *c = &e->cache;
    const uint64_t n = c->hits + c->misses;
    printf("\"cache\":{\"slots\":%d,\"hit\":%llu,\"miss\":%llu,\"evict\":%llu,"
           "\"hit_rate\":%.4f,\"read_gb\":%.4f,\"resident_gb\":%.3f,"
           "\"slot_mb\":%.2f,\"pf\":%d,\"pf_hit\":%llu,\"pf_load\":%llu,"
           "\"pf_read_gb\":%.4f,\"pf_wait_ms\":%.1f}",
           c->nslot,
           (unsigned long long)c->hits, (unsigned long long)c->misses,
           (unsigned long long)c->evictions,
           n ? (double)c->hits / (double)n : 0.0,
           (double)(c->bytes_read + c->pf_bytes) / 1073741824.0,
           (double)c->slot_bytes * (double)c->nslot / 1073741824.0,
           (double)c->slot_bytes / 1048576.0,
           c->pf_enabled,
           (unsigned long long)c->pf_hits,
           (unsigned long long)c->pf_loads,
           (double)c->pf_bytes / 1073741824.0,
           c->pf_wait_s * 1e3);
}

static void write_trunk(const Engine *e)
{
    const DSV4Trunk *t = &e->tr;
    const uint64_t n = t->hits + t->misses;
    printf("\"trunk\":{\"layers\":%d,\"pinned\":%d,\"slots\":%d,"
           "\"slot_mb\":%.2f,\"hit\":%llu,\"miss\":%llu,\"hit_rate\":%.4f,"
           "\"read_gb\":%.4f,\"load_s\":%.3f,\"resident_gb\":%.3f}",
           t->n_layers, t->npin, t->nslot,
           (double)t->slot_bytes / 1048576.0,
           (unsigned long long)t->hits, (unsigned long long)t->misses,
           n ? (double)t->hits / (double)n : 0.0,
           (double)t->bytes_read / 1073741824.0, t->load_seconds,
           (double)dsv4_trunk_resident_bytes(t) / 1073741824.0);
}

static void write_prof(const Engine *e)
{
    const DSV4Prof *p = &dsv4_prof;
    const double hc  = p->hc        - e->prof_base.hc;
    const double at  = p->attn      - e->prof_base.attn;
    const double ga  = p->gate      - e->prof_base.gate;
    const double io  = p->expert_io - e->prof_base.expert_io;
    const double mm  = p->expert_mm - e->prof_base.expert_mm;
    const double sh  = p->shared    - e->prof_base.shared;
    const double hd  = p->head      - e->prof_base.head;
    const double bd  = p->bind      - e->prof_base.bind;
    printf("\"prof\":{\"hc\":%.4f,\"attn\":%.4f,\"gate\":%.4f,"
           "\"expert_io\":%.4f,\"expert_mm\":%.4f,\"shared\":%.4f,"
           "\"head\":%.4f,\"bind\":%.4f,\"accounted\":%.4f}",
           hc, at, ga, io, mm, sh, hd, bd,
           hc + at + ga + io + mm + sh + hd + bd);
}

static void emit_ready(const Engine *e, const char *event)
{
    printf("{\"event\":\"%s\",\"model\":", event);
    json_escape(e->model_dir);
    printf(",\"threads\":");
    json_escape(dsv4_threads_desc());
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
    write_sys_full(e);
    putchar(',');
    write_cache(e);
    putchar(',');
    write_trunk(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

static void emit_start(const Engine *e, int start_pos, int prompt_tokens,
                       int gen_max, const char *mode, int prefill_batch)
{
    printf("{\"event\":\"start\",\"pos\":%d,\"prompt_tokens\":%d,"
           "\"gen_max\":%d,\"mode\":\"%s\",\"prefill_batch\":%d,"
           "\"context_cap\":%d,",
           start_pos, prompt_tokens, gen_max, mode, prefill_batch,
           e->cap_ids);
    write_sys_full(e);
    putchar(',');
    write_cache(e);
    putchar(',');
    write_trunk(e);
    putchar(',');
    write_prof(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

static void emit_layer(const Engine *e, int pos, const char *phase, int layer,
                       double t_turn)
{
    printf("{\"event\":\"layer\",\"pos\":%d,\"phase\":\"%s\",\"layer\":%d,"
           "\"layers\":%d,\"t\":%.4f,", pos, phase, layer, e->c.n_layers, t_turn);
    write_sys_full(e);
    putchar(',');
    write_cache(e);
    putchar(',');
    write_trunk(e);
    putchar(',');
    write_prof(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

typedef struct {
    int    id[5];
    float  logit[5];
    double prob[5];
    int    n;
} TopK;

static void emit_token(const Engine *e, int32_t id, int index, int pos,
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
    write_sys_full(e);
    putchar(',');
    write_cache(e);
    putchar(',');
    write_trunk(e);
    putchar(',');
    write_prof(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

static void emit_done(const Engine *e, int prompt_tokens, int generated,
                      double seconds, double ttft, const char *stop)
{
    const double tps = seconds > 0.0 ? (double)generated / seconds : 0.0;
    printf("{\"event\":\"done\",\"prompt_tokens\":%d,\"generated\":%d,"
           "\"seconds\":%.4f,\"tps\":%.4f,\"s_per_token\":%.4f,"
           "\"ttft\":%.4f,\"stop\":\"%s\",\"context_used\":%d,",
           prompt_tokens, generated, seconds, tps,
           generated ? seconds / (double)generated : 0.0,
           ttft, stop, e->n_ids);
    write_sys_full(e);
    putchar(',');
    write_cache(e);
    putchar(',');
    write_trunk(e);
    putchar(',');
    write_prof(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

/* Prefill progress. The batched path is layer-major: a chunk of prompt tokens
 * walks every layer, so "tokens done" is an estimate until the chunk closes.
 * The event carries the stage, chunk/layer counters and the live cache/trunk/
 * process stats the TUI's existing MEMORY block already knows how to draw. */
static void emit_prefill_progress(Engine *e, const DSV4PrefillProgress *p)
{
    printf("{\"event\":\"prefill\",\"stage\":\"%s\",\"chunk\":%d,\"chunks\":%d,"
           "\"layer\":%d,\"layers\":%d,\"tokens_done\":%d,\"tokens_total\":%d,"
           "\"chunk_tokens\":%d,\"batch\":%d,\"elapsed\":%.4f,"
           "\"progress\":%.4f,",
           p->stage, p->chunk, p->chunks, p->layer, p->layers,
           p->tokens_done, p->tokens_total, p->chunk_tokens, p->batch,
           p->elapsed, p->progress);
    write_sys_full(e);
    putchar(',');
    write_cache(e);
    putchar(',');
    write_trunk(e);
    fputs("}\n", stdout);
    fflush(stdout);
}

static void prefill_progress_cb(void *ctx, const DSV4PrefillProgress *p)
{
    emit_prefill_progress((Engine *)ctx, p);
}

/* ----------------------------------------------------------- engine core -- */

/* The head's own hc reduction. Copied from dsv4_run.c on purpose: it is small,
 * it is model math, and dsv4_run.c is a self-contained CLI rather than a
 * library. If a third caller appears, this moves into dsv4_layer.c. */
static void hc_head(float *y, const float *x, const DSV4HcW *w,
                    int hc, int hidden, float norm_eps, float hc_eps)
{
    const int hcdim = hc * hidden;
    double sq = 0.0;
    for (int i = 0; i < hcdim; i++) sq += (double)x[i] * x[i];
    const float rs = (float)(1.0 / sqrt(sq / (double)hcdim + (double)norm_eps));

    float pre[DSV4_MAX_HC_MULT];
    for (int j = 0; j < hc; j++) {
        const float *row = w->fn + (size_t)j * hcdim;
        double acc = 0.0;
        for (int i = 0; i < hcdim; i++) acc = fma((double)row[i], (double)x[i], acc);
        const float m = (float)acc * rs;
        pre[j] = 1.0f / (1.0f + expf(-(m * w->scale[0] + w->base[j]))) + hc_eps;
    }
    for (int d = 0; d < hidden; d++) {
        double acc = 0.0;
        for (int j = 0; j < hc; j++)
            acc += (double)pre[j] * (double)x[(size_t)j * hidden + d];
        y[d] = (float)acc;
    }
}

/* Expert source: the cache, carried through ctx. */
static const DSV4ExpertW *run_get_expert(void *ctx, int layer, int expert)
{
    return dsv4_cache_get((DSV4Cache *)ctx, layer, expert);
}

static int run_get_experts(void *ctx, int layer, const int *e, int n,
                           const DSV4ExpertW **out)
{
    return dsv4_cache_get_many((DSV4Cache *)ctx, layer, e, n, out);
}

static DSV4CacheBatch g_batch;

static int run_begin_experts(void *ctx, int layer, const int *e, int n,
                             const DSV4ExpertW **out)
{
    return dsv4_cache_get_many_begin((DSV4Cache *)ctx, layer, e, n, out, &g_batch);
}

static int run_end_experts(void *ctx, const DSV4ExpertW **out)
{
    return dsv4_cache_get_many_end((DSV4Cache *)ctx, out, &g_batch);
}

static int run_wait_any(void *ctx, int *k)
{
    return dsv4_cache_wait_any((DSV4Cache *)ctx, &g_batch, k);
}

/* 0 = old end() barrier; >0 = miss reads in flight per layer. */
static int g_stream_miss = 6;

static int engine_init(Engine *e, const char *model, const char *trunkdir,
                       const char *tokfile, double budget_gb, int budget_auto,
                       int ctx, int gen_default, int think_default,
                       int layers_default, int prefill_batch)
{
    memset(e, 0, sizeof *e);
    e->model_dir = model;
    e->gen_default = gen_default;
    e->think_default = think_default;
    e->emit_layers_default = layers_default;
    e->t_start = now_s();
    memset(&dsv4_prof, 0, sizeof dsv4_prof);
    e->prof_base = dsv4_prof;

    if (ctx < 512) ctx = 512;
    e->maxpos = ctx;
    e->cap_ids = ctx;

    char path[1024];
    snprintf(path, sizeof path, "%s/config.json", model);
    /* dsv4_cfg_load_file() prints a human-readable banner on stdout, which is
     * exactly what the one-shot CLI wants and exactly what a JSON-lines reader
     * cannot tolerate. Send that one call's stdout to stderr, then restore. */
    fflush(stdout);
    const int saved_out = dup(STDOUT_FILENO);
    if (saved_out >= 0) dup2(STDERR_FILENO, STDOUT_FILENO);
    const int cfg_ok = dsv4_cfg_load_file(&e->c, e->cr, DSV4_MAX_LAYERS, path);
    fflush(stdout);
    if (saved_out >= 0) {
        dup2(saved_out, STDOUT_FILENO);
        close(saved_out);
    }
    if (!cfg_ok) return -1;

    if (dsv4_st_open(&e->st, model) != 0) return -1;
    if (dsv4_tok_load(&e->tok, tokfile) != 0) return -1;
    if ((int)e->tok.n_vocab != e->c.vocab) {
        fprintf(stderr, "tokenizer has %u ids, config says %d; refusing\n",
                e->tok.n_vocab, e->c.vocab);
        return -1;
    }

    int64_t budget = (int64_t)(budget_gb * 1073741824.0);
    if (budget_auto) {
        const int64_t avail = dsv4_mem_available_bytes();
        const int64_t model_bytes = dsv4_bind_model_bytes(&e->st, &e->c, 1);
        const int64_t reserve = (int64_t)1 << 30;       /* 1 GiB spare */
        /* Scratch, KV state, rope tables, tokenizer and allocator overhead.
         * The budget does not cover these; measured at about 0.34 GiB on
         * Flash. Keep half a gigabyte so the 1 GiB above is actually spare. */
        const int64_t runtime = (int64_t)512 << 20;
        if (avail <= 0) {
            fprintf(stderr, "--budget auto: cannot read available memory; "
                            "pass an explicit --budget GB\n");
            return -1;
        }
        if (model_bytes < 0) {
            fprintf(stderr, "--budget auto: cannot size the model's embed/head "
                            "weights; pass an explicit --budget GB\n");
            return -1;
        }
        budget = avail - reserve - model_bytes - runtime;
        if (budget <= 0) {
            fprintf(stderr, "--budget auto: %.2f GiB available, %.2f GiB model, "
                            "%.2f GiB reserve + runtime leaves nothing\n",
                    (double)avail / 1073741824.0,
                    (double)model_bytes / 1073741824.0,
                    (double)(reserve + runtime) / 1073741824.0);
            return -1;
        }
        fprintf(stderr, "--budget auto: %.2f GiB available - %.2f GiB model - "
                        "%.2f GiB reserve - %.2f GiB runtime = %.2f GiB budget\n",
                (double)avail / 1073741824.0, (double)model_bytes / 1073741824.0,
                (double)reserve / 1073741824.0,
                (double)runtime / 1073741824.0,
                (double)budget / 1073741824.0);
    }

    const int64_t efloor = (int64_t)e->c.topk * dsv4_cache_slot_bytes(&e->c);
    if (budget <= efloor) {
        fprintf(stderr, "--budget %.2f GB cannot work: one layer's %d experts "
                        "alone need %.2f GB\n",
                (double)budget / 1073741824.0, e->c.topk,
                (double)efloor / 1073741824.0);
        return -1;
    }
    if (dsv4_trunk_open(&e->tr, trunkdir, &e->c, budget - efloor) != 0) return -1;
    const int64_t left = budget - dsv4_trunk_resident_bytes(&e->tr);
    if (dsv4_cache_init(&e->cache, &e->st, &e->c, left) != 0) return -1;
    if (g_stream_miss > 0) {
        dsv4_cache_set_stream_cap(&e->cache, g_stream_miss);
        fprintf(stderr, "stream-miss: up to %d miss reads in flight; experts "
                        "are consumed as they land\n", e->cache.stream_cap);
    }
    e->src = (DSV4ExpertSrc){ run_get_expert, run_get_experts,
                              run_begin_experts, run_end_experts,
                              run_wait_any, &e->cache };
    if (dsv4_bind_model(&e->st, &e->c, 1, &e->mb) != 0) return -1;

    e->cs_d = malloc((size_t)e->maxpos * (e->c.qk_rope / 2) * sizeof(float));
    e->sn_d = malloc((size_t)e->maxpos * (e->c.qk_rope / 2) * sizeof(float));
    e->cs_c = malloc((size_t)e->maxpos * (e->c.qk_rope / 2) * sizeof(float));
    e->sn_c = malloc((size_t)e->maxpos * (e->c.qk_rope / 2) * sizeof(float));
    if (!e->cs_d || !e->sn_d || !e->cs_c || !e->sn_c) return -1;
    dsv4_rope_table(e->cs_d, e->sn_d, e->c.qk_rope, e->maxpos, 0, e->c.rope_theta,
                    e->c.yarn_factor, e->c.yarn_beta_fast, e->c.yarn_beta_slow);
    dsv4_rope_table(e->cs_c, e->sn_c, e->c.qk_rope, e->maxpos, e->c.yarn_orig_ctx,
                    e->c.compress_rope_theta, e->c.yarn_factor,
                    e->c.yarn_beta_fast, e->c.yarn_beta_slow);

    if (dsv4_scratch_init(&e->scratch, &e->c, e->maxpos) != 0) return -1;
    e->lst = calloc((size_t)e->c.n_layers, sizeof(DSV4LayerState));
    if (!e->lst) return -1;
    for (int L = 0; L < e->c.n_layers; L++)
        if (dsv4_state_init(&e->lst[L], &e->c, L, e->maxpos) != 0) return -1;

    if (prefill_batch > 1) {
        int pb = prefill_batch;
        if (pb > DSV4_MAX_BATCH) pb = DSV4_MAX_BATCH;
        const DSV4PrefillOpts po = {
            .batch = pb, .max_pos = e->maxpos,
            .on_progress = prefill_progress_cb, .progress_ctx = e,
        };
        if (dsv4_prefill_init(&e->pf, &e->c, &e->tr, e->lst, &e->src,
                              &e->mb.w, &po) != 0) {
            fprintf(stderr, "serve: batched prefill init failed; falling back "
                            "to the per-token path\n");
            e->pf = NULL;
        } else {
            e->prefill_batch = pb;
            fprintf(stderr, "serve: batched dense prefill, %d tokens/chunk\n",
                    pb);
        }
    }

    const int hc = e->c.hc_mult, d = e->c.hidden;
    e->h = calloc((size_t)hc * d, sizeof(float));
    e->xh = calloc((size_t)d, sizeof(float));
    e->logits = calloc((size_t)e->c.vocab, sizeof(float));
    e->ids = calloc((size_t)e->cap_ids, sizeof(int32_t));
    if (!e->h || !e->xh || !e->logits || !e->ids) return -1;

    e->enc_cap = e->cap_ids * 2 + 1024;
    e->enc_tmp = malloc((size_t)e->enc_cap * sizeof(int32_t));
    e->tok_cap = 65536;
    e->tok_bytes = malloc((size_t)e->tok_cap);
    e->hex_cap = e->tok_cap * 2 + 1;
    e->hexbuf = malloc((size_t)e->hex_cap);
    if (!e->enc_tmp || !e->tok_bytes || !e->hexbuf) return -1;

    e->eos_id = dsv4_tok_find(&e->tok, DSV4_EOS, (int)strlen(DSV4_EOS));
    return 0;
}

static void engine_free(Engine *e)
{
    dsv4_prefill_free(e->pf);
    for (int L = 0; L < e->c.n_layers && e->lst; L++) dsv4_state_free(&e->lst[L]);
    free(e->lst);
    free(e->h); free(e->xh); free(e->logits); free(e->ids);
    free(e->cs_d); free(e->sn_d); free(e->cs_c); free(e->sn_c);
    free(e->enc_tmp); free(e->tok_bytes); free(e->hexbuf);
    dsv4_scratch_free(&e->scratch);
    dsv4_bind_model_free(&e->mb);
    dsv4_cache_free(&e->cache);
    dsv4_trunk_close(&e->tr);
    dsv4_tok_free(&e->tok);
    dsv4_st_close(&e->st);
    free(e->system_prompt);
    memset(e, 0, sizeof *e);
}

static int engine_reset_context(Engine *e)
{
    for (int L = 0; L < e->c.n_layers; L++) {
        dsv4_state_free(&e->lst[L]);
        if (dsv4_state_init(&e->lst[L], &e->c, L, e->maxpos) != 0) return -1;
    }
    e->n_ids = 0;
    e->pending_forward = 0;
    memset(&e->prof_base, 0, sizeof e->prof_base);
    memset(&dsv4_prof, 0, sizeof dsv4_prof);
    return 0;
}

/* Append `text` to the sequence, encoding into a temporary first so a context
 * overflow cannot leave a half-appended turn behind. */
static int append_text(Engine *e, const char *text)
{
    const int len = (int)strlen(text);
    const int need = len + 64;
    if (need > e->enc_cap) {
        int32_t *p = realloc(e->enc_tmp, (size_t)need * sizeof(int32_t));
        if (!p) return -1;
        e->enc_tmp = p;
        e->enc_cap = need;
    }
    int n = dsv4_tok_encode(&e->tok, text, len, e->enc_tmp, e->enc_cap);
    if (n < 0) return -1;
    if (n > e->cap_ids - e->n_ids) return -2;
    memcpy(e->ids + e->n_ids, e->enc_tmp, (size_t)n * sizeof(int32_t));
    e->n_ids += n;
    return n;
}

/* Greedy sample, plus the five most likely alternatives and their probabilities.
 * The probability sum is over the full vocab so the numbers are real, not a
 * normalisation over the top five. */
static int sample_top(Engine *e, TopK *top)
{
    const int V = e->c.vocab;
    const float *lg = e->logits;
    int best = 0;
    for (int v = 1; v < V; v++) if (lg[v] > lg[best]) best = v;

    const double mx = (double)lg[best];
    double sum = 0.0;
    for (int v = 0; v < V; v++) sum += exp((double)lg[v] - mx);
    if (!(sum > 0.0)) sum = 1.0;

    top->n = 0;
    for (int k = 0; k < 5; k++) top->logit[k] = -INFINITY;
    for (int v = 0; v < V; v++) {
        if ((double)lg[v] <= (double)top->logit[4]) continue;
        top->logit[4] = lg[v];
        top->id[4] = v;
        for (int k = 4; k > 0 && top->logit[k] > top->logit[k - 1]; k--) {
            float tf = top->logit[k]; top->logit[k] = top->logit[k - 1];
            top->logit[k - 1] = tf;
            int ti = top->id[k]; top->id[k] = top->id[k - 1];
            top->id[k - 1] = ti;
        }
        if (top->n < 5) top->n++;
    }
    for (int k = 0; k < top->n; k++)
        top->prob[k] = exp((double)top->logit[k] - mx) / sum;
    return best;
}

/* Run the output head on whatever residual stream is in e->h. Shared by the
 * per-token path and the batched prefill: the head's inputs are identical, and
 * keeping two copies of this would be one more place for them to drift. */
static void head_from_h(Engine *e, TopK *top)
{
    const int hc = e->c.hc_mult, d = e->c.hidden;
    const double th = now_s();
    hc_head(e->xh, e->h, &e->mb.w.hc_head, hc, d, e->c.rms_eps, e->c.hc_eps);
    dsv4_rmsnorm(e->xh, e->xh, e->mb.w.norm, d, e->c.rms_eps);
    DSV4QMat head = { e->mb.w.head, NULL, DSV4_WBF16, e->c.vocab, d, 0, 0 };
    dsv4_mmq(e->logits, e->xh, &head);
    dsv4_prof.head += now_s() - th;
    sample_top(e, top);
}

/* One token through every layer, optionally followed by the head. */
static int forward(Engine *e, int32_t tid, int pos, int want_logits,
                   int emit_layers, const char *phase, double t_turn, TopK *top)
{
    const int hc = e->c.hc_mult, d = e->c.hidden;

    if (e->cache.pf_enabled)
        dsv4_cache_prefetch_hash_layers(&e->cache, tid);

    dsv4_embed_row(e->xh, e->mb.w.embed, e->mb.w.wdt, tid, d);
    for (int j = 0; j < hc; j++)
        memcpy(e->h + (size_t)j * d, e->xh, (size_t)d * sizeof(float));

    for (int L = 0; L < e->c.n_layers; L++) {
        DSV4LayerBind lb;
        const double tb = now_s();
        const double io0 = e->tr.load_seconds;
        if (dsv4_trunk_bind(&e->tr, &e->c, L, &lb) != 0) {
            emit_error("trunk bind failed on layer %d at position %d", L, pos);
            return -1;
        }
        /* read_run() reports its own wall time; the rest is actual binding. */
        const double io = e->tr.load_seconds - io0;
        const double call = now_s() - tb;
        dsv4_prof.trunk_io += io;
        dsv4_prof.bind += call > io ? call - io : 0.0;
        const int comp = dsv4_compress_ratio(&e->c, L) != 0;
        dsv4_layer_forward(e->h, &lb.w, &e->c, &e->scratch, &e->src,
                           &e->lst[L], comp ? e->cs_c : e->cs_d,
                           comp ? e->sn_c : e->sn_d, pos, tid);
        if (emit_layers) emit_layer(e, pos, phase, L, t_turn);
    }

    if (want_logits) head_from_h(e, top);
    return 0;
}

static void engine_generate(Engine *e, const char *prompt, int ngen,
                            int think, int emit_layers)
{
    const double t0 = now_s();
    cancel_requested = 0;
    e->prof_base = dsv4_prof;

    if (ngen < 0) ngen = 0;
    if (e->eos_id < 0) {
        emit_error("tokenizer has no end-of-turn token; refusing to run");
        return;
    }

    /* Close a truncated previous turn, then forward the token that the previous
     * turn deliberately left pending. See the header note. */
    if (e->n_ids > 0 && e->ids[e->n_ids - 1] != e->eos_id) {
        if (e->n_ids >= e->cap_ids) {
            emit_error("context full (%d tokens); reset before continuing", e->cap_ids);
            return;
        }
        e->ids[e->n_ids++] = e->eos_id;
        e->pending_forward = 1;
    }
    if (e->pending_forward) {
        const int pos = e->n_ids - 1;
        TopK dummy;
        if (forward(e, e->ids[pos], pos, 0, emit_layers, "resume",
                    now_s() - t0, &dummy) != 0) return;
        e->pending_forward = 0;
    }

    const size_t sys_len = e->system_prompt ? strlen(e->system_prompt) : 0;
    const size_t need = strlen(prompt) + sys_len + 256;
    char *turn = malloc(need);
    if (!turn) { emit_error("out of memory formatting turn"); return; }

    if (e->n_ids == 0) {
        snprintf(turn, need, "%s%s%s%s%s%s%s",
                 DSV4_BOS,
                 e->system_prompt ? e->system_prompt : "",
                 e->system_prompt ? "\n" : "",
                 DSV4_USER, prompt, DSV4_ASSISTANT,
                 think ? DSV4_THINK_OPEN : DSV4_THINK_CLOSE);
    } else {
        snprintf(turn, need, "%s%s%s%s",
                 DSV4_USER, prompt, DSV4_ASSISTANT,
                 think ? DSV4_THINK_OPEN : DSV4_THINK_CLOSE);
    }

    const int start_pos = e->n_ids;
    const int n_added = append_text(e, turn);
    free(turn);
    if (n_added < 0) {
        if (n_added == -2)
            emit_error("context full (%d tokens): this turn needs more room. "
                       "Use /reset or raise --ctx.", e->cap_ids);
        else
            emit_error("tokenization failed");
        return;
    }
    if (n_added == 0) {
        emit_error("user message encoded to zero tokens");
        return;
    }

    const int use_batched = (e->pf && n_added > 1);
    emit_start(e, start_pos, n_added, ngen, think ? "thinking" : "chat",
               use_batched ? e->prefill_batch : 0);

    TopK top;
    memset(&top, 0, sizeof top);

    /* Prefill the new turn. The head runs only on the final prompt token; the
     * earlier ones would cost a vocab-sized matmul for a result nothing reads.
     *
     * With batched prefill the new tokens go through the dense GEMM path in
     * the same module the one-shot CLI uses. `start_pos` is where the previous
     * turn stopped, so a conversation's KV cache is extended rather than
     * rebuilt. This path cannot fall back after it starts: it has already
     * written KV and compressor state. An error is terminal for the turn.
     *
     * The timer covers the prompt only, not the head. The TUI prints this line
     * as a transcript message; using ttft there would include the head and,
     * on a follow-up turn, the pending token from the previous one. */
    e->prefill_s = 0.0;
    e->prefill_tokens = 0;
    const double tp0 = now_s();
    if (use_batched) {
        if (dsv4_prefill_range(e->pf, e->ids + start_pos, n_added, start_pos,
                               e->cs_d, e->sn_d, e->cs_c, e->sn_c,
                               e->h) != 0) {
            emit_error("batched prefill failed at position %d; reset the "
                       "context before continuing", start_pos);
            return;
        }
    } else {
        for (int i = start_pos; i < e->n_ids; i++)
            if (forward(e, e->ids[i], i, 0, emit_layers, "prompt",
                        now_s() - t0, &top) != 0) return;
    }
    e->prefill_s = now_s() - tp0;
    e->prefill_tokens = n_added;
    if (use_batched) {
        const int chunks = (n_added + e->prefill_batch - 1) / e->prefill_batch;
        const DSV4PrefillProgress hp = {
            "head", chunks, chunks, e->c.n_layers, e->c.n_layers,
            n_added, n_added, 0, e->prefill_batch, e->prefill_s, 1.0,
        };
        emit_prefill_progress(e, &hp);
    }
    head_from_h(e, &top);

    int generated = 0;
    double ttft = 0.0;
    const char *stop = "max";
    int best = top.n > 0 ? top.id[0] : 0;

    while (generated < ngen) {
        if (cancel_requested) {
            /* Everything emitted so far is already in the KV state. Closing
             * this partial turn at the next call is the same path as an EOS. */
            stop = "cancel";
            break;
        }
        if (e->n_ids >= e->cap_ids) {
            emit_error("context full (%d tokens) while generating", e->cap_ids);
            stop = "context";
            break;
        }
        const int32_t id = best;
        const int pos = e->n_ids;
        e->ids[e->n_ids++] = id;

        const double el = now_s() - t0;
        if (generated == 0) ttft = el;
        emit_token(e, id, generated, pos, el, ttft, &top);
        generated++;

        if (id == e->eos_id) {
            stop = "eos";
            e->pending_forward = 1;   /* forwarded at the start of next turn */
            break;
        }
        if (generated >= ngen) {
            stop = "max";
            e->pending_forward = 1;
            break;
        }

        if (forward(e, id, pos, 1, emit_layers, "gen",
                    now_s() - t0, &top) != 0) {
            e->pending_forward = 1;   /* the emitted token is not in the KV yet */
            stop = "error";
            break;
        }
        e->pending_forward = 0;
        best = top.n > 0 ? top.id[0] : 0;
    }

    const double dt = now_s() - t0;
    emit_done(e, n_added, generated, dt, ttft, stop);
}

/* -------------------------------------------------------------- commands -- */

static void handle_command(Engine *e, jval *v)
{
    const char *cmd = jstr(v, "cmd");
    if (!cmd) { emit_error("command object has no \"cmd\""); return; }

    if (!strcmp(cmd, "generate")) {
        const char *prompt = jstr(v, "prompt");
        if (!prompt) prompt = "";
        const int ngen = (int)jnum(v, "gen", e->gen_default);
        const int think = jbool(v, "think", e->think_default);
        const int layers = jbool(v, "layers", e->emit_layers_default);
        engine_generate(e, prompt, ngen, think, layers);
    } else if (!strcmp(cmd, "reset")) {
        if (engine_reset_context(e) != 0) emit_error("context reset failed");
        else {
            printf("{\"event\":\"reset\",");
            write_sys_full(e);
            putchar(',');
            write_cache(e);
            putchar(',');
            write_trunk(e);
            fputs("}\n", stdout);
            fflush(stdout);
        }
    } else if (!strcmp(cmd, "system")) {
        const char *text = jstr(v, "text");
        free(e->system_prompt);
        e->system_prompt = text ? strdup(text) : NULL;
        printf("{\"event\":\"system\",\"ok\":true,\"chars\":%d}\n",
               text ? (int)strlen(text) : 0);
        fflush(stdout);
    } else if (!strcmp(cmd, "stats")) {
        emit_ready(e, "stats");
    } else if (!strcmp(cmd, "ping")) {
        printf("{\"event\":\"pong\"}\n");
        fflush(stdout);
    } else if (!strcmp(cmd, "quit")) {
        /* handled by the caller */
    } else {
        emit_error("unknown command \"%s\"", cmd);
    }
}

static void usage(const char *me)
{
    fprintf(stderr,
            "usage: %s <model_dir> --trunk <dir> --tok <file> "
            "[--budget GB|auto] [--ctx N] [--gen N] [--think] [--no-layers]\n"
            "       [--threads N] [--pin off|all|p|e] [--readers N]\n"
            "       [--reader-pin off|all|p|e] [--stream-miss N (default 6)]\n"
            "       [--prefill-batch-size N (0 disables; default 0)]\n"
            "\n"
            "Speaks JSON lines on stdin/stdout. Start it from the TUI:\n"
            "  python3 tui/dsv4_tui.py\n", me);
}

int main(int argc, char **argv)
{
    const char *model = NULL, *trunkdir = NULL, *tokfile = NULL;
    double budget_gb = 8.0;
    int budget_auto = 0;
    int ctx = 8192, ngen = 128, think = 0, layers = 1;
    int prefill_batch = 0;
    const char *pin = "p", *reader_pin = "off";
    int threads = 4, readers = 6;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !model) { model = argv[i]; continue; }
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunkdir = argv[++i];
        else if (!strcmp(argv[i], "--tok") && i + 1 < argc) tokfile = argv[++i];
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "auto")) budget_auto = 1;
            else { budget_auto = 0; budget_gb = atof(v); }
        }
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) pin = argv[++i];
        else if (!strcmp(argv[i], "--readers") && i + 1 < argc) readers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reader-pin") && i + 1 < argc) reader_pin = argv[++i];
        else if (!strcmp(argv[i], "--stream-miss") && i + 1 < argc)
            g_stream_miss = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefill-batch-size") && i + 1 < argc)
            prefill_batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--think")) think = 1;
        else if (!strcmp(argv[i], "--no-layers")) layers = 0;
        else if (!strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
    }
    if (!model || !trunkdir || !tokfile) { usage(argv[0]); return 2; }

    struct sigaction cancel_sa;
    memset(&cancel_sa, 0, sizeof cancel_sa);
    cancel_sa.sa_handler = handle_cancel;
    sigemptyset(&cancel_sa.sa_mask);
    cancel_sa.sa_flags = SA_RESTART;
    if (sigaction(SIGUSR1, &cancel_sa, NULL) != 0)
        fprintf(stderr, "warning: cannot install cancel handler: %s\n",
                strerror(errno));

    /* Thread policy before engine_init(): that is where the expert cache
     * creates its reader pthreads, and they inherit the calling thread's mask. */
    const int pin_id = dsv4_threads_parse_pin(pin);
    const int rpin_id = dsv4_threads_parse_pin(reader_pin);
    if (pin_id < 0 || rpin_id < 0) {
        fprintf(stderr, "invalid --pin/--reader-pin (use off|all|p|e)\n");
        return 2;
    }
    const DSV4ThreadOpts topts = { pin_id, threads, readers, rpin_id };
    dsv4_threads_apply(&topts);

    /* Line buffering is not enough on a pipe: every event must be visible to the
     * TUI the moment it is produced. emit_* already fflush; keep stdout clean of
     * anything but JSON. */
    static char outbuf[1 << 16];
    setvbuf(stdout, outbuf, _IOFBF, sizeof outbuf);

    Engine e;
    if (engine_init(&e, model, trunkdir, tokfile, budget_gb, budget_auto, ctx,
                    ngen, think, layers, prefill_batch) != 0) {
        emit_error("engine init failed (see stderr for the loader's message)");
        engine_free(&e);
        return 1;
    }

    emit_ready(&e, "ready");

    char *line = NULL;
    size_t cap = 0;
    for (;;) {
        ssize_t n = getline(&line, &cap, stdin);
        if (n < 0) break;                       /* EOF: parent went away */
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r'))
            line[--n] = 0;
        if (n == 0) continue;

        jval *v = json_parse(line, NULL);
        if (!v || v->t != J_OBJ) {
            emit_error("invalid JSON command");
            jfree(v);
            continue;
        }
        const char *cmd = jstr(v, "cmd");
        const int quit = cmd && !strcmp(cmd, "quit");
        handle_command(&e, v);
        jfree(v);
        if (quit) break;
    }
    free(line);

    engine_free(&e);
    return 0;
}

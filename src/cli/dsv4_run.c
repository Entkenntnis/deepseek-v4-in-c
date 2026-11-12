/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_run.c - the dsv4 binary: load, prompt, decode, report.
 *
 * SCOPE, stated plainly rather than discovered later
 *   Greedy decoding only. No temperature, no top-p, no chat template. Greedy is
 *   what makes the output identical across memory budgets, which every gate in
 *   this repo depends on, and DeepSeek-V4-Flash ships no Jinja template anyway:
 *   the released encoding/ folder is the message-formatting path. So this
 *   produces base-model continuations, not replies.
 *
 *   No chunked prefill. The prompt is fed one token at a time through the same
 *   decode path the generated tokens use, which is correct and not fast.
 *
 * TWO ROPE TABLES, NOT ONE
 *   model.py Attention.__init__ builds freqs_cis with compress_rope_theta and
 *   YaRN enabled for any compress_ratio != 0, and with rope_theta and YaRN
 *   DISABLED for ratio 0. One table for the whole model is wrong on one set of
 *   layers or the other, silently.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "dsv4_cfg.h"
#include "dsv4_bind.h"
#include "dsv4_mem.h"
#include "dsv4_trunk.h"
#include "dsv4_cache.h"
#include "dsv4_layer.h"
#include "dsv4_tok.h"
#include "dsv4_threads.h"

void dsv4_rmsnorm(float *, const float *, const float *, int, float);
void dsv4_mmq(float *, const float *, const DSV4QMat *);
void dsv4_rope_table(float *, float *, int, int, int, double, double, double, double);
int  dsv4_tok_load(DSV4Tok *, const char *);
void dsv4_tok_free(DSV4Tok *);
int  dsv4_tok_encode(const DSV4Tok *, const char *, int, int32_t *, int);
/* Exact-string lookup, so the end-of-turn id is read from the tokenizer
 * rather than hardcoded to 1. */
int32_t dsv4_tok_find(const DSV4Tok *, const char *, int);

/* The expert source every layer is handed. The cache travels through ctx rather
 * than a file-scope global, so two models could run in one process. */
static const DSV4ExpertW *dsv4_run_get_expert(void *ctx, int layer, int e)
{
    return dsv4_cache_get((DSV4Cache *)ctx, layer, e);
}

static int dsv4_run_get_experts(void *ctx, int layer, const int *e, int n,
                                const DSV4ExpertW **out)
{
    return dsv4_cache_get_many((DSV4Cache *)ctx, layer, e, n, out);
}

/* The overlapped pair. The batch is file-scope because exactly one is in flight
 * at a time -- moe() calls begin then end with nothing in between that could
 * start another, and the layers run one after the other. Making it a parameter
 * would put a cache type in dsv4_layer.h for no gain. */
static DSV4CacheBatch g_batch;

static int dsv4_run_begin_experts(void *ctx, int layer, const int *e, int n,
                                  const DSV4ExpertW **out)
{
    return dsv4_cache_get_many_begin((DSV4Cache *)ctx, layer, e, n, out,
                                     &g_batch);
}

static int dsv4_run_end_experts(void *ctx, const DSV4ExpertW **out)
{
    return dsv4_cache_get_many_end((DSV4Cache *)ctx, out, &g_batch);
}

static double now_s(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double peak_rss_gb(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1048576.0;   /* Linux reports KB */
}

/* The head's own hc reduction. NOT Block.hc_pre: there is no Sinkhorn here and
 * no comb matrix, just a sigmoid gate. hc_scale is a single scalar applied to
 * every mix, where a block's is three. */
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

static void usage(const char *me)
{
    printf("usage: %s <model_dir | --model DIR> --trunk <dir> --tok <file> "
           "[--prompt TEXT] [--gen N] [--budget GB|auto] [--route-log FILE]\n"
           "       [--chat] [--think] [--system TEXT]\n"
           "       [--threads N] [--pin off|all|p|e] [--readers N]\n"
           "       [--reader-pin off|all|p|e]\n", me);
}

int main(int argc, char **argv)
{
    const char *model = NULL, *trunkdir = NULL, *tokfile = NULL;
    const char *routelog = NULL;
    int chat = 0, think = 0;
    const char *sysprompt = NULL;
    const char *prompt = "The capital of France is";
    int ngen = 8;
    double budget_gb = 8.0;
    int budget_auto = 0;
    const char *pin = "p", *reader_pin = "off";
    int threads = 4, readers = 6;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !model)      { model = argv[i]; continue; }
        /* --model is an alias for the positional form. Both are accepted
         * because "point it at a model" is the obvious mental model, and a
         * tool that only takes the directory positionally invites exactly the
         * wrong guess. */
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        if (!strcmp(argv[i], "--trunk")  && i + 1 < argc) trunkdir = argv[++i];
        else if (!strcmp(argv[i], "--tok")   && i + 1 < argc) tokfile = argv[++i];
        else if (!strcmp(argv[i], "--prompt")&& i + 1 < argc) prompt  = argv[++i];
        else if (!strcmp(argv[i], "--gen")   && i + 1 < argc) ngen    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--budget")&& i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "auto")) budget_auto = 1;
            else { budget_auto = 0; budget_gb = atof(v); }
        }
        else if (!strcmp(argv[i], "--route-log") && i + 1 < argc) routelog = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) pin = argv[++i];
        else if (!strcmp(argv[i], "--readers") && i + 1 < argc) readers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reader-pin") && i + 1 < argc) reader_pin = argv[++i];
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "--think")) { chat = 1; think = 1; }
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) { sysprompt = argv[++i]; chat = 1; }
        else if (!strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
    }
    if (!model || !trunkdir || !tokfile) { usage(argv[0]); return 2; }

    /* Thread policy first: no OpenMP region has run yet, and the expert cache
     * creates its reader pthreads later, so both inherit what this sets. */
    const int pin_id = dsv4_threads_parse_pin(pin);
    const int rpin_id = dsv4_threads_parse_pin(reader_pin);
    if (pin_id < 0 || rpin_id < 0) {
        fprintf(stderr, "invalid --pin/--reader-pin (use off|all|p|e)\n");
        return 2;
    }
    if (readers > 0) {
        char rb[16];
        snprintf(rb, sizeof rb, "%d", readers);
        setenv("DSV4_READERS", rb, 1);
    }
    const DSV4ThreadOpts topts = { pin_id, threads, readers, rpin_id };
    dsv4_threads_apply(&topts);

    char path[1024];
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    snprintf(path, sizeof path, "%s/config.json", model);
    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, path)) return 1;

    DSV4St st;
    if (dsv4_st_open(&st, model) != 0) return 1;

    DSV4Tok tok;
    if (dsv4_tok_load(&tok, tokfile) != 0) return 1;
    if ((int)tok.n_vocab != c.vocab) {
        fprintf(stderr, "tokenizer has %u ids, config says %d; refusing to run "
                        "a mismatched pair\n", tok.n_vocab, c.vocab);
        return 1;
    }

    int64_t budget = (int64_t)(budget_gb * 1073741824.0);

    /* `--budget auto`: take what the kernel says is available, keep 1 GiB for
     * the machine, and subtract the model-level weights.
     *
     * The last subtraction is the non-obvious one. `--budget` is the trunk plus
     * the expert cache; embed and head are bound separately by dsv4_bind_model
     * and are NOT part of it (about 1.97 GiB on Flash). Setting budget to
     * MemAvailable - 1 GiB without removing them would put peak RSS about a
     * gigabyte ABOVE what was available, which is how an "auto" that looks
     * generous becomes an OOM kill. Ask the binder for the exact blob size
     * rather than hardcoding 2x vocab x hidden: the checkpoint decides it. */
    if (budget_auto) {
        const int64_t avail = dsv4_mem_available_bytes();
        const int64_t model_bytes = dsv4_bind_model_bytes(&st, &c, 1);
        const int64_t reserve = (int64_t)1 << 30;       /* 1 GiB spare */
        /* Scratch, KV state, rope tables, tokenizer and allocator overhead.
         * The budget does not cover these; measured at about 0.34 GiB on
         * Flash. Keep half a gigabyte so the 1 GiB above is actually spare. */
        const int64_t runtime = (int64_t)512 << 20;
        if (avail <= 0) {
            fprintf(stderr, "--budget auto: cannot read available memory; "
                            "pass an explicit --budget GB\n");
            return 1;
        }
        if (model_bytes < 0) {
            fprintf(stderr, "--budget auto: cannot size the model's embed/head "
                            "weights; pass an explicit --budget GB\n");
            return 1;
        }
        budget = avail - reserve - model_bytes - runtime;
        if (budget <= 0) {
            fprintf(stderr, "--budget auto: %.2f GiB available, %.2f GiB model, "
                            "%.2f GiB reserve + runtime leaves nothing\n",
                    (double)avail / 1073741824.0,
                    (double)model_bytes / 1073741824.0,
                    (double)(reserve + runtime) / 1073741824.0);
            return 1;
        }
        fprintf(stderr, "--budget auto: %.2f GiB available - %.2f GiB model - "
                        "%.2f GiB reserve - %.2f GiB runtime = %.2f GiB budget\n",
                (double)avail / 1073741824.0, (double)model_bytes / 1073741824.0,
                (double)reserve / 1073741824.0,
                (double)runtime / 1073741824.0,
                (double)budget / 1073741824.0);
    }

    /* SPLIT THE BUDGET BY WHAT EACH PART CAN USE, NOT BY A FIXED RATIO.
     *
     * This used to hand the trunk 3/4 and the cache 1/4, reasoning that a
     * pinned layer is a guaranteed hit while an expert slot only pays off on a
     * re-route. True as far as it goes, and it produced a cache that could not
     * work: the trunk SATURATES. Once all 43 layers are pinned there is nothing
     * left to put in a ring slot, so trunk budget beyond ~6.4 GB buys exactly
     * nothing -- while the expert cache was held at 1/4 of 8 GB = 2.0 GB, below
     * the 3.29 GB one forward pass touches. A measured run: 2,580 expert
     * requests, ZERO hits.
     *
     * So: let the trunk take what it can use, and give the cache the rest. */
    /* RESERVE THE CACHE'S FLOOR BEFORE THE TRUNK TAKES ITS SHARE.
     *
     * The trunk is given what it can use and the cache gets the remainder --
     * right for Flash, whose 6.27 GB trunk pins completely and leaves plenty.
     * It inverts badly the moment the trunk is larger than RAM. Pro's is ~50
     * GB: handed the whole budget it would pin layers until the money ran out
     * and leave the cache with nothing, and since a layer's top-k experts must
     * be resident together, the engine would refuse to start at all.
     *
     * So the floor comes off the top. A layer needs topk experts at once, and
     * a little more than that is what makes the cache more than a staging
     * buffer -- but the hard minimum is what is enforced here. */
    const int64_t efloor = (int64_t)c.topk * dsv4_cache_slot_bytes(&c);
    if (budget <= efloor) {
        fprintf(stderr, "--budget %.2f GB cannot work: one layer's %d experts "
                        "alone need %.2f GB,\n  and the trunk needs room too.\n",
                (double)budget / 1073741824.0, c.topk,
                (double)efloor / 1073741824.0);
        return 1;
    }
    DSV4Trunk tr;
    if (dsv4_trunk_open(&tr, trunkdir, &c, budget - efloor) != 0) return 1;
    int64_t left = budget - dsv4_trunk_resident_bytes(&tr);
    DSV4Cache cache;
    if (dsv4_cache_init(&cache, &st, &c, left) != 0) return 1;
    if (routelog) {
        cache.route_log = fopen(routelog, "w");
        if (!cache.route_log) { perror(routelog); return 1; }
    }

    DSV4ModelBind mb;
    if (dsv4_bind_model(&st, &c, 1, &mb) != 0) return 1;

    /* Two tables: see the note at the top. */

    const int maxpos = c.sliding_window + 4096;
    float *cs_d = malloc((size_t)maxpos * (c.qk_rope / 2) * sizeof(float));
    float *sn_d = malloc((size_t)maxpos * (c.qk_rope / 2) * sizeof(float));
    float *cs_c = malloc((size_t)maxpos * (c.qk_rope / 2) * sizeof(float));
    float *sn_c = malloc((size_t)maxpos * (c.qk_rope / 2) * sizeof(float));
    dsv4_rope_table(cs_d, sn_d, c.qk_rope, maxpos, 0, c.rope_theta,
                    c.yarn_factor, c.yarn_beta_fast, c.yarn_beta_slow);
    dsv4_rope_table(cs_c, sn_c, c.qk_rope, maxpos, c.yarn_orig_ctx,
                    c.compress_rope_theta, c.yarn_factor,
                    c.yarn_beta_fast, c.yarn_beta_slow);

    DSV4Scratch scratch;
    if (dsv4_scratch_init(&scratch, &c, maxpos) != 0) return 1;

    DSV4LayerState *lst = calloc((size_t)c.n_layers, sizeof(DSV4LayerState));
    for (int L = 0; L < c.n_layers; L++)
        if (dsv4_state_init(&lst[L], &c, L, maxpos) != 0) return 1;

    DSV4ExpertSrc src = { dsv4_run_get_expert, dsv4_run_get_experts,
                          dsv4_run_begin_experts, dsv4_run_end_experts,
                          &cache };

    /* ---- chat encoding, per the checkpoint's own encoding/README.md ----
     *
     *   <BOS>{system}NL<User>{msg}<Assistant>{</think> or <think>}
     *
     * In chat mode </think> closes the reasoning block immediately so the model
     * answers directly; in thinking mode <think> opens it and it reasons first.
     * Without this the model never sees a turn boundary and simply continues
     * the text -- fluent, and not an answer.
     *
     * The special tokens must survive tokenisation atomically, which is why
     * dsv4_tok_encode matches added tokens before the pre-tokeniser runs. */
    static char chatbuf[65536];
    if (chat) {
        snprintf(chatbuf, sizeof chatbuf, "%s%s%s%s%s%s%s",
                 DSV4_BOS, sysprompt ? sysprompt : "", sysprompt ? "\n" : "",
                 DSV4_USER, prompt, DSV4_ASSISTANT,
                 think ? DSV4_THINK_OPEN : DSV4_THINK_CLOSE);
        prompt = chatbuf;
    }

    int32_t ids[8192];
    const int nprompt = dsv4_tok_encode(&tok, prompt, (int)strlen(prompt),
                                        ids, 8192);
    /* The end-of-turn id, looked up rather than hardcoded: it is 1 in this
     * checkpoint, and a checkpoint that numbered it differently would otherwise
     * never stop. */
    const int32_t eos_id = dsv4_tok_find(&tok, DSV4_EOS, (int)strlen(DSV4_EOS));
    printf("\nprompt: \"%s\"  -> %d tokens\n", prompt, nprompt);
    if (nprompt == 0) { fprintf(stderr, "empty prompt\n"); return 1; }

    const int hc = c.hc_mult, d = c.hidden;
    float *h = calloc((size_t)hc * d, sizeof(float));
    float *xh = calloc((size_t)d, sizeof(float));
    float *logits = calloc((size_t)c.vocab, sizeof(float));

    printf("--- generated ids ---\n");
    const double t0 = now_s();
    int ntok = 0;
    /* Time to first token, reported separately because it is a different
     * question from the decode rate and it is the one that decides whether the
     * engine is usable in an agent loop. */
    double ttft = 0.0;

    /* Prompt tokens and generated tokens use the SAME forward path. That keeps
     * prefill and decode timings comparable: there is no separate GEMM path
     * whose cost would have to be reasoned about separately. */
    for (int pos = 0; pos < nprompt + ngen; pos++) {
        const int32_t tid = ids[pos];

        /* embed, then expand to hc_mult identical copies (model.py Transformer:
         * h.unsqueeze(2).repeat(1,1,hc_mult,1)) */
        dsv4_embed_row(xh, mb.w.embed, mb.w.wdt, tid, d);
        for (int j = 0; j < hc; j++) memcpy(h + (size_t)j * d, xh, (size_t)d * sizeof(float));

        for (int L = 0; L < c.n_layers; L++) {
            DSV4LayerBind lb;
            const double tb = now_s();
            if (dsv4_trunk_bind(&tr, &c, L, &lb) != 0) return 1;
            dsv4_prof.bind += now_s() - tb;
            const int comp = dsv4_compress_ratio(&c, L) != 0;
            dsv4_layer_forward(h, &lb.w, &c, &scratch, &src, &lst[L],
                               comp ? cs_c : cs_d, comp ? sn_c : sn_d,
                               pos, tid);
        }

        if (pos + 1 >= nprompt) {
            /* The last prompt token has cleared every layer and the head has
             * not run yet. */
            if (ttft == 0.0 && pos + 1 == nprompt) ttft = now_s() - t0;
            /* The output head. Timed separately from the layers because it is
             * a vocab-wide BF16 GEMV (1.06 GB of weights on Flash) that lives
             * outside dsv4_layer_forward and used to land in "unaccounted". */
            const double th = now_s();
            hc_head(xh, h, &mb.w.hc_head, hc, d, c.rms_eps, c.hc_eps);
            dsv4_rmsnorm(xh, xh, mb.w.norm, d, c.rms_eps);
            DSV4QMat head = { mb.w.head, NULL, DSV4_WBF16, c.vocab, d, 0, 0 };
            dsv4_mmq(logits, xh, &head);
            dsv4_prof.head += now_s() - th;

            int best = 0;
            for (int v = 1; v < c.vocab; v++) if (logits[v] > logits[best]) best = v;
            if (pos + 1 < nprompt + ngen) ids[pos + 1] = best;
            printf(" %d", best);
            fflush(stdout);
            ntok++;
            /* STOP AT THE END OF THE TURN. Without this the model runs past its
             * own answer and keeps inventing conversation, which reads as a
             * broken engine rather than a missing stop condition. */
            if (eos_id >= 0 && best == eos_id) { printf("  <eos>"); break; }
        }
    }

    const double dt = now_s() - t0;

    printf("\n\n%d tokens in %.1f s, %.2f s/token\n", ntok, dt,
           ntok ? dt / ntok : 0.0);
    printf("time to first token: %.1f s over %d prompt tokens "
           "(%.2f s/prompt token)\n",
           ttft, nprompt, ttft / nprompt);
    dsv4_trunk_report(&tr, "run");
    dsv4_cache_report(&cache, "run");
    dsv4_prof_report(dt, ntok);
    printf("PEAK RSS for the whole run: %.2f GB\n", peak_rss_gb());

    for (int L = 0; L < c.n_layers; L++) dsv4_state_free(&lst[L]);
    free(lst); free(h); free(xh); free(logits);
    free(cs_d); free(sn_d); free(cs_c); free(sn_c);
    dsv4_scratch_free(&scratch);
    dsv4_bind_model_free(&mb);
    dsv4_cache_free(&cache);
    dsv4_trunk_close(&tr);
    dsv4_tok_free(&tok);
    dsv4_st_close(&st);
    return 0;
}

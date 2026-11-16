/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_engine_init.c - load the model once and keep it resident. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_cfg.h"
#include "dsv4_expert_vk.h"
#include "dsv4_kernels.h"
#include "dsv4_mem.h"
#include "dsv4_serve_int.h"

static void prefill_progress_cb(void *ctx, const DSV4PrefillProgress *p)
{
    dsv4_emit_prefill_progress((Engine *)ctx, p);
}

int dsv4_engine_init(Engine *e, const char *model, const char *trunkdir,
                     const char *tokfile, double budget_gb, int budget_auto,
                     int ctx, int gen_default, int think_default,
                     int layers_default)
{
    memset(e, 0, sizeof *e);
    e->model_dir = model;
    e->gen_default = gen_default;
    e->think_default = think_default;
    e->emit_layers_default = layers_default;
    e->t_start = dsv4_now_s();
    memset(&dsv4_prof, 0, sizeof dsv4_prof);
    e->prof_base = dsv4_prof;

    if (ctx < 512) ctx = 512;
    e->maxpos = ctx;
    e->cap_ids = ctx;

    char path[1024];
    snprintf(path, sizeof path, "%s/config.json", model);
    /* dsv4_cfg_load_file() prints a human-readable banner on stdout, which a
     * JSON-lines reader cannot tolerate. Send that one call's stdout to stderr,
     * then restore. */
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
         * The budget does not cover these; measured at about 0.34 GiB on Flash.
         * Keep half a gigabyte so the 1 GiB above is actually spare. */
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
    if (dsv4_trunk_open(&e->tr, trunkdir, &e->c, budget - efloor) != 0)
        return -1;
    const int64_t left = budget - dsv4_trunk_resident_bytes(&e->tr);
    if (dsv4_cache_init(&e->cache, &e->st, &e->c, left) != 0) return -1;
    const int stream_miss = dsv4_stream_miss();
    if (stream_miss > 0) {
        dsv4_cache_set_stream_cap(&e->cache, stream_miss);
        fprintf(stderr, "stream-miss: up to %d miss reads in flight; experts "
                        "are consumed as they land\n", e->cache.stream_cap);
    }
    dsv4_make_expert_src(&e->src, &e->cache);
    if (dsv4_expert_vk_init(&e->expert_vk, e->cache.vk, &e->c) != 0)
        return -1;
    e->src.gpu = e->expert_vk;
    if (dsv4_bind_model(&e->st, &e->c, 1, &e->mb) != 0) return -1;

    e->cs_d = malloc((size_t)e->maxpos * (e->c.qk_rope / 2) * sizeof(float));
    e->sn_d = malloc((size_t)e->maxpos * (e->c.qk_rope / 2) * sizeof(float));
    e->cs_c = malloc((size_t)e->maxpos * (e->c.qk_rope / 2) * sizeof(float));
    e->sn_c = malloc((size_t)e->maxpos * (e->c.qk_rope / 2) * sizeof(float));
    if (!e->cs_d || !e->sn_d || !e->cs_c || !e->sn_c) return -1;
    dsv4_rope_table(e->cs_d, e->sn_d, e->c.qk_rope, e->maxpos, 0,
                    e->c.rope_theta, e->c.yarn_factor,
                    e->c.yarn_beta_fast, e->c.yarn_beta_slow);
    dsv4_rope_table(e->cs_c, e->sn_c, e->c.qk_rope, e->maxpos,
                    e->c.yarn_orig_ctx, e->c.compress_rope_theta,
                    e->c.yarn_factor, e->c.yarn_beta_fast,
                    e->c.yarn_beta_slow);

    if (dsv4_scratch_init(&e->scratch, &e->c, e->maxpos) != 0) return -1;
    e->lst = calloc((size_t)e->c.n_layers, sizeof(DSV4LayerState));
    if (!e->lst) return -1;
    for (int L = 0; L < e->c.n_layers; L++)
        if (dsv4_state_init(&e->lst[L], &e->c, L, e->maxpos) != 0) return -1;

    {
        const DSV4PrefillOpts po = {
            .max_pos = e->maxpos,
            .on_progress = prefill_progress_cb, .progress_ctx = e,
        };
        if (dsv4_prefill_init(&e->pf, &e->c, &e->tr, e->lst, &e->src,
                              &e->mb.w, &po) != 0) {
            fprintf(stderr, "serve: block prefill init failed; falling back "
                            "to the per-token path\n");
            e->pf = NULL;
        } else {
            fprintf(stderr, "serve: whole-prompt block prefill ready\n");
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

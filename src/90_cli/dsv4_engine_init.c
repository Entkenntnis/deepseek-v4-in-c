/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_engine_init.c - load the model once and keep it resident. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_kernels.h"
#include "dsv4_util.h"
#include "dsv4_serve_int.h"

static void prefill_progress_cb(void *ctx, const DSV4PrefillProgress *p)
{
    dsv4_emit_prefill_progress((Engine *)ctx, p);
}

static int prefill_cancel_cb(void *ctx)
{
    (void)ctx;
    return dsv4_cancel != 0;
}

int dsv4_engine_init(Engine *e, const char *model, const char *trunkdir,
                     double budget_gb, int ctx, int gen_default)
{
    memset(e, 0, sizeof *e);
    e->model_dir = model;
    e->gen_default = gen_default;

    if (ctx < 512) ctx = 512;
    e->maxpos = ctx;
    e->cap_ids = ctx;

    dsv4_cfg_init(&e->c);
    fprintf(stderr, "config: DeepSeek-V4-Flash (compiled in) | hidden=%d "
                    "layers=%d vocab=%d\n",
            e->c.hidden, e->c.n_layers, e->c.vocab);

    if (dsv4_st_open(&e->st, model, trunkdir) != 0) return -1;

    const int budget_auto = budget_gb < 0.0;
    int64_t budget = (int64_t)(budget_gb * 1073741824.0);
    if (budget_auto) {
        const int64_t avail = dsv4_mem_available_bytes();
        const int64_t model_bytes = dsv4_bind_model_bytes(&e->st, &e->c, 1);
        const int64_t reserve = (int64_t)1 << 30;       /* 1 GiB spare */
        /* Scratch, KV state, rope tables and allocator overhead.
         * The budget does not cover these; the `--budget auto` output shows the
         * allowance. Keep enough spare that the reserve above is actually
         * spare. */
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
    if (dsv4_cache_init(&e->cache, &e->st, &e->tr, &e->c, left) != 0) return -1;
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

    /* The context is a few tens of MB, not a prompt-sized buffer: keeping the
     * rollback point resident lets a cancelled prefill restore it instead of
     * discarding the conversation. If it cannot be allocated, cancellation
     * still works, but falls back to a context reset. */
    e->snap = dsv4_state_snapshot_new(&e->c, e->lst);
    if (!e->snap)
        fprintf(stderr, "serve: no prefill rollback snapshot\n");

    {
        const DSV4PrefillOpts po = {
            .max_pos = e->maxpos,
            .on_progress = prefill_progress_cb, .progress_ctx = e,
            .is_cancelled = prefill_cancel_cb, .cancel_ctx = NULL,
        };
        if (dsv4_prefill_init(&e->pf, &e->c, &e->tr, e->lst, &e->cache,
                              &e->mb.w, &po) != 0) {
            fprintf(stderr, "serve: block prefill init failed\n");
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

    e->in_cap = e->cap_ids;
    e->in_ids = malloc((size_t)e->in_cap * sizeof(int32_t));
    if (!e->in_ids) return -1;
    return 0;
}

void dsv4_engine_free(Engine *e)
{
    dsv4_prefill_free(e->pf);
    dsv4_state_snapshot_free(e->snap);
    for (int L = 0; L < e->c.n_layers && e->lst; L++) dsv4_state_free(&e->lst[L]);
    free(e->lst);
    free(e->h); free(e->xh); free(e->logits); free(e->ids);
    free(e->cs_d); free(e->sn_d); free(e->cs_c); free(e->sn_c);
    free(e->in_ids);
    dsv4_scratch_free(&e->scratch);
    dsv4_bind_model_free(&e->mb);
    dsv4_cache_free(&e->cache);
    dsv4_trunk_close(&e->tr);
    dsv4_st_close(&e->st);
    memset(e, 0, sizeof *e);
}

int dsv4_engine_reset_context(Engine *e)
{
    dsv4_prefill_release(e->pf);
    for (int L = 0; L < e->c.n_layers; L++) {
        dsv4_state_free(&e->lst[L]);
        if (dsv4_state_init(&e->lst[L], &e->c, L, e->maxpos) != 0) return -1;
    }
    e->n_ids = 0;
    e->pending_forward = 0;
    e->kv_prefix = 0;
    return 0;
}

/* Range-check before appending so a bad command cannot leave half a turn. */
int dsv4_append_ids(Engine *e, const int32_t *ids, int n)
{
    if (n < 0) return -1;
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= e->c.vocab) return -1;
    if (n > e->cap_ids - e->n_ids) return -2;
    memcpy(e->ids + e->n_ids, ids, (size_t)n * sizeof(int32_t));
    e->n_ids += n;
    return n;
}

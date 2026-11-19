/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_kvfile.c - the prompt-prefix KV cache, version 2. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "dsv4_kvfile.h"

#define KV_MAGIC "DSV4KVC2"
#define KV_VERSION 2u
#define KV_TAG 64

static int write_all(FILE *f, const void *p, size_t n)
{
    return n == 0 || fwrite(p, 1, n, f) == n ? 0 : -1;
}

static int layer_ratio(const DSV4Cfg *c, int L)
{
    const int r = dsv4_compress_ratio(c, L);
    return r < 0 ? 0 : r;
}

static size_t layer_cn(const DSV4Cfg *c, int L)
{
    const int r = layer_ratio(c, L);
    if (!r) return 0;
    const int coff = r == 4 ? 2 : 1;
    return (size_t)coff * (size_t)r * (size_t)coff * (size_t)c->head_dim;
}

static int64_t payload_floats(const DSV4Cfg *c, const int *ncomp, int n_ids)
{
    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;
    int64_t n = 0;
    for (int L = 0; L < c->n_layers; L++) {
        const int r = layer_ratio(c, L);
        const int want = r ? n_ids / r : 0;
        if (ncomp[L] != want) return -1;
        n += (int64_t)wn + (int64_t)ncomp[L] * c->head_dim
           + 2 * (int64_t)layer_cn(c, L);
    }
    return n;
}

static void mkdir_parents(const char *path)
{
    char buf[1024];
    const size_t n = strlen(path);
    if (n >= sizeof buf) return;
    memcpy(buf, path, n + 1);
    char *slash = strrchr(buf, '/');
    if (!slash || slash == buf) return;
    *slash = 0;
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        mkdir(buf, 0755);
        *p = '/';
    }
    mkdir(buf, 0755);
}

uint64_t dsv4_cfg_hash(const DSV4Cfg *c)
{
    uint64_t h = 1469598103934665603ULL;
    const int32_t  i32[] = {
        c->hidden, c->n_layers, c->vocab, c->n_heads, c->n_kv_heads,
        c->head_dim, c->q_lora, c->o_lora, c->o_groups, c->qk_rope,
        c->sliding_window, c->index_n_heads, c->index_head_dim,
        c->index_topk, c->n_compress, c->hc_mult, c->hc_sinkhorn_iters,
        c->n_experts, c->topk, c->n_shared, c->moe_inter, c->norm_topk,
        c->scoring_func, c->topk_method, c->wblock_m, c->wblock_n,
        c->num_hash_layers, c->num_nextn_predict, c->max_pos,
        c->yarn_orig_ctx
    };
    const float f32[] = {
        c->rms_eps, c->compress_rope_theta, c->hc_eps, c->routed_scale,
        c->swiglu_limit, c->rope_theta, c->yarn_factor,
        c->yarn_beta_fast, c->yarn_beta_slow
    };
    for (size_t i = 0; i < sizeof i32 / sizeof i32[0]; i++)
        for (int b = 0; b < 4; b++) {
            h ^= (uint8_t)((uint32_t)i32[i] >> (8 * b));
            h *= 1099511628211ULL;
        }
    for (size_t i = 0; i < sizeof f32 / sizeof f32[0]; i++) {
        union { float f; uint32_t u; } v = { f32[i] };
        for (int b = 0; b < 4; b++) {
            h ^= (uint8_t)(v.u >> (8 * b));
            h *= 1099511628211ULL;
        }
    }
    for (int i = 0; i < c->n_compress; i++) {
        const uint32_t u = (uint32_t)c->compress_ratios[i];
        for (int b = 0; b < 4; b++) {
            h ^= (uint8_t)(u >> (8 * b));
            h *= 1099511628211ULL;
        }
    }
    return h;
}

int dsv4_kv_save(const char *path, const char *model_tag,
                 const DSV4Cfg *c, const DSV4LayerState *lst,
                 const int32_t *ids, int n_ids)
{
    if (!path || !c || !lst || !ids || n_ids <= 0 ||
        c->n_layers <= 0 || c->n_layers > DSV4_MAX_LAYERS)
        return -1;

    int ncomp[DSV4_MAX_LAYERS];
    for (int L = 0; L < c->n_layers; L++) ncomp[L] = lst[L].n_compressed;
    const int64_t floats = payload_floats(c, ncomp, n_ids);
    if (floats < 0) return -1;

    mkdir_parents(path);
    char tmp[1024];
    if (strlen(path) + 5 > sizeof tmp) return -1;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;

    char tag[KV_TAG] = { 0 };
    if (model_tag) snprintf(tag, sizeof tag, "%s", model_tag);
    const uint32_t version = KV_VERSION, reserved = 0;
    const uint64_t hash = dsv4_cfg_hash(c), fl = (uint64_t)floats;
    const int32_t nids = n_ids, nl = c->n_layers, mp = c->max_pos;
    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;

    int ok = 0;
    if (write_all(f, KV_MAGIC, 8) ||
        write_all(f, &version, 4) || write_all(f, &reserved, 4) ||
        write_all(f, &hash, 8) || write_all(f, &fl, 8) ||
        write_all(f, &nids, 4) || write_all(f, &nl, 4) ||
        write_all(f, &mp, 4) || write_all(f, tag, KV_TAG) ||
        write_all(f, ids, (size_t)n_ids * sizeof *ids) ||
        write_all(f, ncomp, (size_t)c->n_layers * sizeof *ncomp))
        goto done;

    for (int L = 0; L < c->n_layers; L++) {
        const DSV4LayerState *st = &lst[L];
        const int r = layer_ratio(c, L);
        if (!st->kv_cache) goto done;
        if (write_all(f, st->kv_cache, wn * sizeof(float))) goto done;
        if (r) {
            const size_t cn = layer_cn(c, L);
            const size_t rows = (size_t)st->n_compressed * c->head_dim;
            if (!st->comp_kv || !st->comp_score) goto done;
            if (write_all(f, st->kv_cache + wn, rows * sizeof(float)) ||
                write_all(f, st->comp_kv, cn * sizeof(float)) ||
                write_all(f, st->comp_score, cn * sizeof(float)))
                goto done;
        }
    }
    if (fflush(f) != 0 || fsync(fileno(f)) != 0) { /* fsync is advisory */ }
    if (fclose(f) != 0) { f = NULL; goto done; }
    f = NULL;
    if (rename(tmp, path) != 0) goto done;
    ok = 1;

done:
    if (f) fclose(f);
    if (!ok) {
        fprintf(stderr, "dsv4_kv: failed to write %s\n", path);
        remove(tmp);
    }
    return ok ? 0 : -1;
}

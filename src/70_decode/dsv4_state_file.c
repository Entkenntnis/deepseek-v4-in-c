/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_state_file.c - the prompt-prefix KV cache file.
 *
 * FORMAT (all integers and floats little-endian):
 *
 *   char     magic[8]        "DSV4KVC\0"
 *   uint32   version         1
 *   uint32   reserved        0
 *   uint64   cfg_hash        dsv4_cfg_hash()
 *   uint64   payload_hash    FNV-1a over the payload that follows
 *   uint64   payload_bytes
 *   int32    n_layers, hidden, head_dim, sliding_window, qk_rope, vocab
 *   int32    saved_max_pos, n_ids
 *   char     model_tag[128]
 *   int32    compress_ratios[n_layers]
 *   int32    n_compressed[n_layers]
 *   int32    ids[n_ids]
 *   per layer:
 *     float window[sliding_window * head_dim]
 *     float compressed[n_compressed * head_dim]      (ratio != 0)
 *     float comp_kv[cn]                              (ratio != 0)
 *     float comp_score[cn]                           (ratio != 0)
 *
 * WHY THE WHOLE WINDOW RING IS SAVED
 *   The sliding window is a ring indexed by absolute position. Saving only the
 *   "last N tokens" and rebuilding the ring would have to reproduce the exact
 *   modulo placement the model used. Copying the whole ring is both exact and
 *   simpler; it is ~11 MB for Flash.
 *
 * WHY COMPRESSED ROWS AND comp_* ARE SAVED
 *   The compressed rows are append-only and live after the ring in kv_cache.
 *   comp_kv/comp_score are the in-progress pooling window, including the
 *   -INFINITY score slots. Dropping either would make the first new token
 *   attend to a plausible but different past.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "dsv4_state_file.h"

#define KV_MAGIC       "DSV4KVC"
#define KV_VERSION     1u
#define KV_TAG_MAX     128

#define KV_FNV_OFFSET  1469598103934665603ULL
#define KV_FNV_PRIME   1099511628211ULL

static uint64_t fnv_u64(uint64_t h, uint64_t v)
{
    for (int i = 0; i < 8; i++) {
        h ^= (v >> (8 * i)) & 0xffu;
        h *= KV_FNV_PRIME;
    }
    return h;
}

static uint64_t fnv_i32(uint64_t h, int32_t v)
{
    return fnv_u64(h, (uint64_t)(uint32_t)v);
}

static uint64_t fnv_f32(uint64_t h, float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    return fnv_u64(h, (uint64_t)v.u);
}

static uint64_t fnv_bytes(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= KV_FNV_PRIME;
    }
    return h;
}

uint64_t dsv4_cfg_hash(const DSV4Cfg *c)
{
    uint64_t h = KV_FNV_OFFSET;
    if (!c) return h;

    h = fnv_i32(h, c->hidden);
    h = fnv_i32(h, c->n_layers);
    h = fnv_i32(h, c->vocab);
    h = fnv_f32(h, c->rms_eps);
    h = fnv_i32(h, c->max_pos);

    h = fnv_i32(h, c->n_heads);
    h = fnv_i32(h, c->n_kv_heads);
    h = fnv_i32(h, c->head_dim);
    h = fnv_i32(h, c->q_lora);
    h = fnv_i32(h, c->o_lora);
    h = fnv_i32(h, c->o_groups);
    h = fnv_i32(h, c->qk_rope);
    h = fnv_i32(h, c->sliding_window);

    h = fnv_i32(h, c->index_n_heads);
    h = fnv_i32(h, c->index_head_dim);
    h = fnv_i32(h, c->index_topk);

    h = fnv_i32(h, c->n_compress);
    for (int i = 0; i < c->n_compress && c->compress_ratios; i++)
        h = fnv_i32(h, c->compress_ratios[i]);
    h = fnv_f32(h, c->compress_rope_theta);

    h = fnv_i32(h, c->hc_mult);
    h = fnv_i32(h, c->hc_sinkhorn_iters);
    h = fnv_f32(h, c->hc_eps);

    h = fnv_i32(h, c->n_experts);
    h = fnv_i32(h, c->topk);
    h = fnv_i32(h, c->n_shared);
    h = fnv_i32(h, c->moe_inter);
    h = fnv_f32(h, c->routed_scale);
    h = fnv_i32(h, c->norm_topk);
    h = fnv_i32(h, c->scoring_func);
    h = fnv_i32(h, c->topk_method);

    h = fnv_f32(h, c->swiglu_limit);
    h = fnv_f32(h, c->rope_theta);
    h = fnv_f32(h, c->yarn_factor);
    h = fnv_f32(h, c->yarn_beta_fast);
    h = fnv_f32(h, c->yarn_beta_slow);
    h = fnv_i32(h, c->yarn_orig_ctx);
    h = fnv_i32(h, c->wblock_m);
    h = fnv_i32(h, c->wblock_n);
    h = fnv_i32(h, c->num_hash_layers);
    h = fnv_i32(h, c->num_nextn_predict);
    return h;
}

/* ---- little-endian field I/O --------------------------------------------- */

static int w_u32(FILE *f, uint32_t v, uint64_t *hash)
{
    unsigned char b[4];
    for (int i = 0; i < 4; i++) b[i] = (unsigned char)(v >> (8 * i));
    if (fwrite(b, 1, sizeof b, f) != sizeof b) return -1;
    if (hash) *hash = fnv_bytes(*hash, b, sizeof b);
    return 0;
}

static int w_u64(FILE *f, uint64_t v, uint64_t *hash)
{
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    if (fwrite(b, 1, sizeof b, f) != sizeof b) return -1;
    if (hash) *hash = fnv_bytes(*hash, b, sizeof b);
    return 0;
}

static int w_i32(FILE *f, int32_t v, uint64_t *hash)
{
    return w_u32(f, (uint32_t)v, hash);
}

static int w_f32(FILE *f, float val, uint64_t *hash)
{
    union { float f; uint32_t u; } v;
    v.f = val;
    return w_u32(f, v.u, hash);
}

static int r_u32(FILE *f, uint32_t *v, uint64_t *hash)
{
    unsigned char b[4];
    if (fread(b, 1, sizeof b, f) != sizeof b) return -1;
    if (hash) *hash = fnv_bytes(*hash, b, sizeof b);
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

static int r_u64(FILE *f, uint64_t *v, uint64_t *hash)
{
    unsigned char b[8];
    if (fread(b, 1, sizeof b, f) != sizeof b) return -1;
    if (hash) *hash = fnv_bytes(*hash, b, sizeof b);
    *v = 0;
    for (int i = 0; i < 8; i++) *v |= (uint64_t)b[i] << (8 * i);
    return 0;
}

static int r_i32(FILE *f, int32_t *v, uint64_t *hash)
{
    uint32_t u;
    if (r_u32(f, &u, hash) != 0) return -1;
    *v = (int32_t)u;
    return 0;
}

static int r_f32(FILE *f, float *fout, uint64_t *hash)
{
    uint32_t u;
    union { float f; uint32_t u; } v;
    if (r_u32(f, &u, hash) != 0) return -1;
    v.u = u;
    *fout = v.f;
    return 0;
}

/* ---- geometry helpers (must mirror dsv4_state_init) ---------------------- */

static int layer_ratio(const DSV4Cfg *c, int layer)
{
    int r = dsv4_compress_ratio(c, layer);
    return r < 0 ? 0 : r;
}

static int layer_coff(int ratio)
{
    return ratio == 4 ? 2 : 1;
}

static size_t layer_cn(const DSV4Cfg *c, int layer)
{
    const int ratio = layer_ratio(c, layer);
    if (!ratio) return 0;
    const int coff = layer_coff(ratio);
    return (size_t)coff * (size_t)ratio * (size_t)coff *
           (size_t)c->head_dim;
}

static size_t layer_window_floats(const DSV4Cfg *c)
{
    return (size_t)c->sliding_window * (size_t)c->head_dim;
}

/* ---- directory creation -------------------------------------------------- */

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
        if (mkdir(buf, 0755) != 0 && errno != EEXIST) { /* leave fopen to report */ }
        *p = '/';
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) { /* leave fopen to report */ }
}

/* ---- save ---------------------------------------------------------------- */

int dsv4_state_file_save(const char *path, const char *model_tag,
                         const DSV4Cfg *c, const DSV4LayerState *lst,
                         const int32_t *ids, int n_ids)
{
    if (!path || !c || !lst || !ids || n_ids <= 0) return -1;
    if (c->n_layers <= 0 || c->n_layers > DSV4_MAX_LAYERS) return -1;

    const size_t wn = layer_window_floats(c);
    uint64_t body = (uint64_t)c->n_layers * 4u * 2u + (uint64_t)n_ids * 4u;
    for (int L = 0; L < c->n_layers; L++) {
        const int ratio = layer_ratio(c, L);
        const int ncomp = lst[L].n_compressed;
        if (lst[L].n_compressed < 0) return -1;
        if (ratio == 0 && ncomp != 0) return -1;
        if (ratio > 0 && ncomp != n_ids / ratio) return -1;
        body += (uint64_t)(wn + (size_t)ncomp * (size_t)c->head_dim +
                          2u * layer_cn(c, L)) * 4u;
    }

    mkdir_parents(path);

    char *tmp = malloc(strlen(path) + 8);
    if (!tmp) return -1;
    sprintf(tmp, "%s.tmp", path);

    FILE *f = fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "dsv4_kv: cannot create %s: %s\n", tmp,
                strerror(errno));
        free(tmp);
        return -1;
    }

    int ok = 0;
    uint64_t h = KV_FNV_OFFSET;
    off_t hash_pos = 0;
    char tag[KV_TAG_MAX];
    memset(tag, 0, sizeof tag);
    if (model_tag) snprintf(tag, sizeof tag, "%s", model_tag);

    if (fwrite(KV_MAGIC, 1, 8, f) != 8) goto done;
    if (w_u32(f, KV_VERSION, NULL) != 0) goto done;
    if (w_u32(f, 0, NULL) != 0) goto done;
    if (w_u64(f, dsv4_cfg_hash(c), NULL) != 0) goto done;

    hash_pos = ftello(f);
    if (hash_pos < 0) goto done;
    if (w_u64(f, 0, NULL) != 0) goto done;          /* hash placeholder */
    if (w_u64(f, body, NULL) != 0) goto done;

    if (w_i32(f, c->n_layers, NULL) != 0 ||
        w_i32(f, c->hidden, NULL) != 0 ||
        w_i32(f, c->head_dim, NULL) != 0 ||
        w_i32(f, c->sliding_window, NULL) != 0 ||
        w_i32(f, c->qk_rope, NULL) != 0 ||
        w_i32(f, c->vocab, NULL) != 0 ||
        w_i32(f, c->max_pos, NULL) != 0 ||
        w_i32(f, n_ids, NULL) != 0) goto done;
    if (fwrite(tag, 1, sizeof tag, f) != sizeof tag) goto done;

    for (int L = 0; L < c->n_layers; L++)
        if (w_i32(f, layer_ratio(c, L), &h) != 0) goto done;
    for (int L = 0; L < c->n_layers; L++)
        if (w_i32(f, lst[L].n_compressed, &h) != 0) goto done;
    for (int i = 0; i < n_ids; i++)
        if (w_i32(f, ids[i], &h) != 0) goto done;

    for (int L = 0; L < c->n_layers; L++) {
        const DSV4LayerState *st = &lst[L];
        const int ratio = layer_ratio(c, L);
        if (!st->kv_cache) goto done;
        for (size_t i = 0; i < wn; i++)
            if (w_f32(f, st->kv_cache[i], &h) != 0) goto done;
        if (ratio) {
            const size_t ncomp = (size_t)st->n_compressed;
            const size_t cn = layer_cn(c, L);
            if (!st->comp_kv || !st->comp_score) goto done;
            for (size_t i = 0; i < ncomp * (size_t)c->head_dim; i++)
                if (w_f32(f, st->kv_cache[wn + i], &h) != 0) goto done;
            for (size_t i = 0; i < cn; i++)
                if (w_f32(f, st->comp_kv[i], &h) != 0) goto done;
            for (size_t i = 0; i < cn; i++)
                if (w_f32(f, st->comp_score[i], &h) != 0) goto done;
        }
    }

    if (fflush(f) != 0) goto done;
    if (hash_pos >= 0) {
        if (fseeko(f, hash_pos, SEEK_SET) != 0) goto done;
        if (w_u64(f, h, NULL) != 0) goto done;
        if (fseeko(f, 0, SEEK_END) != 0) goto done;
    }
    if (fsync(fileno(f)) != 0) { /* not fatal on filesystems that reject it */ }
    if (fclose(f) != 0) { f = NULL; goto done; }
    f = NULL;

    if (rename(tmp, path) != 0) {
        fprintf(stderr, "dsv4_kv: cannot rename %s to %s: %s\n", tmp, path,
                strerror(errno));
        goto done;
    }
    ok = 1;

done:
    if (f) fclose(f);
    if (!ok) {
        fprintf(stderr, "dsv4_kv: failed to write %s\n", path);
        remove(tmp);
    }
    free(tmp);
    return ok ? 0 : -1;
}

/* ---- load ---------------------------------------------------------------- */

int dsv4_state_file_load(const char *path, const char *model_tag,
                         const DSV4Cfg *c, DSV4LayerState *lst, int max_pos,
                         int32_t *ids, int ids_cap, int *n_ids_out,
                         const int32_t *expect_ids, int n_expect)
{
    if (!path || !c || !lst || !ids || !n_ids_out || max_pos <= 0) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "dsv4_kv: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    int ok = 0;
    int32_t *file_ids = NULL;
    int *ncomp = NULL;
    float *buf = NULL;
    uint64_t h = KV_FNV_OFFSET;
    off_t body_start = 0;
    struct stat sb;

    char magic[8];
    uint32_t version = 0, reserved = 0;
    uint64_t cfg_hash = 0, payload_hash = 0, payload_bytes = 0;
    int32_t n_layers = 0, hidden = 0, head_dim = 0, sliding_window = 0;
    int32_t qk_rope = 0, vocab = 0, saved_max_pos = 0, n_ids = 0;
    char tag[KV_TAG_MAX];

    if (fread(magic, 1, sizeof magic, f) != sizeof magic ||
        memcmp(magic, KV_MAGIC, 8) != 0) {
        fprintf(stderr, "dsv4_kv: %s is not a DSV4 KV cache\n", path);
        goto done;
    }
    if (r_u32(f, &version, NULL) != 0 || r_u32(f, &reserved, NULL) != 0 ||
        r_u64(f, &cfg_hash, NULL) != 0 || r_u64(f, &payload_hash, NULL) != 0 ||
        r_u64(f, &payload_bytes, NULL) != 0) goto bad;
    if (version != KV_VERSION) {
        fprintf(stderr, "dsv4_kv: %s has version %u, expected %u\n", path,
                version, KV_VERSION);
        goto done;
    }
    (void)reserved;

    if (r_i32(f, &n_layers, NULL) != 0 || r_i32(f, &hidden, NULL) != 0 ||
        r_i32(f, &head_dim, NULL) != 0 ||
        r_i32(f, &sliding_window, NULL) != 0 ||
        r_i32(f, &qk_rope, NULL) != 0 || r_i32(f, &vocab, NULL) != 0 ||
        r_i32(f, &saved_max_pos, NULL) != 0 || r_i32(f, &n_ids, NULL) != 0)
        goto bad;
    if (fread(tag, 1, sizeof tag, f) != sizeof tag) goto bad;
    tag[sizeof tag - 1] = 0;

    if (n_layers != c->n_layers || hidden != c->hidden ||
        head_dim != c->head_dim || sliding_window != c->sliding_window ||
        qk_rope != c->qk_rope || vocab != c->vocab) {
        fprintf(stderr, "dsv4_kv: %s was built for a different model geometry\n",
                path);
        goto done;
    }
    if (cfg_hash != dsv4_cfg_hash(c)) {
        fprintf(stderr, "dsv4_kv: %s does not match this config\n", path);
        goto done;
    }
    if (model_tag && model_tag[0] && tag[0] && strcmp(tag, model_tag) != 0) {
        fprintf(stderr, "dsv4_kv: %s belongs to model \"%s\", not \"%s\"\n",
                path, tag, model_tag);
        goto done;
    }
    if (n_ids <= 0) {
        fprintf(stderr, "dsv4_kv: %s holds no prefix tokens\n", path);
        goto done;
    }
    if (n_ids > max_pos || n_ids > ids_cap) {
        fprintf(stderr, "dsv4_kv: %s holds %d tokens, context cap is %d\n",
                path, n_ids, max_pos);
        goto done;
    }
    if (saved_max_pos <= 0 || n_ids > saved_max_pos) {
        fprintf(stderr, "dsv4_kv: %s has an invalid saved context size\n", path);
        goto done;
    }
    if (expect_ids && n_expect != n_ids) {
        fprintf(stderr, "dsv4_kv: %s has %d prefix tokens, text has %d\n",
                path, n_ids, n_expect);
        goto done;
    }

    ncomp = (int *)calloc((size_t)n_layers, sizeof *ncomp);
    file_ids = (int32_t *)malloc((size_t)n_ids * sizeof *file_ids);
    if (!ncomp || !file_ids) goto oom;

    for (int L = 0; L < n_layers; L++) {
        int32_t got = 0;
        if (r_i32(f, &got, &h) != 0) goto bad;
        if (got != layer_ratio(c, L)) {
            fprintf(stderr, "dsv4_kv: %s layer %d compress ratio is %d, "
                            "expected %d\n", path, L, got, layer_ratio(c, L));
            goto done;
        }
    }
    for (int L = 0; L < n_layers; L++) {
        if (r_i32(f, &ncomp[L], &h) != 0) goto bad;
        const int ratio = layer_ratio(c, L);
        const int want = ratio ? n_ids / ratio : 0;
        if (ncomp[L] != want) {
            fprintf(stderr, "dsv4_kv: %s layer %d has %d compressed rows, "
                            "expected %d\n", path, L, ncomp[L], want);
            goto done;
        }
    }
    for (int i = 0; i < n_ids; i++)
        if (r_i32(f, &file_ids[i], &h) != 0) goto bad;
    if (expect_ids && memcmp(file_ids, expect_ids,
                             (size_t)n_ids * sizeof *file_ids) != 0) {
        fprintf(stderr, "dsv4_kv: %s does not hold the expected prefix\n", path);
        goto done;
    }
    for (int i = 0; i < n_ids; i++) {
        if (file_ids[i] < 0 || file_ids[i] >= c->vocab) {
            fprintf(stderr, "dsv4_kv: %s has out-of-vocab token %d\n", path,
                    file_ids[i]);
            goto done;
        }
    }

    const size_t wn = layer_window_floats(c);
    uint64_t floats = 0;
    for (int L = 0; L < n_layers; L++) {
        const int ratio = layer_ratio(c, L);
        const size_t rows = (size_t)ncomp[L] * (size_t)c->head_dim;
        floats += (uint64_t)wn + (uint64_t)rows + 2u * layer_cn(c, L);
        if (ratio == 0 && ncomp[L] != 0) goto bad;
        if (ratio > 0 && ncomp[L] > saved_max_pos / ratio + 1) goto bad;
        if (ratio > 0 && ncomp[L] > max_pos / ratio + 1) goto bad;
    }
    if (payload_bytes != (uint64_t)n_layers * 8u +
                         (uint64_t)n_ids * 4u + floats * 4u) {
        fprintf(stderr, "dsv4_kv: %s payload size does not match its header\n",
                path);
        goto done;
    }
    if (payload_bytes > (uint64_t)SIZE_MAX) goto oom;

    body_start = ftello(f);
    if (body_start < 0) goto bad;
    if (fstat(fileno(f), &sb) != 0) goto bad;
    if ((uint64_t)body_start + floats * 4u != (uint64_t)sb.st_size) {
        fprintf(stderr, "dsv4_kv: %s is truncated or has trailing data\n", path);
        goto done;
    }

    buf = (float *)malloc((size_t)floats * sizeof *buf);
    if (!buf) goto oom;
    for (uint64_t i = 0; i < floats; i++)
        if (r_f32(f, &buf[i], &h) != 0) goto bad;
    if (h != payload_hash) {
        fprintf(stderr, "dsv4_kv: %s checksum mismatch\n", path);
        goto done;
    }

    /* Validate every destination pointer before the first memcpy. A commit
     * must be all-or-nothing even if a caller hands us an uninitialised state. */
    for (int L = 0; L < n_layers; L++) {
        const int ratio = layer_ratio(c, L);
        if (!lst[L].kv_cache) goto bad;
        if (ratio && (!lst[L].comp_kv || !lst[L].comp_score)) goto bad;
    }

    /* All validation passed. Commit with memcpy only, so this cannot fail
     * halfway through. */
    size_t at = 0;
    for (int L = 0; L < n_layers; L++) {
        DSV4LayerState *st = &lst[L];
        const int ratio = layer_ratio(c, L);
        const size_t cn = layer_cn(c, L);
        memcpy(st->kv_cache, buf + at, wn * sizeof(float));
        at += wn;
        if (ratio) {
            const size_t rows = (size_t)ncomp[L] * (size_t)c->head_dim;
            memcpy(st->kv_cache + wn, buf + at, rows * sizeof(float));
            at += rows;
            memcpy(st->comp_kv, buf + at, cn * sizeof(float));
            at += cn;
            memcpy(st->comp_score, buf + at, cn * sizeof(float));
            at += cn;
            st->n_compressed = ncomp[L];
        } else {
            st->n_compressed = 0;
        }
    }
    memcpy(ids, file_ids, (size_t)n_ids * sizeof *ids);
    *n_ids_out = n_ids;
    ok = 1;
    goto done;

bad:
    fprintf(stderr, "dsv4_kv: %s is malformed\n", path);
    goto done;
oom:
    fprintf(stderr, "dsv4_kv: out of memory loading %s\n", path);

done:
    free(file_ids);
    free(ncomp);
    free(buf);
    fclose(f);
    return ok ? 0 : -1;
}

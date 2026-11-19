/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_state_load.c - validate and read a prompt-prefix KV cache. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "dsv4_state_file.h"
#include "dsv4_state_file_int.h"

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
    if (kv_r_u32(f, &version, NULL) != 0 || kv_r_u32(f, &reserved, NULL) != 0 ||
        kv_r_u64(f, &cfg_hash, NULL) != 0 || kv_r_u64(f, &payload_hash, NULL) != 0 ||
        kv_r_u64(f, &payload_bytes, NULL) != 0) goto bad;
    if (version != KV_VERSION) {
        fprintf(stderr, "dsv4_kv: %s has version %u, expected %u\n", path,
                version, KV_VERSION);
        goto done;
    }
    (void)reserved;

    if (kv_r_i32(f, &n_layers, NULL) != 0 || kv_r_i32(f, &hidden, NULL) != 0 ||
        kv_r_i32(f, &head_dim, NULL) != 0 ||
        kv_r_i32(f, &sliding_window, NULL) != 0 ||
        kv_r_i32(f, &qk_rope, NULL) != 0 || kv_r_i32(f, &vocab, NULL) != 0 ||
        kv_r_i32(f, &saved_max_pos, NULL) != 0 || kv_r_i32(f, &n_ids, NULL) != 0)
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
        if (kv_r_i32(f, &got, &h) != 0) goto bad;
        if (got != kv_layer_ratio(c, L)) {
            fprintf(stderr, "dsv4_kv: %s layer %d compress ratio is %d, "
                            "expected %d\n", path, L, got, kv_layer_ratio(c, L));
            goto done;
        }
    }
    for (int L = 0; L < n_layers; L++) {
        if (kv_r_i32(f, &ncomp[L], &h) != 0) goto bad;
        const int ratio = kv_layer_ratio(c, L);
        const int want = ratio ? n_ids / ratio : 0;
        if (ncomp[L] != want) {
            fprintf(stderr, "dsv4_kv: %s layer %d has %d compressed rows, "
                            "expected %d\n", path, L, ncomp[L], want);
            goto done;
        }
    }
    for (int i = 0; i < n_ids; i++)
        if (kv_r_i32(f, &file_ids[i], &h) != 0) goto bad;
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

    const size_t wn = kv_layer_window_floats(c);
    uint64_t floats = 0;
    for (int L = 0; L < n_layers; L++) {
        const int ratio = kv_layer_ratio(c, L);
        const size_t rows = (size_t)ncomp[L] * (size_t)c->head_dim;
        floats += (uint64_t)wn + (uint64_t)rows + 2u * kv_layer_cn(c, L);
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
        if (kv_r_f32(f, &buf[i], &h) != 0) goto bad;
    if (h != payload_hash) {
        fprintf(stderr, "dsv4_kv: %s checksum mismatch\n", path);
        goto done;
    }

    if (kv_commit(c, lst, ids, file_ids, n_ids, ncomp, buf) != 0) goto bad;
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

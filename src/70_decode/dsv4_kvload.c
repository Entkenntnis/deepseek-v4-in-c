/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_kvload.c - validate and commit a prompt-prefix KV cache. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_kvfile.h"

static int read_all(FILE *f, void *p, size_t n)
{
    return n == 0 || fread(p, 1, n, f) == n ? 0 : -1;
}

int dsv4_kv_load(const char *path, const char *model_tag,
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

    char magic[8], tag[DSV4_KV_TAG];
    uint32_t version = 0, reserved = 0;
    uint64_t hash = 0, fl = 0;
    int32_t n_ids = 0, nl = 0, saved_mp = 0;
    int *ncomp = NULL;
    float *buf = NULL;
    int32_t *file_ids = NULL;
    int ok = 0;

    if (read_all(f, magic, 8) || memcmp(magic, DSV4_KV_MAGIC, 8) != 0 ||
        read_all(f, &version, 4) || version != DSV4_KV_VERSION ||
        read_all(f, &reserved, 4) || read_all(f, &hash, 8) ||
        read_all(f, &fl, 8) || read_all(f, &n_ids, 4) ||
        read_all(f, &nl, 4) || read_all(f, &saved_mp, 4) ||
        read_all(f, tag, DSV4_KV_TAG)) {
        fprintf(stderr, "dsv4_kv: %s is not a version-2 cache\n", path);
        goto done;
    }
    (void)reserved;
    tag[DSV4_KV_TAG - 1] = 0;
    if (nl != c->n_layers || hash != dsv4_cfg_hash(c) ||
        saved_mp <= 0 || n_ids <= 0 ||
        n_ids > max_pos || n_ids > ids_cap) {
        fprintf(stderr, "dsv4_kv: %s was built for a different model or "
                        "context\n", path);
        goto done;
    }
    if (model_tag && model_tag[0] && tag[0] && strcmp(tag, model_tag) != 0) {
        fprintf(stderr, "dsv4_kv: %s belongs to model \"%s\", not \"%s\"\n",
                path, tag, model_tag);
        goto done;
    }
    if (expect_ids && n_expect != n_ids) {
        fprintf(stderr, "dsv4_kv: %s has %d tokens, expected %d\n",
                path, n_ids, n_expect);
        goto done;
    }

    ncomp = (int *)calloc((size_t)nl, sizeof *ncomp);
    file_ids = (int32_t *)malloc((size_t)n_ids * sizeof *file_ids);
    if (!ncomp || !file_ids) goto done;
    if (read_all(f, file_ids, (size_t)n_ids * sizeof *file_ids) ||
        read_all(f, ncomp, (size_t)nl * sizeof *ncomp)) {
        fprintf(stderr, "dsv4_kv: %s is truncated\n", path);
        goto done;
    }
    if (expect_ids && memcmp(file_ids, expect_ids,
                             (size_t)n_ids * sizeof *file_ids) != 0) {
        fprintf(stderr, "dsv4_kv: %s does not hold the expected prefix\n", path);
        goto done;
    }
    for (int i = 0; i < n_ids; i++)
        if (file_ids[i] < 0 || file_ids[i] >= c->vocab) goto done;

    const int64_t floats = dsv4_kv_payload_floats(c, ncomp, n_ids);
    if (floats < 0 || (uint64_t)floats != fl) goto done;

    /* Validate every destination before touching any of them. */
    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;
    for (int L = 0; L < nl; L++) {
        if (!lst[L].kv_cache) goto done;
        if (dsv4_kv_layer_ratio(c, L) &&
            (!lst[L].comp_kv || !lst[L].comp_score)) goto done;
    }

    buf = (float *)malloc((size_t)floats * sizeof *buf);
    if (!buf || read_all(f, buf, (size_t)floats * sizeof *buf)) {
        fprintf(stderr, "dsv4_kv: %s is truncated\n", path);
        goto done;
    }

    int64_t at = 0;
    for (int L = 0; L < nl; L++) {
        DSV4LayerState *st = &lst[L];
        const int r = dsv4_kv_layer_ratio(c, L);
        memcpy(st->kv_cache, buf + at, wn * sizeof(float));
        at += (int64_t)wn;
        if (r) {
            const size_t cn = dsv4_kv_layer_cn(c, L);
            const size_t rows = (size_t)ncomp[L] * c->head_dim;
            memcpy(st->kv_cache + wn, buf + at, rows * sizeof(float));
            at += (int64_t)rows;
            memcpy(st->comp_kv, buf + at, cn * sizeof(float));
            at += (int64_t)cn;
            memcpy(st->comp_score, buf + at, cn * sizeof(float));
            at += (int64_t)cn;
        }
        st->n_compressed = ncomp[L];
    }
    memcpy(ids, file_ids, (size_t)n_ids * sizeof *ids);
    *n_ids_out = n_ids;
    ok = 1;

done:
    free(ncomp);
    free(file_ids);
    free(buf);
    fclose(f);
    return ok ? 0 : -1;
}

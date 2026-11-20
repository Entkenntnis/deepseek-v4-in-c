/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_kvload.c - read and commit the minimal prompt-prefix KV cache.
 *
 * The file is trusted: an int32 token count, that many int32 token ids, then
 * the float payload. There is no header, config hash, model tag or geometry
 * check. Only I/O/allocation failures and the caller's id-buffer capacity are
 * checked.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_kvfile.h"

static int read_all(FILE *f, void *p, size_t n)
{
    return n == 0 || fread(p, 1, n, f) == n ? 0 : -1;
}

int dsv4_kv_load(const char *path, const DSV4Cfg *c, DSV4LayerState *lst,
                 int32_t *ids, int ids_cap, int *n_ids_out)
{
    if (!path || !c || !lst || !ids || !n_ids_out) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "dsv4_kv: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    int32_t n_ids = 0;
    int32_t *file_ids = NULL;
    float *buf = NULL;
    int ok = 0;

    if (read_all(f, &n_ids, sizeof n_ids)) {
        fprintf(stderr, "dsv4_kv: %s is truncated\n", path);
        goto done;
    }
    if (n_ids > ids_cap) {
        fprintf(stderr, "dsv4_kv: %s has more ids than fit in the destination\n",
                path);
        goto done;
    }

    file_ids = (int32_t *)malloc((size_t)n_ids * sizeof *file_ids);
    const int64_t floats = dsv4_kv_payload_floats(c, n_ids);
    buf = (float *)malloc((size_t)floats * sizeof *buf);
    if (!file_ids || !buf) goto done;
    if (read_all(f, file_ids, (size_t)n_ids * sizeof *file_ids) ||
        read_all(f, buf, (size_t)floats * sizeof *buf)) {
        fprintf(stderr, "dsv4_kv: %s is truncated\n", path);
        goto done;
    }

    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;
    int64_t at = 0;
    for (int L = 0; L < c->n_layers; L++) {
        DSV4LayerState *st = &lst[L];
        const int r = dsv4_kv_layer_ratio(c, L);
        const int ncomp = r ? n_ids / r : 0;
        memcpy(st->kv_cache, buf + at, wn * sizeof(float));
        at += (int64_t)wn;
        if (r) {
            const size_t cn = dsv4_kv_layer_cn(c, L);
            const size_t rows = (size_t)ncomp * c->head_dim;
            memcpy(st->kv_cache + wn, buf + at, rows * sizeof(float));
            at += (int64_t)rows;
            memcpy(st->comp_kv, buf + at, cn * sizeof(float));
            at += (int64_t)cn;
            memcpy(st->comp_score, buf + at, cn * sizeof(float));
            at += (int64_t)cn;
        }
        st->n_compressed = ncomp;
    }
    memcpy(ids, file_ids, (size_t)n_ids * sizeof *ids);
    *n_ids_out = n_ids;
    ok = 1;

done:
    free(file_ids);
    free(buf);
    fclose(f);
    return ok ? 0 : -1;
}

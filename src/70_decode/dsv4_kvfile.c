/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_kvfile.c - write the minimal prompt-prefix KV cache. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "dsv4_kvfile.h"

static int write_all(FILE *f, const void *p, size_t n)
{
    return n == 0 || fwrite(p, 1, n, f) == n ? 0 : -1;
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

int dsv4_kv_save(const char *path, const DSV4Cfg *c, const DSV4LayerState *lst,
                 const int32_t *ids, int n_ids)
{
    if (!path || !c || !lst || !ids || n_ids <= 0 ||
        c->n_layers <= 0 || c->n_layers > DSV4_MAX_LAYERS)
        return -1;

    mkdir_parents(path);
    char tmp[1024];
    if (strlen(path) + 5 > sizeof tmp) return -1;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;

    const int32_t nids = n_ids;
    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;
    int ok = 0;

    if (write_all(f, &nids, sizeof nids) ||
        write_all(f, ids, (size_t)n_ids * sizeof *ids))
        goto done;

    for (int L = 0; L < c->n_layers; L++) {
        const DSV4LayerState *st = &lst[L];
        const int r = dsv4_kv_layer_ratio(c, L);
        if (!st->kv_cache) goto done;
        if (write_all(f, st->kv_cache, wn * sizeof(float))) goto done;
        if (r) {
            const size_t cn = dsv4_kv_layer_cn(c, L);
            const size_t rows = (size_t)(n_ids / r) * c->head_dim;
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

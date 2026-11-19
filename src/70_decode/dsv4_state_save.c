/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_state_save.c - atomic write of a prompt-prefix KV cache. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "dsv4_state_file.h"
#include "dsv4_state_file_int.h"

/* ---- save ---------------------------------------------------------------- */

int dsv4_state_file_save(const char *path, const char *model_tag,
                         const DSV4Cfg *c, const DSV4LayerState *lst,
                         const int32_t *ids, int n_ids)
{
    if (!path || !c || !lst || !ids || n_ids <= 0) return -1;
    if (c->n_layers <= 0 || c->n_layers > DSV4_MAX_LAYERS) return -1;

    const size_t wn = kv_layer_window_floats(c);
    uint64_t body = (uint64_t)c->n_layers * 4u * 2u + (uint64_t)n_ids * 4u;
    for (int L = 0; L < c->n_layers; L++) {
        const int ratio = kv_layer_ratio(c, L);
        const int ncomp = lst[L].n_compressed;
        if (lst[L].n_compressed < 0) return -1;
        if (ratio == 0 && ncomp != 0) return -1;
        if (ratio > 0 && ncomp != n_ids / ratio) return -1;
        body += (uint64_t)(wn + (size_t)ncomp * (size_t)c->head_dim +
                          2u * kv_layer_cn(c, L)) * 4u;
    }

    kv_mkdir_parents(path);

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
    if (kv_w_u32(f, KV_VERSION, NULL) != 0) goto done;
    if (kv_w_u32(f, 0, NULL) != 0) goto done;
    if (kv_w_u64(f, dsv4_cfg_hash(c), NULL) != 0) goto done;

    hash_pos = ftello(f);
    if (hash_pos < 0) goto done;
    if (kv_w_u64(f, 0, NULL) != 0) goto done;          /* hash placeholder */
    if (kv_w_u64(f, body, NULL) != 0) goto done;

    if (kv_w_i32(f, c->n_layers, NULL) != 0 ||
        kv_w_i32(f, c->hidden, NULL) != 0 ||
        kv_w_i32(f, c->head_dim, NULL) != 0 ||
        kv_w_i32(f, c->sliding_window, NULL) != 0 ||
        kv_w_i32(f, c->qk_rope, NULL) != 0 ||
        kv_w_i32(f, c->vocab, NULL) != 0 ||
        kv_w_i32(f, c->max_pos, NULL) != 0 ||
        kv_w_i32(f, n_ids, NULL) != 0) goto done;
    if (fwrite(tag, 1, sizeof tag, f) != sizeof tag) goto done;

    for (int L = 0; L < c->n_layers; L++)
        if (kv_w_i32(f, kv_layer_ratio(c, L), &h) != 0) goto done;
    for (int L = 0; L < c->n_layers; L++)
        if (kv_w_i32(f, lst[L].n_compressed, &h) != 0) goto done;
    for (int i = 0; i < n_ids; i++)
        if (kv_w_i32(f, ids[i], &h) != 0) goto done;

    for (int L = 0; L < c->n_layers; L++) {
        const DSV4LayerState *st = &lst[L];
        const int ratio = kv_layer_ratio(c, L);
        if (!st->kv_cache) goto done;
        for (size_t i = 0; i < wn; i++)
            if (kv_w_f32(f, st->kv_cache[i], &h) != 0) goto done;
        if (ratio) {
            const size_t ncomp = (size_t)st->n_compressed;
            const size_t cn = kv_layer_cn(c, L);
            if (!st->comp_kv || !st->comp_score) goto done;
            for (size_t i = 0; i < ncomp * (size_t)c->head_dim; i++)
                if (kv_w_f32(f, st->kv_cache[wn + i], &h) != 0) goto done;
            for (size_t i = 0; i < cn; i++)
                if (kv_w_f32(f, st->comp_kv[i], &h) != 0) goto done;
            for (size_t i = 0; i < cn; i++)
                if (kv_w_f32(f, st->comp_score[i], &h) != 0) goto done;
        }
    }

    if (fflush(f) != 0) goto done;
    if (hash_pos >= 0) {
        if (fseeko(f, hash_pos, SEEK_SET) != 0) goto done;
        if (kv_w_u64(f, h, NULL) != 0) goto done;
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

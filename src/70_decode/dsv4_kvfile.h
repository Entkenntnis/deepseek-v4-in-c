/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_kvfile.h - save and restore a prompt-prefix KV/compressor state.
 *
 * One versioned header, the token ids, the per-layer compressed-row counts and
 * one float payload. The loader validates geometry, config hash, model tag and
 * the row counts before it touches live state, then commits in one copy.
 */
#ifndef DSV4_KVFILE_H
#define DSV4_KVFILE_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_layer.h"

/* On-disk header shared by save and load. */
#define DSV4_KV_MAGIC   "DSV4KVC2"
#define DSV4_KV_VERSION 2u
#define DSV4_KV_TAG     64

/* The state payload is described entirely by the per-layer compression
 * geometry. These helpers are the single definition used by both directions,
 * so a load can never validate against a different layout than the save wrote. */
static inline int dsv4_kv_layer_ratio(const DSV4Cfg *c, int layer)
{
    const int r = dsv4_compress_ratio(c, layer);
    return r < 0 ? 0 : r;
}

static inline size_t dsv4_kv_layer_cn(const DSV4Cfg *c, int layer)
{
    return dsv4_comp_state_floats(c, layer);
}

static inline int64_t dsv4_kv_payload_floats(const DSV4Cfg *c,
                                             const int *ncomp, int n_ids)
{
    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;
    int64_t n = 0;
    for (int L = 0; L < c->n_layers; L++) {
        const int r = dsv4_kv_layer_ratio(c, L);
        const int want = r ? n_ids / r : 0;
        if (ncomp[L] != want) return -1;
        n += (int64_t)wn + (int64_t)ncomp[L] * c->head_dim
           + 2 * (int64_t)dsv4_kv_layer_cn(c, L);
    }
    return n;
}

/* FNV-1a over every config field that changes the shape or arithmetic of the
 * saved state. */
uint64_t dsv4_cfg_hash(const DSV4Cfg *c);

int dsv4_kv_save(const char *path, const char *model_tag,
                 const DSV4Cfg *c, const DSV4LayerState *lst,
                 const int32_t *ids, int n_ids);

/* Load a cache into `lst`. `max_pos` is the receiving engine's context cap.
 * When `expect_ids` is non-NULL its `n_expect` ids must match the file exactly.
 * On failure the live state and `ids` are untouched. */
int dsv4_kv_load(const char *path, const char *model_tag,
                 const DSV4Cfg *c, DSV4LayerState *lst, int max_pos,
                 int32_t *ids, int ids_cap, int *n_ids_out,
                 const int32_t *expect_ids, int n_expect);

#endif /* DSV4_KVFILE_H */

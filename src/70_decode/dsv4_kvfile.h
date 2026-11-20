/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_kvfile.h - save and restore a prompt-prefix KV/compressor state.
 *
 * Minimal trusted format for one model: an int32 token count, that many int32
 * token ids, then one float payload in layer order. The model geometry comes
 * from the running configuration; nothing model-specific is stored in the
 * file.
 */
#ifndef DSV4_KVFILE_H
#define DSV4_KVFILE_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_layer.h"

/* The state payload is described entirely by the per-layer compression
 * geometry. This is the single definition used by both directions. */
static inline int dsv4_kv_layer_ratio(const DSV4Cfg *c, int layer)
{
    const int r = dsv4_compress_ratio(c, layer);
    return r < 0 ? 0 : r;
}

static inline size_t dsv4_kv_layer_cn(const DSV4Cfg *c, int layer)
{
    return dsv4_comp_state_floats(c, layer);
}

static inline int64_t dsv4_kv_payload_floats(const DSV4Cfg *c, int n_ids)
{
    const size_t wn = (size_t)c->sliding_window * (size_t)c->head_dim;
    int64_t n = 0;
    for (int L = 0; L < c->n_layers; L++) {
        const int r = dsv4_kv_layer_ratio(c, L);
        n += (int64_t)wn;
        if (r) {
            n += (int64_t)(n_ids / r) * c->head_dim;
            n += 2 * (int64_t)dsv4_kv_layer_cn(c, L);
        }
    }
    return n;
}

int dsv4_kv_save(const char *path, const DSV4Cfg *c, const DSV4LayerState *lst,
                 const int32_t *ids, int n_ids);

/* Load a trusted cache into `lst`. Only I/O/allocation failures and the size
 * of the caller's token-id buffer are checked. */
int dsv4_kv_load(const char *path, const DSV4Cfg *c, DSV4LayerState *lst,
                 int32_t *ids, int ids_cap, int *n_ids_out);

#endif /* DSV4_KVFILE_H */

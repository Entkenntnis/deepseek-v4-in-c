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

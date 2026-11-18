/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_state_file.h - save and restore a prompt-prefix KV/compressor state.
 *
 * The file is deliberately self-describing. A cache is only valid for one
 * model geometry and one token prefix, so the loader validates the config hash,
 * the model tag, every per-layer compressed-row count and the exact token ids
 * before it touches any live layer state. The write is atomic (temp file plus
 * rename), and the read is atomic too (validate into a temporary buffer, then
 * commit), so a damaged or stale cache never leaves a half-loaded context.
 */
#ifndef DSV4_STATE_FILE_H
#define DSV4_STATE_FILE_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_layer.h"

/* FNV-1a over every config field that changes the shape or arithmetic of the
 * saved state. Exposed for tests; save and load use the same function. */
uint64_t dsv4_cfg_hash(const DSV4Cfg *c);

/* Save `n_ids` tokens at positions 0..n_ids-1 and the layer state they produced.
 * `model_tag` is a short, human-readable identity (the model directory's
 * basename). Returns 0 on success. */
int dsv4_state_file_save(const char *path, const char *model_tag,
                         const DSV4Cfg *c, const DSV4LayerState *lst,
                         const int32_t *ids, int n_ids);

/* Load a cache into `lst`. `max_pos` is the receiving engine's context cap.
 *
 * When `expect_ids` is non-NULL its `n_expect` ids must match the file exactly;
 * this is how the caller proves the saved prefix is the text it is about to
 * use. On success `ids[0..*n_ids_out)` and every layer are updated. On any
 * failure the live state and `ids` are untouched and -1 is returned. */
int dsv4_state_file_load(const char *path, const char *model_tag,
                         const DSV4Cfg *c, DSV4LayerState *lst, int max_pos,
                         int32_t *ids, int ids_cap, int *n_ids_out,
                         const int32_t *expect_ids, int n_expect);

#endif /* DSV4_STATE_FILE_H */

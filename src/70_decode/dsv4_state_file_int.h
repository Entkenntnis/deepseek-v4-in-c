/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_state_file_int.h - seams between the KV-cache file helpers. */
#ifndef DSV4_STATE_FILE_INT_H
#define DSV4_STATE_FILE_INT_H

#include <stdint.h>
#include <stdio.h>

#include "dsv4.h"
#include "dsv4_layer.h"

#define KV_MAGIC       "DSV4KVC"
#define KV_VERSION     1u
#define KV_TAG_MAX     128

#define KV_FNV_OFFSET  1469598103934665603ULL
#define KV_FNV_PRIME   1099511628211ULL

int    kv_w_u32(FILE *f, uint32_t v, uint64_t *hash);
int    kv_w_u64(FILE *f, uint64_t v, uint64_t *hash);
int    kv_w_i32(FILE *f, int32_t v, uint64_t *hash);
int    kv_w_f32(FILE *f, float val, uint64_t *hash);
int    kv_r_u32(FILE *f, uint32_t *v, uint64_t *hash);
int    kv_r_u64(FILE *f, uint64_t *v, uint64_t *hash);
int    kv_r_i32(FILE *f, int32_t *v, uint64_t *hash);
int    kv_r_f32(FILE *f, float *fout, uint64_t *hash);

int    kv_layer_ratio(const DSV4Cfg *c, int layer);
int    kv_layer_coff(int ratio);
size_t kv_layer_cn(const DSV4Cfg *c, int layer);
size_t kv_layer_window_floats(const DSV4Cfg *c);
void   kv_mkdir_parents(const char *path);

/* Validate the destination and commit a fully-read payload. */
int    kv_commit(const DSV4Cfg *c, DSV4LayerState *lst, int32_t *ids,
                 const int32_t *file_ids, int n_ids, const int *ncomp,
                 const float *buf);

#endif /* DSV4_STATE_FILE_INT_H */

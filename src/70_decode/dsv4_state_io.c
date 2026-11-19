/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_state_io.c - little-endian fields, geometry helpers and directory setup. */
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "dsv4_state_file_int.h"

static uint64_t fnv_bytes(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= KV_FNV_PRIME;
    }
    return h;
}

/* ---- little-endian field I/O --------------------------------------------- */

int kv_w_u32(FILE *f, uint32_t v, uint64_t *hash)
{
    unsigned char b[4];
    for (int i = 0; i < 4; i++) b[i] = (unsigned char)(v >> (8 * i));
    if (fwrite(b, 1, sizeof b, f) != sizeof b) return -1;
    if (hash) *hash = fnv_bytes(*hash, b, sizeof b);
    return 0;
}

int kv_w_u64(FILE *f, uint64_t v, uint64_t *hash)
{
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    if (fwrite(b, 1, sizeof b, f) != sizeof b) return -1;
    if (hash) *hash = fnv_bytes(*hash, b, sizeof b);
    return 0;
}

int kv_w_i32(FILE *f, int32_t v, uint64_t *hash)
{
    return kv_w_u32(f, (uint32_t)v, hash);
}

int kv_w_f32(FILE *f, float val, uint64_t *hash)
{
    union { float f; uint32_t u; } v;
    v.f = val;
    return kv_w_u32(f, v.u, hash);
}

int kv_r_u32(FILE *f, uint32_t *v, uint64_t *hash)
{
    unsigned char b[4];
    if (fread(b, 1, sizeof b, f) != sizeof b) return -1;
    if (hash) *hash = fnv_bytes(*hash, b, sizeof b);
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

int kv_r_u64(FILE *f, uint64_t *v, uint64_t *hash)
{
    unsigned char b[8];
    if (fread(b, 1, sizeof b, f) != sizeof b) return -1;
    if (hash) *hash = fnv_bytes(*hash, b, sizeof b);
    *v = 0;
    for (int i = 0; i < 8; i++) *v |= (uint64_t)b[i] << (8 * i);
    return 0;
}

int kv_r_i32(FILE *f, int32_t *v, uint64_t *hash)
{
    uint32_t u;
    if (kv_r_u32(f, &u, hash) != 0) return -1;
    *v = (int32_t)u;
    return 0;
}

int kv_r_f32(FILE *f, float *fout, uint64_t *hash)
{
    uint32_t u;
    union { float f; uint32_t u; } v;
    if (kv_r_u32(f, &u, hash) != 0) return -1;
    v.u = u;
    *fout = v.f;
    return 0;
}

/* ---- geometry helpers (must mirror dsv4_state_init) ---------------------- */

int kv_layer_ratio(const DSV4Cfg *c, int layer)
{
    int r = dsv4_compress_ratio(c, layer);
    return r < 0 ? 0 : r;
}

int kv_layer_coff(int ratio)
{
    return ratio == 4 ? 2 : 1;
}

size_t kv_layer_cn(const DSV4Cfg *c, int layer)
{
    const int ratio = kv_layer_ratio(c, layer);
    if (!ratio) return 0;
    const int coff = kv_layer_coff(ratio);
    return (size_t)coff * (size_t)ratio * (size_t)coff *
           (size_t)c->head_dim;
}

size_t kv_layer_window_floats(const DSV4Cfg *c)
{
    return (size_t)c->sliding_window * (size_t)c->head_dim;
}

/* ---- directory creation -------------------------------------------------- */

void kv_mkdir_parents(const char *path)
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
        if (mkdir(buf, 0755) != 0 && errno != EEXIST) { /* leave fopen to report */ }
        *p = '/';
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) { /* leave fopen to report */ }
}

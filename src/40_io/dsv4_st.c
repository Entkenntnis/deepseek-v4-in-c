/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st.c - safetensors index: open, close and look up. */
#define _GNU_SOURCE            /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "dsv4_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before libc */

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_st.h"
#include "dsv4_st_int.h"

int dsv4_st_elemsize(DSV4Dtype d)
{
    switch (d) {
    case DSV4_DT_U8:      return 1;
    /* One STORED byte each. For I8 that byte holds two FP4 values, so this is
     * bytes-per-stored-unit, not bytes-per-value; see DSV4_DT_I8 in the header. */
    case DSV4_DT_F8_E4M3:
    case DSV4_DT_F8_E8M0:
    case DSV4_DT_I8:      return 1;
    case DSV4_DT_BF16:
    case DSV4_DT_F16:     return 2;
    case DSV4_DT_F32:     return 4;
    case DSV4_DT_I64:     return 8;
    default:              return 0;
    }
}

int64_t dsv4_st_numel(const DSV4Tensor *t)
{
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return t->ndim ? n : 1;
}

/* FNV-1a. Names are long and share deep prefixes, so the hash must mix every
 * byte; a prefix-only or length-only hash would pile every expert of a layer
 * into one bucket. */
static uint64_t fnv1a(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ull; }
    return h;
}

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

int dsv4_st_open(DSV4St *s, const char *dir)
{
    memset(s, 0, sizeof *s);

    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "dsv4_st: cannot open directory %s\n", dir);
        return -1;
    }

    char **files = NULL; int nf = 0, cf = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 12 || strcmp(e->d_name + n - 12, ".safetensors")) continue;
        if (nf == cf) {
            cf = cf ? cf * 2 : 32;
            files = (char **)realloc(files, cf * sizeof *files);
        }
        size_t len = strlen(dir) + 1 + n + 1;
        files[nf] = (char *)malloc(len);
        snprintf(files[nf], len, "%s/%s", dir, e->d_name);
        nf++;
    }
    closedir(d);

    if (nf == 0) {
        fprintf(stderr, "dsv4_st: no .safetensors files in %s\n", dir);
        free(files);
        return -1;
    }
    /* Sort so shard indices are stable across runs and machines; readdir order
     * is not. */
    qsort(files, nf, sizeof *files, cmp_str);

    s->path = files; s->nshard = nf;
    s->fd  = (int *)malloc(nf * sizeof(int));
    s->dfd = (int *)malloc(nf * sizeof(int));
    /* dsv4_st_close, not free(files): s->path was aliased to `files` above, so
     * freeing it here leaves s->path dangling and close would free it again.
     * Let the one function that owns the teardown do all of it. */
    if (!s->fd || !s->dfd) { dsv4_st_close(s); return -1; }
    for (int i = 0; i < nf; i++) { s->fd[i] = -1; s->dfd[i] = -1; }

    DSV4StBuild b; memset(&b, 0, sizeof b);
    for (int i = 0; i < nf; i++) {
        if (dsv4_st_scan_shard(s, &b, i, files[i]) < 0) {
            free(b.t); free(b.noff); dsv4_st_close(s); return -1;
        }
    }

    /* Resolve names now that the pool has stopped moving. Storing char* during
     * the scan would leave every earlier pointer dangling after a realloc. */
    s->t  = b.t;
    s->nt = (int)b.n;
    for (size_t i = 0; i < b.n; i++) s->t[i].name = s->strpool + b.noff[i];
    free(b.noff);

    int nb = 1024;
    while (nb < s->nt * 2) nb <<= 1;              /* load factor below 0.5 */
    s->nbucket = nb;
    s->bucket = (int32_t *)malloc((size_t)nb * sizeof(int32_t));
    if (!s->bucket) { dsv4_st_close(s); return -1; }
    memset(s->bucket, 0xFF, (size_t)nb * sizeof(int32_t));   /* -1 */

    for (int i = 0; i < s->nt; i++) {
        uint64_t h = fnv1a(s->t[i].name);
        int j = (int)(h & (uint64_t)(nb - 1));
        while (s->bucket[j] >= 0) {
            if (!strcmp(s->t[s->bucket[j]].name, s->t[i].name)) {
                fprintf(stderr, "dsv4_st: duplicate tensor name %s\n", s->t[i].name);
                dsv4_st_close(s); return -1;
            }
            j = (j + 1) & (nb - 1);
        }
        s->bucket[j] = i;
    }
    return 0;
}

void dsv4_st_close(DSV4St *s)
{
    if (s->fd) {
        for (int i = 0; i < s->nshard; i++) if (s->fd[i] >= 0) close(s->fd[i]);
        free(s->fd);
    }
    if (s->dfd) {
        for (int i = 0; i < s->nshard; i++) if (s->dfd[i] >= 0) close(s->dfd[i]);
        free(s->dfd);
    }
    if (s->path) {
        for (int i = 0; i < s->nshard; i++) free(s->path[i]);
        free(s->path);
    }
    free(s->t); free(s->bucket); free(s->strpool);
    memset(s, 0, sizeof *s);
}

const DSV4Tensor *dsv4_st_find(const DSV4St *s, const char *name)
{
    if (!s->bucket) return NULL;
    uint64_t h = fnv1a(name);
    int j = (int)(h & (uint64_t)(s->nbucket - 1));
    while (s->bucket[j] >= 0) {
        const DSV4Tensor *t = &s->t[s->bucket[j]];
        if (!strcmp(t->name, name)) return t;
        j = (j + 1) & (s->nbucket - 1);
    }
    return NULL;
}

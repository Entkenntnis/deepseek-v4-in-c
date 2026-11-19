/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st.c - open the shards and read the packed layout indexes. */
#define _GNU_SOURCE            /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_st.h"

#define ST_ALIGN 4096

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static FILE *open_index(const char *dir, const char *name, const char *magic)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "dsv4_st: cannot open %s\n", path);
        return NULL;
    }
    char got[8];
    if (fread(got, 1, 8, f) != 8 || memcmp(got, magic, 8) != 0) {
        fprintf(stderr, "dsv4_st: %s is not a %s index; re-run "
                        "tools/pack_trunk.py\n", path, name);
        fclose(f);
        return NULL;
    }
    return f;
}

static int load_models(DSV4St *s, const char *dir)
{
    FILE *f = open_index(dir, "model.idx", "DSV4MDL1");
    if (!f) return -1;
    int32_t n = 0, pad = 0;
    if (fread(&n, sizeof n, 1, f) != 1 || fread(&pad, sizeof pad, 1, f) != 1 ||
        n <= 0 || n > 64) {
        fclose(f);
        return -1;
    }
    s->model = (DSV4ModelLoc *)calloc((size_t)n, sizeof *s->model);
    if (!s->model) { fclose(f); return -1; }
    if (fread(s->model, sizeof *s->model, (size_t)n, f) != (size_t)n) {
        fclose(f);
        return -1;
    }
    s->nmodel = n;
    fclose(f);
    return 0;
}

static int load_experts(DSV4St *s, const char *dir)
{
    FILE *f = open_index(dir, "experts.idx", "DSV4EXP1");
    if (!f) return -1;
    int32_t nl = 0, ne = 0;
    if (fread(&nl, sizeof nl, 1, f) != 1 || fread(&ne, sizeof ne, 1, f) != 1 ||
        nl <= 0 || ne <= 0) {
        fclose(f);
        return -1;
    }
    const size_t n = (size_t)nl * (size_t)ne * DSV4_EXP_TENSORS;
    s->exp = (DSV4Loc *)malloc(n * sizeof *s->exp);
    if (!s->exp) { fclose(f); return -1; }
    if (fread(s->exp, sizeof *s->exp, n, f) != n) {
        fclose(f);
        return -1;
    }
    s->n_layers = nl;
    s->n_experts = ne;
    fclose(f);
    return 0;
}

static int open_shards(DSV4St *s, const char *dir)
{
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "dsv4_st: cannot open directory %s\n", dir);
        return -1;
    }

    char **files = NULL;
    int nf = 0, cf = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        const size_t n = strlen(e->d_name);
        if (n < 12 || strcmp(e->d_name + n - 12, ".safetensors")) continue;
        if (nf == cf) {
            cf = cf ? cf * 2 : 32;
            char **p = (char **)realloc(files, (size_t)cf * sizeof *files);
            if (!p) { closedir(d); return -1; }
            files = p;
        }
        const size_t len = strlen(dir) + 1 + n + 1;
        files[nf] = (char *)malloc(len);
        if (!files[nf]) { closedir(d); return -1; }
        snprintf(files[nf], len, "%s/%s", dir, e->d_name);
        nf++;
    }
    closedir(d);
    if (nf == 0) {
        fprintf(stderr, "dsv4_st: no .safetensors files in %s\n", dir);
        free(files);
        return -1;
    }
    qsort(files, (size_t)nf, sizeof *files, cmp_str);

    s->nshard = nf;
    s->fd = (int *)malloc((size_t)nf * sizeof *s->fd);
    s->dfd = (int *)malloc((size_t)nf * sizeof *s->dfd);
    if (!s->fd || !s->dfd) {
        for (int i = 0; i < nf; i++) free(files[i]);
        free(files);
        return -1;
    }
    for (int i = 0; i < nf; i++) {
        s->fd[i] = open(files[i], O_RDONLY);
        s->dfd[i] = s->fd[i] < 0 ? -1 : open(files[i], O_RDONLY | O_DIRECT);
        if (s->fd[i] < 0)
            fprintf(stderr, "dsv4_st: cannot open %s\n", files[i]);
        free(files[i]);
    }
    free(files);
    return 0;
}

int dsv4_st_open(DSV4St *s, const char *model_dir, const char *index_dir)
{
    memset(s, 0, sizeof *s);
    if (open_shards(s, model_dir) != 0) { dsv4_st_close(s); return -1; }
    if (load_models(s, index_dir) != 0) { dsv4_st_close(s); return -1; }
    if (load_experts(s, index_dir) != 0) { dsv4_st_close(s); return -1; }
    return 0;
}

void dsv4_st_close(DSV4St *s)
{
    if (!s) return;
    for (int i = 0; i < s->nshard; i++) {
        if (s->fd && s->fd[i] >= 0) close(s->fd[i]);
        if (s->dfd && s->dfd[i] >= 0) close(s->dfd[i]);
    }
    free(s->fd);
    free(s->dfd);
    free(s->exp);
    free(s->model);
    memset(s, 0, sizeof *s);
}

const DSV4Loc *dsv4_st_expert(const DSV4St *s, int layer, int expert)
{
    if (!s || layer < 0 || layer >= s->n_layers ||
        expert < 0 || expert >= s->n_experts)
        return NULL;
    const size_t row = (size_t)layer * (size_t)s->n_experts + (size_t)expert;
    return s->exp + row * DSV4_EXP_TENSORS;
}

const DSV4ModelLoc *dsv4_st_model(const DSV4St *s, int slot)
{
    if (!s) return NULL;
    for (int i = 0; i < s->nmodel; i++)
        if (s->model[i].slot == slot) return &s->model[i];
    return NULL;
}

int dsv4_st_elemsize(DSV4Dtype d)
{
    switch (d) {
    case DSV4_DT_U8:
    case DSV4_DT_F8_E4M3:
    case DSV4_DT_F8_E8M0:
    case DSV4_DT_I8:   return 1;
    case DSV4_DT_BF16:
    case DSV4_DT_F16:  return 2;
    case DSV4_DT_F32:  return 4;
    case DSV4_DT_I64:  return 8;
    default:           return 0;
    }
}

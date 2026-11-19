/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk_manifest.c - read trunk.idx and resolve the bind order.
 *
 * The manifest is line-based:
 *
 *   DSV4TRUNK 1
 *   n_layers 43
 *   layer <file_off> <nbytes> <ntensor>
 *   tensor <name> <off> <nbytes> <dtype>
 *   ...
 *
 * One `layer` line per layer, including absent ones as `layer 0 0 0`, followed
 * by exactly `ntensor` tensor lines. Names contain no whitespace.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_bind.h"
#include "dsv4_trunk.h"
#include "dsv4_trunk_int.h"

/* The manifest stores dtypes as the safetensors spelling, so the mapping is the
 * reader's, not a second table that could drift from it. */
static int dtype_from_name(const char *s)
{
    if (!strcmp(s, "BF16"))    return DSV4_DT_BF16;
    if (!strcmp(s, "F32"))     return DSV4_DT_F32;
    if (!strcmp(s, "F16"))     return DSV4_DT_F16;
    if (!strcmp(s, "U8"))      return DSV4_DT_U8;
    if (!strcmp(s, "I8"))      return DSV4_DT_I8;
    if (!strcmp(s, "I64"))     return DSV4_DT_I64;
    if (!strcmp(s, "F8_E4M3")) return DSV4_DT_F8_E4M3;
    if (!strcmp(s, "F8_E8M0")) return DSV4_DT_F8_E8M0;
    return DSV4_DT_UNKNOWN;
}

/* Next non-blank line, with its newline removed. */
static int next_line(FILE *f, char **line, size_t *cap)
{
    ssize_t n;
    while ((n = getline(line, cap, f)) >= 0) {
        while (n > 0 && ((*line)[n - 1] == '\n' || (*line)[n - 1] == '\r'))
            (*line)[--n] = 0;
        char *p = *line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p) return 1;
    }
    return 0;
}

static int bad(const char *path, const char *what, int L)
{
    fprintf(stderr, "dsv4_trunk: %s: bad %s line for layer %d\n",
            path, what, L);
    return -1;
}

int dsv4_trunk_load_manifest(DSV4Trunk *tr, const char *dir)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/trunk.idx", dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "dsv4_trunk: cannot open %s (re-run tools/pack_trunk.py "
                        "or tools/pack_trunk.py --reindex)\n", path);
        return -1;
    }

    char *line = NULL;
    size_t cap = 0;
    int rc = -1;
    char magic[32];
    int version = 0;

    if (!next_line(f, &line, &cap) ||
        sscanf(line, "%31s %d", magic, &version) != 2 ||
        strcmp(magic, "DSV4TRUNK") != 0 || version != 1) {
        fprintf(stderr, "dsv4_trunk: %s: not a version-1 trunk index\n", path);
        goto out;
    }
    if (!next_line(f, &line, &cap) ||
        sscanf(line, "n_layers %d", &tr->n_layers) != 1 ||
        tr->n_layers <= 0) {
        fprintf(stderr, "dsv4_trunk: %s: missing or invalid n_layers\n", path);
        goto out;
    }

    tr->lay = (DSV4TrunkLayer *)calloc((size_t)tr->n_layers,
                                       sizeof(DSV4TrunkLayer));
    if (!tr->lay) goto out;

    for (int L = 0; L < tr->n_layers; L++) {
        long long file_off = 0, nbytes = 0;
        int nt = 0;
        if (!next_line(f, &line, &cap) ||
            sscanf(line, "layer %lld %lld %d", &file_off, &nbytes, &nt) != 3) {
            bad(path, "layer", L);
            goto out;
        }
        if (nt < 0) { bad(path, "tensor count in layer", L); goto out; }

        DSV4TrunkLayer *lay = &tr->lay[L];
        lay->file_off = file_off;
        lay->nbytes = nbytes;
        lay->nt = nt;
        lay->t = (DSV4TrunkTensor *)calloc((size_t)(nt ? nt : 1),
                                           sizeof(DSV4TrunkTensor));
        if (!lay->t) goto out;

        for (int i = 0; i < nt; i++) {
            char name[512], dtype[512];
            long long off = 0, nb = 0;
            if (!next_line(f, &line, &cap) ||
                sscanf(line, "tensor %511s %lld %lld %511s",
                       name, &off, &nb, dtype) != 4) {
                bad(path, "tensor", L);
                goto out;
            }
            lay->t[i].name = strdup(name);
            if (!lay->t[i].name) goto out;
            lay->t[i].off = off;
            lay->t[i].nbytes = nb;
            lay->t[i].dtype = dtype_from_name(dtype);
            if (lay->t[i].dtype == DSV4_DT_UNKNOWN) {
                fprintf(stderr, "dsv4_trunk: layer %d tensor %s has dtype '%s', "
                                "which this build does not implement\n",
                        L, name, dtype);
                goto out;
            }
        }
    }
    rc = 0;

out:
    free(line);
    fclose(f);
    return rc;
}

/* Resolve plan_layer()'s request order to manifest indices once per layer.
 * That turns the per-token bind loop into an array walk instead of formatting
 * and comparing names every token. */
typedef struct { DSV4TrunkLayer *lay; } BindMapCtx;

static int bind_map_emit(void *ctx, int i, const char *name)
{
    DSV4TrunkLayer *lay = ((BindMapCtx *)ctx)->lay;
    int found = -1;
    for (int j = 0; j < lay->nt; j++)
        if (!strcmp(lay->t[j].name, name)) { found = j; break; }
    lay->bind_of[i] = found;
    return 0;
}

int dsv4_trunk_build_maps(DSV4Trunk *tr, const DSV4Cfg *c)
{
    /* A trunk can legally carry layers this config does not use; only build
     * maps for layers the engine can ask for, because plan_layer() reads the
     * per-layer ratio from config. */
    for (int L = 0; L < tr->n_layers && L < c->n_layers; L++) {
        DSV4TrunkLayer *lay = &tr->lay[L];
        if (lay->nt == 0) continue;           /* layer absent: nothing to bind */
        const int n = dsv4_bind_layer_plan(c, L, NULL, NULL);
        if (n < 0) return -1;
        lay->bind_of = (int32_t *)malloc((size_t)(n ? n : 1)
                                         * sizeof *lay->bind_of);
        if (!lay->bind_of) return -1;
        lay->nbind = n;
        BindMapCtx ctx = { lay };
        if (dsv4_bind_layer_plan(c, L, bind_map_emit, &ctx) != n) return -1;
    }
    return 0;
}

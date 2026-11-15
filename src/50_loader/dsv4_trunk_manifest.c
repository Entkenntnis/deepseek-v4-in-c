/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk_manifest.c - read trunk.json and resolve the bind order. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_bind.h"
#include "dsv4_trunk.h"
#include "dsv4_trunk_int.h"
#include "json.h"

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

static char *slurp(const char *path, long *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) { fclose(f); return NULL; }
    if (fread(s, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    s[n] = 0; fclose(f);
    if (len) *len = n;
    return s;
}

int dsv4_trunk_load_manifest(DSV4Trunk *tr, const char *dir)
{
    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.json", dir);
    tr->json_text = slurp(p, NULL);
    if (!tr->json_text) return -1;

    jval *root = json_parse(tr->json_text, &tr->json_arena);
    if (!root) { fprintf(stderr, "%s: not valid JSON\n", p); return -1; }

    jval *nl = json_get(root, "n_layers");
    jval *ls = json_get(root, "layers");
    if (!nl || !ls || ls->t != J_ARR) {
        fprintf(stderr, "%s: missing n_layers or layers\n", p);
        return -1;
    }
    tr->n_layers = (int)nl->num;
    tr->lay = (DSV4TrunkLayer *)calloc((size_t)tr->n_layers,
                                       sizeof(DSV4TrunkLayer));
    if (!tr->lay) return -1;

    for (int L = 0; L < tr->n_layers && L < ls->len; L++) {
        jval *o = ls->kids[L];
        if (o->t != J_OBJ) continue;                 /* null = layer absent */
        DSV4TrunkLayer *lay = &tr->lay[L];
        lay->file_off = (int64_t)json_get(o, "file_off")->num;
        lay->nbytes   = (int64_t)json_get(o, "nbytes")->num;
        jval *ts = json_get(o, "tensors");
        lay->nt = ts->len;
        lay->t = (DSV4TrunkTensor *)calloc((size_t)lay->nt,
                                           sizeof(DSV4TrunkTensor));
        for (int i = 0; i < lay->nt; i++) {
            jval *e = ts->kids[i];
            lay->t[i].name   = json_get(e, "name")->str;   /* borrowed */
            lay->t[i].off    = (int64_t)json_get(e, "off")->num;
            lay->t[i].nbytes = (int64_t)json_get(e, "nbytes")->num;
            lay->t[i].dtype  = dtype_from_name(json_get(e, "dtype")->str);
            if (lay->t[i].dtype == DSV4_DT_UNKNOWN) {
                fprintf(stderr, "dsv4_trunk: layer %d tensor %s has dtype '%s', "
                                "which this build does not implement\n",
                        L, lay->t[i].name, json_get(e, "dtype")->str);
                return -1;
            }
        }
    }
    return 0;
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

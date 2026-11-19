/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk_manifest.c - read trunk.idx, the packer's layer layout. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_st.h"
#include "dsv4_trunk.h"

/* Next non-blank line, newline removed. */
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

int dsv4_trunk_load_manifest(DSV4Trunk *tr, const char *dir)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/trunk.idx", dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "dsv4_trunk: cannot open %s (re-run "
                        "tools/pack_trunk.py)\n", path);
        return -1;
    }

    char *line = NULL;
    size_t cap = 0;
    int rc = -1;
    char magic[32];
    int version = 0, layout = 0;

    if (!next_line(f, &line, &cap) ||
        sscanf(line, "%31s %d", magic, &version) != 2 ||
        strcmp(magic, "DSV4TRUNK") != 0 || version != 2) {
        fprintf(stderr, "dsv4_trunk: %s is not a version-2 trunk index\n", path);
        goto out;
    }
    if (!next_line(f, &line, &cap) ||
        sscanf(line, "n_layers %d", &tr->n_layers) != 1 ||
        tr->n_layers <= 0) {
        fprintf(stderr, "dsv4_trunk: %s: missing or invalid n_layers\n", path);
        goto out;
    }
    if (!next_line(f, &line, &cap) ||
        sscanf(line, "layout %d", &layout) != 1 || layout != 1) {
        fprintf(stderr, "dsv4_trunk: %s: unsupported layer layout\n", path);
        goto out;
    }

    tr->lay = (DSV4TrunkLayer *)calloc((size_t)tr->n_layers,
                                       sizeof(DSV4TrunkLayer));
    if (!tr->lay) goto out;

    for (int L = 0; L < tr->n_layers; L++) {
        long long file_off = 0, nbytes = 0, widen = 0;
        int nt = 0, gotL = -1;
        if (!next_line(f, &line, &cap) ||
            sscanf(line, "layer %d %lld %lld %d %lld",
                   &gotL, &file_off, &nbytes, &nt, &widen) != 5 ||
            gotL != L) {
            fprintf(stderr, "dsv4_trunk: %s: bad layer %d line\n", path, L);
            goto out;
        }
        DSV4TrunkLayer *lay = &tr->lay[L];
        lay->file_off = file_off;
        lay->nbytes = nbytes;
        lay->widen_bytes = widen;
        lay->nt = nt;
        if (nt <= 0) continue;

        lay->t = (DSV4TrunkTensor *)calloc((size_t)nt, sizeof *lay->t);
        if (!lay->t) goto out;
        for (int i = 0; i < nt; i++) {
            DSV4TrunkTensor *t = &lay->t[i];
            long long off_val = 0, nb_val = 0;
            if (!next_line(f, &line, &cap) ||
                sscanf(line, "slot %d %lld %lld %d %d %d %d %d %d",
                       &t->slot, &off_val, &nb_val, &t->dtype,
                       &t->rows, &t->cols, &t->wdt, &t->blk_r,
                       &t->blk_c) != 9) {
                fprintf(stderr, "dsv4_trunk: %s: bad tensor in layer %d\n",
                        path, L);
                goto out;
            }
            t->off = off_val;
            t->nbytes = nb_val;
            if (t->dtype < DSV4_DT_U8 || t->dtype > DSV4_DT_I64 ||
                t->wdt < DSV4_WF32 || t->wdt > DSV4_WFP4) {
                fprintf(stderr, "dsv4_trunk: %s: layer %d slot %d has an "
                                "invalid dtype/format\n", path, L, t->slot);
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

const DSV4TrunkTensor *dsv4_trunk_slot(const DSV4Trunk *tr, int L, int slot)
{
    if (!tr || L < 0 || L >= tr->n_layers) return NULL;
    const DSV4TrunkLayer *lay = &tr->lay[L];
    for (int i = 0; i < lay->nt; i++)
        if (lay->t[i].slot == slot) return &lay->t[i];
    return NULL;
}

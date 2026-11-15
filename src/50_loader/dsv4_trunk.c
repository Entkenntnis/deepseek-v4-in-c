/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk.c - open and size the streamed trunk. */

/* O_DIRECT is a GNU extension, so the feature macro has to precede EVERY libc
 * header. Defining it after an include silently leaves O_DIRECT undeclared and
 * the fallback path becomes the only path -- a 15x slowdown that compiles. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "dsv4_portable_io.h"   /* then: sets _DARWIN_C_SOURCE before libc */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_trunk.h"
#include "dsv4_trunk_int.h"

int dsv4_trunk_open(DSV4Trunk *tr, const char *dir, const DSV4Cfg *c,
                    int64_t budget_bytes)
{
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
    if (dsv4_trunk_load_manifest(tr, dir) != 0) { dsv4_trunk_close(tr); return -1; }
    if (dsv4_trunk_build_maps(tr, c) != 0) {
        fprintf(stderr, "dsv4_trunk: cannot resolve the bind map\n");
        dsv4_trunk_close(tr); return -1;
    }

    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.bin", dir);
    tr->fd = open(p, O_RDONLY | O_DIRECT);
    tr->direct = 1;
    if (tr->fd < 0) {
        /* O_DIRECT is unavailable on some filesystems -- notably /mnt/c under
         * WSL, where it works but much slower. Falling back keeps the engine
         * CORRECT there; it does not make it fast. */
        tr->fd = open(p, O_RDONLY);
        tr->direct = 0;
        if (tr->fd < 0) { perror(p); dsv4_trunk_close(tr); return -1; }
        fprintf(stderr, "dsv4_trunk: O_DIRECT unavailable on %s, using buffered "
                        "reads (expect a large slowdown)\n", p);
    }
    dsv4_set_direct(tr->fd);

    /* Widest layer decides the slot size, because ring slots are uniform. */
    int64_t widest = 0;
    for (int L = 0; L < tr->n_layers; L++)
        if (tr->lay[L].nbytes > widest) widest = tr->lay[L].nbytes;
    /* Room for an O_DIRECT read widened out to the enclosing aligned window. */
    tr->slot_bytes = (widest + 2 * DSV4_TRUNK_ALIGN + DSV4_TRUNK_ALIGN - 1)
                   / DSV4_TRUNK_ALIGN * DSV4_TRUNK_ALIGN;

    tr->widen_cap = dsv4_bind_widen_bytes(c);
    if (posix_memalign((void **)&tr->widen, DSV4_TRUNK_ALIGN,
                       tr->widen_cap ? tr->widen_cap : 1) != 0) {
        dsv4_trunk_close(tr); return -1;
    }

    /* Spend the budget on pinned layers first, keeping one ring slot. A pinned
     * layer gets an EXACT allocation: sizing every pin like the widest layer
     * would waste 34 MB per pin between an indexed and a dense layer. */
    int64_t left = budget_bytes - tr->slot_bytes - (int64_t)tr->widen_cap;
    int npin = 0;
    while (npin < tr->n_layers && tr->lay[npin].nbytes > 0
           && left >= tr->lay[npin].nbytes) {
        left -= tr->lay[npin].nbytes;
        npin++;
    }
    tr->npin = npin;
    tr->nslot = 1 + (int)(left / tr->slot_bytes);
    if (tr->nslot < 1) tr->nslot = 1;
    /* Never more slots than unpinned layers to put in them: a generous budget
     * otherwise asks for thousands of slots and the allocation fails, so an
     * over-provisioned machine could not open the trunk at all. */
    {
        const int tail = tr->n_layers - tr->npin;
        const int cap = tail > 0 ? tail : 1;
        if (tr->nslot > cap) tr->nslot = cap;
    }

    tr->pin = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof(void *));
    tr->layer_of = (int *)malloc((size_t)tr->nslot * sizeof(int));
    tr->slot_of  = (int32_t *)malloc((size_t)tr->n_layers * sizeof(int32_t));
    if (!tr->pin || !tr->layer_of || !tr->slot_of) {
        dsv4_trunk_close(tr); return -1;
    }
    for (int i = 0; i < tr->nslot; i++) tr->layer_of[i] = -1;
    for (int i = 0; i < tr->n_layers; i++) tr->slot_of[i] = -1;

    if (posix_memalign((void **)&tr->ring, DSV4_TRUNK_ALIGN,
                       (size_t)tr->nslot * (size_t)tr->slot_bytes) != 0) {
        dsv4_trunk_close(tr); return -1;
    }
    return 0;
}

int64_t dsv4_trunk_resident_bytes(const DSV4Trunk *tr)
{
    int64_t n = (int64_t)tr->nslot * tr->slot_bytes + (int64_t)tr->widen_cap;
    for (int i = 0; i < tr->npin; i++) n += tr->lay[i].nbytes;
    return n;
}

void dsv4_trunk_close(DSV4Trunk *tr)
{
    if (!tr) return;
    if (tr->fd >= 0) close(tr->fd);
    for (int i = 0; i < tr->npin; i++) if (tr->pin) free(tr->pin[i]);
    free(tr->pin); free(tr->ring); free(tr->widen);
    free(tr->layer_of); free(tr->slot_of);
    if (tr->lay) for (int L = 0; L < tr->n_layers; L++) {
        free(tr->lay[L].t);
        free(tr->lay[L].bind_of);
        free(tr->lay[L].bind_widen);
    }
    free(tr->lay); free(tr->json_arena); free(tr->json_text);
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
}

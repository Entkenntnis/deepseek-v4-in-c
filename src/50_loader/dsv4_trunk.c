/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk.c - open, stream and bind the dense trunk. */
#define _GNU_SOURCE

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_trunk.h"
#include "dsv4_bind.h"

int dsv4_trunk_read_run(DSV4Trunk *tr, int L, unsigned char *dst)
{
    const int64_t off = tr->lay[L].file_off;
    int64_t want = tr->lay[L].nbytes;
    if (tr->direct)
        want = (want + DSV4_TRUNK_ALIGN - 1) / DSV4_TRUNK_ALIGN
             * DSV4_TRUNK_ALIGN;

    int64_t got = 0;
    while (got < want) {
        const ssize_t r = pread(tr->fd, dst + got, (size_t)(want - got),
                                (off_t)(off + got));
        if (r <= 0) {
            if (r == 0 && got >= tr->lay[L].nbytes) break;
            fprintf(stderr, "dsv4_trunk: short read on layer %d at %lld\n",
                    L, (long long)(off + got));
            return -1;
        }
        got += r;
    }
    return 0;
}

int dsv4_trunk_open(DSV4Trunk *tr, const char *dir, const DSV4Cfg *c,
                    int64_t budget_bytes)
{
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
    tr->bfd = -1;
    if (dsv4_trunk_load_manifest(tr, dir) != 0) {
        dsv4_trunk_close(tr);
        return -1;
    }

    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.bin", dir);
    tr->fd = open(p, O_RDONLY | O_DIRECT);
    tr->direct = 1;
    if (tr->fd < 0) {
        tr->fd = open(p, O_RDONLY);
        tr->direct = 0;
        if (tr->fd < 0) { perror(p); dsv4_trunk_close(tr); return -1; }
        fprintf(stderr, "dsv4_trunk: O_DIRECT unavailable on %s, using "
                        "buffered reads (expect a large slowdown)\n", p);
    }
    tr->bfd = open(p, O_RDONLY);
    if (tr->bfd < 0) { dsv4_trunk_close(tr); return -1; }

    int64_t widest = 0;
    tr->widen_cap = 0;
    for (int L = 0; L < tr->n_layers; L++) {
        if (tr->lay[L].nbytes > widest) widest = tr->lay[L].nbytes;
        if (tr->lay[L].widen_bytes > (int64_t)tr->widen_cap)
            tr->widen_cap = (size_t)tr->lay[L].widen_bytes;
    }
    tr->slot_bytes = (widest + 2 * DSV4_TRUNK_ALIGN + DSV4_TRUNK_ALIGN - 1)
                   / DSV4_TRUNK_ALIGN * DSV4_TRUNK_ALIGN;

    if (posix_memalign((void **)&tr->widen, DSV4_TRUNK_ALIGN,
                       tr->widen_cap ? tr->widen_cap : 1) != 0) {
        dsv4_trunk_close(tr);
        return -1;
    }

        int64_t left = budget_bytes - tr->slot_bytes - (int64_t)tr->widen_cap;
    int npin = 0;
    while (npin < tr->n_layers && tr->lay[npin].nbytes > 0
           && left >= tr->lay[npin].nbytes) {
        left -= tr->lay[npin].nbytes;
        npin++;
    }
    tr->npin = npin;

    const int tail = tr->n_layers - tr->npin;
    if (tail <= 0) {
        tr->nslot = 0;
    } else {
        tr->nslot = 1 + (int)(left / tr->slot_bytes);
        if (tr->nslot < 1) tr->nslot = 1;
        if (tr->nslot > tail) tr->nslot = tail;
    }

    tr->pin = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof(void *));
    tr->layer_of = tr->nslot
                 ? (int *)malloc((size_t)tr->nslot * sizeof(int)) : NULL;
    tr->slot_of = (int32_t *)malloc((size_t)tr->n_layers * sizeof(int32_t));
    if (!tr->pin || (tr->nslot && !tr->layer_of) || !tr->slot_of) {
        dsv4_trunk_close(tr);
        return -1;
    }
    for (int i = 0; i < tr->nslot; i++) tr->layer_of[i] = -1;
    for (int i = 0; i < tr->n_layers; i++) tr->slot_of[i] = -1;

    if (tr->nslot > 0 &&
        posix_memalign((void **)&tr->ring, DSV4_TRUNK_ALIGN,
                       (size_t)tr->nslot * (size_t)tr->slot_bytes) != 0) {
        dsv4_trunk_close(tr);
        return -1;
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
    if (tr->bfd >= 0) close(tr->bfd);
    for (int i = 0; i < tr->npin; i++) if (tr->pin) free(tr->pin[i]);
    free(tr->pin);
    free(tr->ring);
    free(tr->widen);
    free(tr->layer_of);
    free(tr->slot_of);
    if (tr->lay) for (int L = 0; L < tr->n_layers; L++) {
        free(tr->lay[L].t);
        free(tr->lay[L].bind_widen);
    }
    free(tr->lay);
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
    tr->bfd = -1;
}

int dsv4_trunk_bind(DSV4Trunk *tr, const DSV4Cfg *c, int L, DSV4LayerW *w)
{
    if (L < 0 || L >= tr->n_layers || tr->lay[L].nt == 0) {
        fprintf(stderr, "dsv4_trunk: layer %d is not in the trunk\n", L);
        return -1;
    }

    unsigned char *run;
    if (L < tr->npin) {
        if (!tr->pin[L]) {
            if (posix_memalign((void **)&tr->pin[L], DSV4_TRUNK_ALIGN,
                               (size_t)tr->lay[L].nbytes
                               + 2 * DSV4_TRUNK_ALIGN) != 0)
                return -1;
            if (dsv4_trunk_read_run(tr, L, tr->pin[L]) != 0) return -1;
        }
        run = tr->pin[L];
    } else if (tr->slot_of[L] >= 0) {
        run = tr->ring + (size_t)tr->slot_of[L] * (size_t)tr->slot_bytes;
    } else {
        if (tr->nslot <= 0) {
            fprintf(stderr, "dsv4_trunk: layer %d is neither pinned nor has a "
                            "ring slot\n", L);
            return -1;
        }
        const int slot = tr->next_slot;
        tr->next_slot = (tr->next_slot + 1) % tr->nslot;
        if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;
        run = tr->ring + (size_t)slot * (size_t)tr->slot_bytes;
        if (dsv4_trunk_read_run(tr, L, run) != 0) return -1;
        tr->layer_of[slot] = L;
        tr->slot_of[L] = slot;
    }

    DSV4TrunkLayer *lay = &tr->lay[L];
    if (L < tr->npin && lay->bind_ready) {
        *w = lay->bind_cached;
        return 0;
    }

    if (L < tr->npin) {
        unsigned char *wbuf = (unsigned char *)malloc(
            lay->widen_bytes ? (size_t)lay->widen_bytes : 1);
        if (!wbuf) return -1;
        if (dsv4_bind_layer(c, L, lay, run, w, wbuf,
                            (size_t)lay->widen_bytes) != 0) {
            free(wbuf);
            return -1;
        }
        lay->bind_widen = wbuf;
        lay->bind_cached = *w;
        lay->bind_ready = 1;
        return 0;
    }

    return dsv4_bind_layer(c, L, lay, run, w, tr->widen, tr->widen_cap);
}

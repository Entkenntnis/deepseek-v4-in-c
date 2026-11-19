/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk_bind.c - make a layer resident and point its weight structs at it. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_bind.h"
#include "dsv4_trunk.h"
#include "dsv4_trunk_int.h"

/* Read one layer's run into `dst`, which must hold slot_bytes.
 *
 * O_DIRECT requires the offset, length AND buffer to be aligned. The packer
 * aligns every run's START, so only the length needs widening here. */
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
            /* A short read at the very end is expected when the last run is
             * not a whole number of alignment units; the packer pads for
             * exactly this, so anything else is a real failure. */
            if (r == 0 && got >= tr->lay[L].nbytes) break;
            fprintf(stderr, "dsv4_trunk: short read on layer %d at %lld\n",
                    L, (long long)(off + got));
            return -1;
        }
        got += r;
    }
    return 0;
}

/* Resolvers handed to dsv4_bind_layer_mem. The by-name one is kept for the
 * generic DSV4MemSrc contract; the trunk uses the by-index one, whose map was
 * built at open. */
typedef struct { const DSV4TrunkLayer *lay; } FindCtx;

static int trunk_find(void *ctx, const char *name,
                      int64_t *off, int64_t *nbytes, int *dtype)
{
    const DSV4TrunkLayer *lay = ((FindCtx *)ctx)->lay;
    for (int i = 0; i < lay->nt; i++)
        if (!strcmp(lay->t[i].name, name)) {
            *off = lay->t[i].off; *nbytes = lay->t[i].nbytes;
            *dtype = lay->t[i].dtype;
            return 0;
        }
    return -1;
}

static int trunk_find_idx(void *ctx, int i, const char *name,
                          int64_t *off, int64_t *nbytes, int *dtype)
{
    const DSV4TrunkLayer *lay = ((FindCtx *)ctx)->lay;
    if (i < 0 || i >= lay->nbind) return -1;
    const int ti = lay->bind_of[i];
    if (ti < 0 || ti >= lay->nt) return -1;
    *off = lay->t[ti].off; *nbytes = lay->t[ti].nbytes;
    *dtype = lay->t[ti].dtype;
    return 0;
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
        /* Round-robin, not LRU: a cyclic scan defeats LRU entirely, and
         * round-robin over the unpinned tail makes the hit rate exactly
         * npin/n_layers rather than zero. */
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
    /* A pinned run is immutable, so its binding is too. Re-planning and
     * re-widening it every token costs more than it saves. */
    if (L < tr->npin && lay->bind_ready) {
        *w = lay->bind_cached;
        return 0;
    }

    FindCtx fc = { lay };
    DSV4MemSrc src = { trunk_find, &fc, trunk_find_idx };

    if (L < tr->npin) {
        /* Size the persistent widen area exactly: the first pass is the same
         * validation every bind does; the second writes the bytes we keep. */
        size_t used = 0;
        if (dsv4_bind_layer_mem(c, L, w, run, &src, tr->widen, tr->widen_cap,
                                &used) != 0)
            return -1;
        unsigned char *wbuf = (unsigned char *)malloc(used ? used : 1);
        if (!wbuf) return -1;
        if (dsv4_bind_layer_mem(c, L, w, run, &src, wbuf, used, &used) != 0) {
            free(wbuf);
            return -1;
        }
        lay->bind_widen = wbuf;
        lay->bind_cached = *w;
        lay->bind_ready = 1;
        return 0;
    }

    return dsv4_bind_layer_mem(c, L, w, run, &src, tr->widen, tr->widen_cap,
                               NULL);
}

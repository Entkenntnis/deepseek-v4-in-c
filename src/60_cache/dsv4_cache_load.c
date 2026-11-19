/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_load.c - resolve six tensors, then read them into a slot with
 * coalescing and no copy. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"

/* RESOLVE BEFORE EVICTING. Every tensor is located and size-checked before any
 * slot is chosen, so a request that cannot be satisfied costs nothing. Reading
 * straight into the victim slot and clearing it on failure threw away a
 * perfectly good cached expert because a DIFFERENT one was missing. */
int dsv4_cache_resolve(DSV4Cache *c, int layer, int expert, DSV4Loc out[6])
{
    const DSV4Loc *loc = dsv4_st_expert(c->st, layer, expert);
    if (!loc) {
        fprintf(stderr, "dsv4_cache: expert %d of layer %d is not indexed\n",
                expert, layer);
        return -1;
    }
    for (int i = 0; i < 3; i++) {
        out[i * 2] = loc[i * 2];
        out[i * 2 + 1] = loc[i * 2 + 1];
        if (out[i * 2].nbytes != c->len_w[i]) {
            fprintf(stderr, "dsv4_cache: layer %d expert %d w%d has %lld bytes, "
                            "expected %lld\n", layer, expert, i + 1,
                    (long long)out[i * 2].nbytes, (long long)c->len_w[i]);
            return -1;
        }
        if (out[i * 2 + 1].nbytes != c->len_s[i]) {
            fprintf(stderr, "dsv4_cache: layer %d expert %d w%d scale has "
                            "%lld bytes, expected %lld\n", layer, expert, i + 1,
                    (long long)out[i * 2 + 1].nbytes, (long long)c->len_s[i]);
            return -1;
        }
    }
    return 0;
}

/* Read six already-resolved tensors into a slot, coalescing adjacent ones. The
 * six arrive as two contiguous runs and the slot is laid out to match, so each
 * run is one pread straight into place: coalescing into two runs is much faster
 * than six separate reads. The cache stats from `make e2e` show the
 * effect. A short read leaves the slot
 * unusable, so the caller must clear the key. */
int dsv4_cache_load(DSV4Cache *c, DSV4Slot *s, const DSV4Loc t[6],
                    unsigned char *bounce, uint64_t *bytes_read)
{
    /* (file offset, length, shard, which tensor), in slot order: s,s,s,w,w,w. */
    struct { int64_t foff, len; int shard, idx; } r[6];
    for (int i = 0; i < 3; i++) {
        r[i].foff     = t[i * 2 + 1].off;
        r[i].len      = t[i * 2 + 1].nbytes;
        r[i].shard    = t[i * 2 + 1].shard;
        r[i].idx      = i;
        r[3 + i].foff = t[i * 2].off;
        r[3 + i].len  = t[i * 2].nbytes;
        r[3 + i].shard = t[i * 2].shard;
        r[3 + i].idx  = 3 + i;
    }
    for (int i = 1; i < 6; i++)              /* six elements: insertion sort */
        for (int j = i; j > 0 && r[j].foff < r[j - 1].foff; j--) {
            const typeof(r[0]) tmp = r[j]; r[j] = r[j - 1]; r[j - 1] = tmp;
        }

    int64_t cursor = 0;
    int i = 0;
    while (i < 6) {
        int j = i;
        /* Extend while the FILE stays contiguous. The slot no longer has to
         * agree: the run is placed as a unit. */
        while (j + 1 < 6
               && r[j + 1].shard == r[i].shard
               && r[j + 1].foff  == r[j].foff + r[j].len)
            j++;

        int64_t len = 0;
        for (int k = i; k <= j; k++) len += r[k].len;

        /* PLACE THE RUN SO O_DIRECT CAN LAND IT WITHOUT A COPY.
         *
         * The read widens to the enclosing 4096 window and reports the payload
         * `pad` bytes into the buffer, which must itself be 4096-aligned. Choose
         * the slot offset with the SAME residue as the file offset, and the
         * window lands straight in the slot. The window starts `pad` bytes
         * BEFORE the payload, so the payload must sit at least `pad` past the
         * previous run, or this read clobbers its tail. That bug was silent: the
         * hit rate went UP. */
        const int64_t pad  = r[i].foff & (int64_t)4095;
        const int64_t base = cursor + pad;
        const int64_t soff = base + ((pad - base) & (int64_t)4095);
        const int64_t wlen = (pad + len + 4095) & ~(int64_t)4095;
        const int use_slot = (soff - pad + wlen) <= c->slot_bytes;
        int64_t place;

        if (c->direct && use_slot) {
            /* The fast path: no copy at all. */
            int64_t poff = 0;
            const int64_t n = dsv4_st_read_aligned(
                c->st, r[i].shard, r[i].foff, len,
                s->bytes + soff - pad, c->slot_bytes - (soff - pad), &poff);
            if (n != len || poff != pad) return -1;
            place = soff;
        } else if (c->direct && len <= c->bounce_cap - 2 * 4096) {
            /* Placement did not fit; load into the staging buffer and copy.
             * Kept so a checkpoint that scatters these into more runs than the
             * slack allows still loads correctly. */
            int64_t poff = 0;
            const int64_t n = dsv4_st_read_aligned(c->st, r[i].shard, r[i].foff,
                                                   len, bounce, c->bounce_cap,
                                                   &poff);
            if (n != len) return -1;
            if (cursor + len > c->slot_bytes) return -1;
            memcpy(s->bytes + cursor, bounce + poff, (size_t)len);
            place = cursor;
        } else {
            if (cursor + len > c->slot_bytes) return -1;
            place = cursor;
            int64_t got = 0;
            while (got < len) {
                const ssize_t n = pread(c->st->fd[r[i].shard],
                                        s->bytes + place + got,
                                        (size_t)(len - got),
                                        (off_t)(r[i].foff + got));
                if (n <= 0) return -1;
                got += n;
            }
        }

        /* Record where each tensor of this run actually landed. */
        int64_t at = place;
        for (int k = i; k <= j; k++) { s->toff[r[k].idx] = at; at += r[k].len; }
        cursor = place + len;

        *bytes_read += (uint64_t)len;
        i = j + 1;
    }

    /* The layout is per-load, so rebound here. Touches only this slot, which is
     * what makes it safe inside begin()/end()'s parallel read pass. */
    dsv4_cache_bind_slot(c, s);
    return 0;
}

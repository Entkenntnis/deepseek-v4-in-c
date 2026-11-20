/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk.c - open, load and bind the dense trunk. */
#define _GNU_SOURCE

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_trunk.h"
#include "dsv4_bind.h"
#include "dsv4_util.h"

static int dsv4_trunk_read_run(DSV4Trunk *tr, int L, unsigned char *dst)
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

    int64_t data_bytes = 0, widen_total = 0;
    for (int L = 0; L < tr->n_layers; L++) {
        DSV4TrunkLayer *lay = &tr->lay[L];
        if (lay->nbytes < 0 || lay->widen_bytes < 0) {
            fprintf(stderr, "dsv4_trunk: layer %d has a negative size\n", L);
            dsv4_trunk_close(tr);
            return -1;
        }
        lay->run_bytes = (size_t)((lay->nbytes + DSV4_TRUNK_ALIGN - 1)
                                  / DSV4_TRUNK_ALIGN * DSV4_TRUNK_ALIGN);
        data_bytes += (int64_t)lay->run_bytes;
        widen_total += lay->widen_bytes;
    }

    if (data_bytes + widen_total > budget_bytes) {
        fprintf(stderr, "dsv4_trunk: the whole trunk needs %.2f GiB "
                        "(%.2f GiB weights + %.2f GiB widened vectors), "
                        "but the budget is %.2f GiB\n",
                (double)(data_bytes + widen_total) / 1073741824.0,
                (double)data_bytes / 1073741824.0,
                (double)widen_total / 1073741824.0,
                (double)budget_bytes / 1073741824.0);
        dsv4_trunk_close(tr);
        return -1;
    }

    if (data_bytes > 0 &&
        posix_memalign((void **)&tr->data, DSV4_TRUNK_ALIGN,
                       (size_t)data_bytes) != 0) {
        dsv4_trunk_close(tr);
        return -1;
    }
    if (data_bytes > 0) dsv4_hugepages(tr->data, (size_t)data_bytes);

    int64_t at = 0;
    for (int L = 0; L < tr->n_layers; L++) {
        DSV4TrunkLayer *lay = &tr->lay[L];
        if (lay->run_bytes == 0) continue;
        lay->run = tr->data + at;
        if (dsv4_trunk_read_run(tr, L, lay->run) != 0) {
            dsv4_trunk_close(tr);
            return -1;
        }
        at += (int64_t)lay->run_bytes;
    }

    for (int L = 0; L < tr->n_layers; L++) {
        DSV4TrunkLayer *lay = &tr->lay[L];
        if (lay->nt == 0) continue;
        if (!lay->run) {
            fprintf(stderr, "dsv4_trunk: layer %d has tensors but no data\n", L);
            dsv4_trunk_close(tr);
            return -1;
        }
        lay->bind_widen = (unsigned char *)malloc(
            lay->widen_bytes ? (size_t)lay->widen_bytes : 1);
        if (!lay->bind_widen) {
            dsv4_trunk_close(tr);
            return -1;
        }
        if (dsv4_bind_layer(c, L, lay, lay->run, &lay->bind_cached,
                            lay->bind_widen, (size_t)lay->widen_bytes) != 0) {
            dsv4_trunk_close(tr);
            return -1;
        }
        lay->bind_ready = 1;
    }

    tr->resident_bytes = data_bytes + widen_total;
    return 0;
}

int64_t dsv4_trunk_resident_bytes(const DSV4Trunk *tr)
{
    return tr->resident_bytes;
}

void dsv4_trunk_close(DSV4Trunk *tr)
{
    if (!tr) return;
    if (tr->fd >= 0) close(tr->fd);
    if (tr->bfd >= 0) close(tr->bfd);
    free(tr->data);
    if (tr->lay) for (int L = 0; L < tr->n_layers; L++) {
        free(tr->lay[L].t);
        free(tr->lay[L].bind_widen);
    }
    free(tr->lay);
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
    tr->bfd = -1;
}

const DSV4LayerW *dsv4_trunk_weights(const DSV4Trunk *tr, int L)
{
    if (!tr || L < 0 || L >= tr->n_layers || !tr->lay[L].bind_ready)
        return NULL;
    return &tr->lay[L].bind_cached;
}

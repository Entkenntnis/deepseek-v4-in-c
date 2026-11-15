/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st_shard.c - index one safetensors shard. */
#define _GNU_SOURCE            /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "dsv4_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before libc */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_st.h"
#include "dsv4_st_int.h"

int dsv4_st_push(DSV4StBuild *b, DSV4St *s, const char *name, size_t nlen,
                 const DSV4Tensor *src)
{
    if (b->n == b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 4096;
        DSV4Tensor *nt = (DSV4Tensor *)realloc(b->t, nc * sizeof *nt);
        size_t   *no = (size_t   *)realloc(b->noff, nc * sizeof *no);
        if (!nt || !no) {
            b->t = nt ? nt : b->t;
            b->noff = no ? no : b->noff;
            return -1;
        }
        b->t = nt; b->noff = no; b->cap = nc;
    }
    if (s->strlen_ + nlen + 1 > s->strcap) {
        size_t nc = s->strcap ? s->strcap : 1 << 20;
        while (s->strlen_ + nlen + 1 > nc) nc *= 2;
        char *np = (char *)realloc(s->strpool, nc);
        if (!np) return -1;
        s->strpool = np; s->strcap = nc;
    }
    b->noff[b->n] = s->strlen_;
    memcpy(s->strpool + s->strlen_, name, nlen + 1);
    s->strlen_ += nlen + 1;
    b->t[b->n] = *src;
    b->n++;
    return 0;
}

int dsv4_st_scan_shard(DSV4St *s, DSV4StBuild *b, int shard, const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "dsv4_st: cannot open %s\n", path); return -1; }

    unsigned char lenbuf[8];
    if (pread(fd, lenbuf, 8, 0) != 8) {
        fprintf(stderr, "dsv4_st: %s is too short for a header length\n", path);
        close(fd); return -1;
    }
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | lenbuf[i];   /* little endian */

    off_t fsize = lseek(fd, 0, SEEK_END);
    if (hlen == 0 || (uint64_t)fsize < 8 + hlen) {
        fprintf(stderr, "dsv4_st: %s header length %llu is impossible (file %lld bytes)\n",
                path, (unsigned long long)hlen, (long long)fsize);
        close(fd); return -1;
    }

    char *json = (char *)malloc(hlen + 1);
    if (!json) { close(fd); return -1; }
    ssize_t got = 0;
    while ((uint64_t)got < hlen) {
        ssize_t r = pread(fd, json + got, hlen - got, 8 + got);
        if (r <= 0) break;
        got += r;
    }
    if ((uint64_t)got != hlen) {
        fprintf(stderr, "dsv4_st: short read of %s header\n", path);
        free(json); close(fd); return -1;
    }
    json[hlen] = '\0';

    const int64_t base = (int64_t)(8 + hlen);   /* data_offsets are relative to this */

    DSV4Scan sc = { json, json + hlen };
    if (!dsv4_st_lit(&sc, '{')) {
        fprintf(stderr, "dsv4_st: %s header is not a JSON object\n", path);
        goto bad;
    }

    static char name[512];
    int first = 1, ntensor = 0;
    int64_t maxend = 0;
    for (;;) {
        dsv4_st_ws(&sc);
        if (sc.p < sc.end && *sc.p == '}') { sc.p++; break; }
        if (!first && !dsv4_st_lit(&sc, ',')) {
            fprintf(stderr, "dsv4_st: %s expected ','\n", path);
            goto bad;
        }
        first = 0;
        dsv4_st_ws(&sc);
        if (sc.p < sc.end && *sc.p == '}') { sc.p++; break; }   /* trailing comma */

        size_t nlen = 0;
        if (!dsv4_st_str(&sc, name, sizeof name, &nlen)) {
            fprintf(stderr, "dsv4_st: %s bad tensor name near byte %ld\n",
                    path, (long)(sc.p - json));
            goto bad;
        }
        if (!dsv4_st_lit(&sc, ':')) {
            fprintf(stderr, "dsv4_st: %s expected ':'\n", path);
            goto bad;
        }

        if (!strcmp(name, "__metadata__")) {
            if (!dsv4_st_skip_value(&sc)) goto bad;
            continue;
        }

        DSV4Tensor t; memset(&t, 0, sizeof t);
        t.shard = shard;
        if (dsv4_st_parse_entry(&sc, path, name, base, &t) != 0) goto bad;

        const int64_t o1 = t.off - base + t.nbytes;
        if (base + o1 > fsize) {
            fprintf(stderr, "dsv4_st: %s: %s ends past EOF\n", path, name);
            goto bad;
        }
        if (o1 > maxend) maxend = o1;

        if (dsv4_st_push(b, s, name, nlen, &t) != 0) {
            fprintf(stderr, "dsv4_st: out of memory\n");
            goto bad;
        }
        ntensor++;
    }

    if (base + maxend != fsize)
        fprintf(stderr, "dsv4_st: note: %s has %lld trailing bytes after the last tensor\n",
                path, (long long)(fsize - base - maxend));

    free(json);
    s->fd[shard] = fd;
    /* A second descriptor on the same file, for streamed expert reads that must
     * not go through the page cache. Optional: if the filesystem refuses
     * O_DIRECT the reader falls back to fd[]. */
    if (s->dfd) {
        s->dfd[shard] = open(path, O_RDONLY | O_DIRECT);
        dsv4_set_direct(s->dfd[shard]);   /* advisory on Linux; failure is fine */
    }
    return ntensor;
bad:
    free(json);
    close(fd);
    return -1;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st_read.c - buffered, aligned and widening reads. */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "dsv4_st.h"

#define ST_ALIGN 4096

int64_t dsv4_st_read(const DSV4St *s, int shard, int64_t off, int64_t nbytes,
                     void *buf)
{
    if (!s || shard < 0 || shard >= s->nshard || s->fd[shard] < 0)
        return 0;
    int64_t got = 0;
    while (got < nbytes) {
        const ssize_t n = pread(s->fd[shard], (char *)buf + got,
                                (size_t)(nbytes - got), (off_t)(off + got));
        if (n <= 0) return got;
        got += n;
    }
    return got;
}

int64_t dsv4_st_read_aligned(const DSV4St *s, int shard, int64_t off,
                             int64_t nbytes, void *buf, int64_t bufcap,
                             int64_t *payload_off)
{
    if (!s || shard < 0 || shard >= s->nshard) return 0;
    const int dfd = s->dfd ? s->dfd[shard] : -1;

    if (dfd < 0) {
        if (payload_off) *payload_off = 0;
        return dsv4_st_read(s, shard, off, nbytes, buf);
    }

    const int64_t lo = off & ~(int64_t)(ST_ALIGN - 1);
    const int64_t hi = (off + nbytes + ST_ALIGN - 1) & ~(int64_t)(ST_ALIGN - 1);
    const int64_t len = hi - lo;
    const int64_t pad = off - lo;
    if (len > bufcap) return 0;
    if (payload_off) *payload_off = pad;

    int64_t got = 0;
    while (got < len) {
        const ssize_t r = pread(dfd, (char *)buf + got, (size_t)(len - got),
                                (off_t)(lo + got));
        if (r <= 0) break;
        got += r;
    }
    if (got >= pad + nbytes) return nbytes;
    return got > pad ? got - pad : 0;
}

int64_t dsv4_st_read_f32(const DSV4St *s, int shard, int64_t off,
                         int64_t nbytes, int dtype, float *out)
{
    const int esz = dsv4_st_elemsize((DSV4Dtype)dtype);
    if (esz <= 0 || nbytes % esz) return 0;
    const int64_t n = nbytes / esz;

    if (dtype == DSV4_DT_F32) {
        return dsv4_st_read(s, shard, off, nbytes, out) == nbytes ? n : 0;
    }
    if (dtype == DSV4_DT_BF16) {
        uint16_t *tmp = (uint16_t *)malloc((size_t)nbytes);
        if (!tmp) return 0;
        if (dsv4_st_read(s, shard, off, nbytes, tmp) != nbytes ||
            dsv4_widen_f32(out, tmp, nbytes, dtype) != 0) {
            free(tmp);
            return 0;
        }
        free(tmp);
        return n;
    }
    fprintf(stderr, "dsv4_st: cannot widen dtype %d\n", dtype);
    return 0;
}

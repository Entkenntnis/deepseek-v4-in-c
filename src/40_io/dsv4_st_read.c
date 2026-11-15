/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st_read.c - buffered, aligned and widening reads. */
#define _GNU_SOURCE            /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "dsv4_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before libc */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "dsv4_st.h"

int64_t dsv4_st_read_aligned(const DSV4St *s, int shard, int64_t off,
                             int64_t nbytes, void *buf, int64_t bufcap,
                             int64_t *payload_off)
{
    if (shard < 0 || shard >= s->nshard) return 0;
    const int dfd = s->dfd ? s->dfd[shard] : -1;

    if (dfd < 0) {                      /* no O_DIRECT: plain buffered read */
        if (payload_off) *payload_off = 0;
        if (bufcap < nbytes) return 0;
        int64_t got = 0;
        while (got < nbytes) {
            ssize_t r = pread(s->fd[shard], (char *)buf + got,
                              (size_t)(nbytes - got), (off_t)(off + got));
            if (r <= 0) return got;
            got += r;
        }
        return got;
    }

    /* Widen outward to the enclosing aligned window. */
    const int64_t lo = off & ~(int64_t)(DSV4_ST_ALIGN - 1);
    const int64_t hi = (off + nbytes + DSV4_ST_ALIGN - 1)
                     & ~(int64_t)(DSV4_ST_ALIGN - 1);
    const int64_t len = hi - lo;
    const int64_t pad = off - lo;
    if (len > bufcap) return 0;
    if (payload_off) *payload_off = pad;

    int64_t got = 0;
    while (got < len) {
        ssize_t r = pread(dfd, (char *)buf + got, (size_t)(len - got),
                          (off_t)(lo + got));
        if (r <= 0) {
            /* The final window of a shard can extend past EOF, which is a short
             * read rather than an error. Accept it once the payload is
             * covered. */
            break;
        }
        got += r;
    }
    return got >= pad + nbytes ? nbytes : (got > pad ? got - pad : 0);
}

int64_t dsv4_st_read(const DSV4St *s, const DSV4Tensor *t, void *buf)
{
    /* One coalesced pread, looped only because the kernel may return short.
     * This is the call the streaming tier makes per expert. */
    int64_t got = 0;
    while (got < t->nbytes) {
        ssize_t r = pread(s->fd[t->shard], (char *)buf + got,
                          (size_t)(t->nbytes - got), (off_t)(t->off + got));
        if (r <= 0) {
            fprintf(stderr, "dsv4_st: short read on %s at +%lld\n",
                    t->name, (long long)got);
            return got;
        }
        got += r;
    }
    return got;
}

/* Widening is chunked: embed_tokens and lm_head are 2.35 GB each as BF16, and
 * a full-size staging buffer on top of the destination is enough to get the
 * process OOM-killed. A few megabytes at a time caps the transient. */
#define DSV4_WIDEN_CHUNK (4 << 20)

int64_t dsv4_st_read_f32(const DSV4St *s, const DSV4Tensor *t, float *out)
{
    const int64_t n = dsv4_st_numel(t);
    if (t->dtype == DSV4_DT_F32) return dsv4_st_read(s, t, out) / 4;

    /* REFUSE the dtypes this function cannot widen, rather than falling through
     * and returning n with `out` untouched. FP8/FP4 need their paired .scale
     * tensor, which this signature cannot reach; I64 is an index table. */
    switch (t->dtype) {
    case DSV4_DT_F8_E4M3:
    case DSV4_DT_F8_E8M0:
    case DSV4_DT_I8:
    case DSV4_DT_I64:
        fprintf(stderr,
                "dsv4_st: %s is %s; dsv4_st_read_f32 cannot widen it.\n"
                "  FP8/FP4 need their paired .scale tensor, and I64 is an index\n"
                "  table. Read the raw bytes with dsv4_st_read() and dequantise\n"
                "  with the scale in hand.\n",
                t->name,
                t->dtype == DSV4_DT_F8_E4M3 ? "F8_E4M3" :
                t->dtype == DSV4_DT_F8_E8M0 ? "F8_E8M0" :
                t->dtype == DSV4_DT_I8      ? "I8 (packed FP4)" : "I64");
        return 0;
    default:
        break;
    }

    const int esz = dsv4_st_elemsize(t->dtype);
    if (esz <= 0) return 0;
    /* A whole number of elements per chunk, so none straddles a boundary. */
    const int64_t chunk_elems = DSV4_WIDEN_CHUNK / esz;

    void *raw = malloc((size_t)chunk_elems * esz);
    if (!raw) return 0;

    int64_t done = 0;
    while (done < n) {
        const int64_t take = (n - done < chunk_elems) ? (n - done) : chunk_elems;
        const int64_t want = take * esz;
        int64_t got = 0;
        while (got < want) {
            ssize_t r = pread(s->fd[t->shard], (char *)raw + got,
                              (size_t)(want - got),
                              (off_t)(t->off + done * esz + got));
            if (r <= 0) {
                fprintf(stderr, "dsv4_st: short read widening %s at element %lld\n",
                        t->name, (long long)done);
                free(raw);
                return done;
            }
            got += r;
        }

        float *o = out + done;
        if (t->dtype == DSV4_DT_BF16) {
            const uint16_t *p = (const uint16_t *)raw;
            for (int64_t i = 0; i < take; i++) o[i] = dsv4_bf16_to_f32(p[i]);
        } else if (t->dtype == DSV4_DT_U8) {
            const unsigned char *p = (const unsigned char *)raw;
            for (int64_t i = 0; i < take; i++) o[i] = (float)p[i];
        } else if (t->dtype == DSV4_DT_F16) {
            const uint16_t *p = (const uint16_t *)raw;
            for (int64_t i = 0; i < take; i++) {
                uint16_t h = p[i];
                uint32_t sign = (uint32_t)(h & 0x8000) << 16;
                uint32_t exp  = (h >> 10) & 0x1F, man = h & 0x3FF;
                union { uint32_t u; float f; } v;
                if (exp == 0) {
                    if (man == 0) v.u = sign;
                    else {                              /* subnormal: renormalise */
                        int sh = 0;
                        while (!(man & 0x400)) { man <<= 1; sh++; }
                        man &= 0x3FF;
                        v.u = sign | ((uint32_t)(127 - 15 - sh + 1) << 23) | (man << 13);
                    }
                } else if (exp == 31) v.u = sign | 0x7F800000u | (man << 13);
                else v.u = sign | ((exp - 15 + 127) << 23) | (man << 13);
                o[i] = v.f;
            }
        }
        done += take;
    }
    free(raw);
    return n;
}

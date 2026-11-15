/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st_int.h - seams between the safetensors reader files. Not public. */
#ifndef DSV4_ST_INT_H
#define DSV4_ST_INT_H

#include <stddef.h>
#include <stdint.h>

#include "dsv4_st.h"

/* A cursor over one header's JSON text. */
typedef struct { const char *p, *end; } DSV4Scan;

DSV4Dtype dsv4_st_dtype_of(const char *s, size_t n);
void dsv4_st_ws(DSV4Scan *s);
int  dsv4_st_lit(DSV4Scan *s, char c);
int  dsv4_st_str(DSV4Scan *s, char *out, size_t cap, size_t *len);
int  dsv4_st_i64(DSV4Scan *s, int64_t *v);
int  dsv4_st_skip_value(DSV4Scan *s);

/* Growth state for one open() call. noff are name offsets into strpool,
 * resolved to pointers only once the pool has stopped moving. */
typedef struct {
    DSV4Tensor *t;
    size_t      n, cap;
    size_t     *noff;
} DSV4StBuild;

int dsv4_st_push(DSV4StBuild *b, DSV4St *s, const char *name, size_t nlen,
                 const DSV4Tensor *src);
int dsv4_st_parse_entry(DSV4Scan *sc, const char *path, const char *name,
                        int64_t base, DSV4Tensor *t);
int dsv4_st_scan_shard(DSV4St *s, DSV4StBuild *b, int shard, const char *path);

#endif /* DSV4_ST_INT_H */

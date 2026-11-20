/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk.h - load the dense trunk fully resident.
 *
 * The packer lays each layer out as ONE contiguous run in trunk.bin and writes
 * trunk.idx with a slot id for every tensor. The reader allocates one arena for
 * every layer run and binds each layer once; there is no streaming ring and no
 * partially pinned prefix.
 */
#ifndef DSV4_TRUNK_H
#define DSV4_TRUNK_H

#include "dsv4.h"

#define DSV4_TRUNK_ALIGN 4096

/* Layer slot ids used outside the binder. Keep in step with the packer. */
#define DSV4_TRUNK_SLOT_TID2EID 41

typedef struct {
    int32_t slot;
    int32_t dtype;
    int32_t rows, cols;
    int32_t wdt, blk_r, blk_c;
    int64_t off;              /* byte offset WITHIN the layer run */
    int64_t nbytes;
} DSV4TrunkTensor;

typedef struct DSV4TrunkLayer {
    int64_t          file_off;   /* offset in trunk.bin */
    int64_t          nbytes;
    int64_t          widen_bytes;
    DSV4TrunkTensor *t;
    int              nt;
    unsigned char   *run;        /* resident layer run */
    size_t           run_bytes;  /* aligned allocation size */
    DSV4LayerW       bind_cached;
    unsigned char   *bind_widen;
    int              bind_ready;
} DSV4TrunkLayer;

typedef struct DSV4Trunk {
    int              fd;
    int              bfd;       /* buffered, for small unaligned reads */
    int              direct;
    int              n_layers;
    DSV4TrunkLayer  *lay;

    unsigned char   *data;      /* one arena for every layer run */
    int64_t          resident_bytes;
} DSV4Trunk;

int  dsv4_trunk_open(DSV4Trunk *tr, const char *dir, const DSV4Cfg *c,
                     int64_t budget_bytes);
void dsv4_trunk_close(DSV4Trunk *tr);
int64_t dsv4_trunk_resident_bytes(const DSV4Trunk *tr);
int  dsv4_trunk_bind(DSV4Trunk *tr, int L, DSV4LayerW *w);

/* One tensor of a layer by slot, or NULL. */
const DSV4TrunkTensor *dsv4_trunk_slot(const DSV4Trunk *tr, int L, int slot);

/* Reader used by bind; exposed for the trunk implementation files. */
int dsv4_trunk_load_manifest(DSV4Trunk *tr, const char *dir);

#endif /* DSV4_TRUNK_H */

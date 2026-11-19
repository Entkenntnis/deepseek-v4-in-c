/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st.h - checkpoint shards and the layout indexes written by the packer.
 *
 * The C engine no longer parses safetensors JSON. tools/pack_trunk.py reads
 * every header once and writes:
 *
 *   trunk.idx    dense layer runs (slot order)
 *   experts.idx  one fixed record per routed-expert tensor
 *   model.idx    embed, final norm, head and the head's mHC
 *
 * This header is only about opening shards and reading bytes. The layer and
 * model slot tables live in the loader.
 */
#ifndef DSV4_ST_H
#define DSV4_ST_H

#include <stddef.h>
#include <stdint.h>

/* Storage dtypes, as numbered in the packer. I8 is PACKED FP4: two values per
 * byte, exactly as the checkpoint spells it. F8_E8M0 is an exponent-only scale
 * byte b meaning 2^(b-127). */
typedef enum {
    DSV4_DT_UNKNOWN = 0,
    DSV4_DT_U8 = 1,
    DSV4_DT_BF16 = 2,
    DSV4_DT_F16 = 3,
    DSV4_DT_F32 = 4,
    DSV4_DT_F8_E4M3 = 5,
    DSV4_DT_F8_E8M0 = 6,
    DSV4_DT_I8 = 7,
    DSV4_DT_I64 = 8
} DSV4Dtype;

/* Order of the six tensors in experts.idx. Must match tools/pack_trunk.py. */
enum {
    DSV4_EXP_W1W = 0,
    DSV4_EXP_W1S = 1,
    DSV4_EXP_W2W = 2,
    DSV4_EXP_W2S = 3,
    DSV4_EXP_W3W = 4,
    DSV4_EXP_W3S = 5,
    DSV4_EXP_TENSORS = 6
};

/* A byte location in one shard. The packer writes 24-byte records
 * (int32 shard, int32 pad, int64 off, int64 nbytes). */
typedef struct {
    int32_t shard;
    int32_t pad;
    int64_t off;
    int64_t nbytes;
} DSV4Loc;

/* model.idx record: int32 slot, int32 shard, int64 off, int64 nbytes,
 * int32 dtype, int32 pad. */
typedef struct {
    int32_t slot;
    int32_t shard;
    int64_t off;
    int64_t nbytes;
    int32_t dtype;
    int32_t pad;
} DSV4ModelLoc;

typedef struct {
    int *fd;                 /* one descriptor per shard */
    int *dfd;                /* the same shards opened O_DIRECT, or -1 */
    int  nshard;

    DSV4Loc *exp;            /* [n_layers][n_experts][DSV4_EXP_TENSORS] */
    int      n_layers, n_experts;

    DSV4ModelLoc *model;
    int           nmodel;
} DSV4St;

/* Open every *.safetensors in model_dir and load the three indexes from
 * index_dir (the trunk directory). Returns 0 on success. */
int  dsv4_st_open(DSV4St *s, const char *model_dir, const char *index_dir);
void dsv4_st_close(DSV4St *s);

/* The six locations of one routed expert, in DSV4_EXP_* order. */
const DSV4Loc *dsv4_st_expert(const DSV4St *s, int layer, int expert);

/* One model-level tensor by slot, or NULL. */
const DSV4ModelLoc *dsv4_st_model(const DSV4St *s, int slot);

/* Raw bytes from one shard, looped only because pread may return short. */
int64_t dsv4_st_read(const DSV4St *s, int shard, int64_t off, int64_t nbytes,
                     void *buf);

/* Buffered read that handles O_DIRECT alignment. Returns payload bytes and
 * sets *payload_off to the payload start inside buf. Falls back to a plain
 * read when O_DIRECT is unavailable. */
int64_t dsv4_st_read_aligned(const DSV4St *s, int shard, int64_t off,
                             int64_t nbytes, void *buf, int64_t bufcap,
                             int64_t *payload_off);

/* Widen a whole tensor to fp32. Handles F32, BF16 and F16. Refuses packed
 * formats (FP8/FP4/I8/I64), which need their scale partner or are indices. */
int64_t dsv4_st_read_f32(const DSV4St *s, int shard, int64_t off,
                         int64_t nbytes, int dtype, float *out);

int dsv4_st_elemsize(DSV4Dtype d);

static inline float dsv4_bf16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

#endif /* DSV4_ST_H */

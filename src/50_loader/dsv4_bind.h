/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind.h - bind checkpoint tensors into the engine's weight structs.
 *
 * ONE plan_layer() declares what a layer needs, and both binding paths -- from
 * safetensors shards and from a streamed trunk run -- walk that same plan. Two
 * independent name lists would drift, and the drift would show up as a
 * transposed or truncated matrix rather than as an error.
 */
#ifndef DSV4_BIND_H
#define DSV4_BIND_H

#include "dsv4.h"
#include "dsv4_st.h"

typedef struct {
    void     *blob;      /* one allocation holding embed/head/norm/hc_head */
    DSV4ModelW w;
} DSV4ModelBind;

/* Model-level weights: embed, final norm, head, and the head's mHC. embed and
 * head are distinct tensors (tie_word_embeddings is false). Pass want_head = 0
 * when only the trunk is exercised. */
int  dsv4_bind_model(const DSV4St *s, const DSV4Cfg *c, int want_head,
                     DSV4ModelBind *m);
void dsv4_bind_model_free(DSV4ModelBind *m);

/* Bytes dsv4_bind_model() will allocate, without reading any. The budget
 * planner needs this: --budget covers the trunk and expert cache but NOT
 * embed/head, so `auto` must subtract them before promising a spare. */
int64_t dsv4_bind_model_bytes(const DSV4St *s, const DSV4Cfg *c, int want_head);

/* ---- binding from a memory buffer, for the streaming trunk ----------------
 *
 * The trunk holds a layer's tensors as a verbatim copy of their bytes. Binding
 * from that buffer MUST use exactly the same tensor names, shapes and
 * narrow/wide decisions as binding from the shards, so it walks the SAME
 * plan_layer(); only the source of the bytes differs. Most tensors need no copy
 * at all: a narrow matrix is pointed at directly, in place. Only the small
 * vectors that kernels read elementwise are widened into `widen`. Every widened
 * tensor is BF16 or F32.
 */
typedef struct {
    int (*find)(void *ctx, const char *name,
                int64_t *off, int64_t *nbytes, int *dtype);
    void *ctx;
    /* Optional fast path: a source can resolve the whole plan ONCE and then
     * answer by request index instead of by name. When set, find_idx is called
     * instead of find(); `i` is the request's position in
     * dsv4_bind_layer_plan()'s order. Both must describe the same tensor. */
    int (*find_idx)(void *ctx, int i, const char *name,
                    int64_t *off, int64_t *nbytes, int *dtype);
} DSV4MemSrc;

/* Walk the requests plan_layer() will make for layer `L`, in order. `emit`
 * (when non-NULL) is called once per request with its index and name. Returns
 * the request count, or -1 if emit rejects one. This is the contract that lets
 * a source pre-resolve names without duplicating the plan. */
int dsv4_bind_layer_plan(const DSV4Cfg *c, int L,
                         int (*emit)(void *ctx, int i, const char *name),
                         void *ctx);

int dsv4_bind_layer_mem(const DSV4Cfg *c, int layer, DSV4LayerW *w,
                        const unsigned char *run, const DSV4MemSrc *src,
                        unsigned char *widen, size_t widen_cap,
                        size_t *widen_used);

/* Upper bound on the widen area one layer needs, so a trunk slot can be sized
 * once for every layer kind. Uses the WIDEST layer, because slots are uniform. */
size_t dsv4_bind_widen_bytes(const DSV4Cfg *c);

#endif /* DSV4_BIND_H */

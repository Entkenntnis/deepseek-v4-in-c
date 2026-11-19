/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind.h - bind checkpoint bytes into the engine's weight structs.
 *
 * The packer writes every tensor's slot id. This header's tables are the only
 * place that maps a slot to a field in DSV4LayerW or DSV4ModelW. There is no
 * name formatting and no runtime plan: a missing or unknown slot is an error.
 */
#ifndef DSV4_BIND_H
#define DSV4_BIND_H

#include "dsv4.h"
#include "dsv4_st.h"

struct DSV4TrunkLayer;

typedef struct {
    void     *blob;      /* one allocation holding embed/head/norm/hc_head */
    DSV4ModelW w;
} DSV4ModelBind;

/* Model-level weights: embed, final norm, head, and the head's mHC. Pass
 * want_head = 0 when only the trunk is exercised. */
int  dsv4_bind_model(const DSV4St *s, const DSV4Cfg *c, int want_head,
                     DSV4ModelBind *m);
void dsv4_bind_model_free(DSV4ModelBind *m);

/* Bytes dsv4_bind_model() will allocate, without reading any. */
int64_t dsv4_bind_model_bytes(const DSV4St *s, const DSV4Cfg *c, int want_head);

/* Bind one layer's run. `widen` holds the widened small vectors; the packer
 * recorded its exact size in `lay->widen_bytes`. Returns 0 on success. */
int dsv4_bind_layer(const DSV4Cfg *c, int layer,
                    const struct DSV4TrunkLayer *lay,
                    const unsigned char *run, DSV4LayerW *w,
                    unsigned char *widen, size_t widen_cap);

#endif /* DSV4_BIND_H */

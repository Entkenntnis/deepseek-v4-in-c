/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_geom.c - expert slot geometry and pointer binding. */
#include <string.h>

#include "dsv4_cache.h"
#include "dsv4_cache_int.h"

/* An expert's six tensors, in the order they are laid out inside a slot.
 * w2 is the down projection and has the transposed shape, which is why its
 * lengths are computed separately. The order mirrors DISK order (scales, then
 * weights) so each run lands in one pread; see README.md. */
int dsv4_cache_expert_geometry(DSV4Cache *c, const DSV4Cfg *cfg)
{
    const int64_t inter = cfg->moe_inter, hid = cfg->hidden;
    /* FP4: stored bytes are half the logical columns; scales are 1x32 on the
     * reduce dimension. Verified against the checkpoint: w1.weight I8
     * [2048,2048] with w1.scale F8_E8M0 [2048,128]. */
    const int64_t wb[3] = { inter * (hid / 2), hid * (inter / 2),
                            inter * (hid / 2) };
    const int64_t sb[3] = { inter * (hid / 32), hid * (inter / 32),
                            inter * (hid / 32) };

    int64_t off = 0;
    for (int i = 0; i < 3; i++) { c->off_s[i] = off; c->len_s[i] = sb[i]; off += sb[i]; }
    for (int i = 0; i < 3; i++) { c->off_w[i] = off; c->len_w[i] = wb[i]; off += wb[i]; }
    c->expert_bytes = off;
    c->slot_bytes   = off + DSV4_SLOT_SLACK;
    return 0;
}

/* Point a slot's DSV4ExpertW at its own bytes. Done once per load, not per use. */
void dsv4_cache_bind_slot(DSV4Cache *c, DSV4Slot *s)
{
    const DSV4Cfg *cfg = c->cfg;
    DSV4QMat *m[3] = { &s->w.w1, &s->w.w2, &s->w.w3 };
    const int64_t rows[3] = { cfg->moe_inter, cfg->hidden,    cfg->moe_inter };
    const int64_t cols[3] = { cfg->hidden,    cfg->moe_inter, cfg->hidden    };

    for (int i = 0; i < 3; i++) {
        memset(m[i], 0, sizeof *m[i]);
        m[i]->wdt   = DSV4_WFP4;
        m[i]->rows  = (int)rows[i];
        m[i]->cols  = (int)cols[i];
        m[i]->blk_r = 1;
        m[i]->blk_c = 32;
        m[i]->w = s->bytes + s->toff[3 + i];
        m[i]->s = s->bytes + s->toff[i];
    }
}

int64_t dsv4_cache_expert_bytes(const DSV4Cfg *cfg)
{
    DSV4Cache tmp;
    memset(&tmp, 0, sizeof tmp);
    dsv4_cache_expert_geometry(&tmp, cfg);
    return tmp.expert_bytes;
}

int64_t dsv4_cache_slot_bytes(const DSV4Cfg *cfg)
{
    DSV4Cache tmp;
    memset(&tmp, 0, sizeof tmp);
    dsv4_cache_expert_geometry(&tmp, cfg);
    return tmp.slot_bytes;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_geom.c - expert slot geometry and pointer binding. */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "dsv4_cache.h"
#include "dsv4_trunk.h"
#include "dsv4_cache_int.h"

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

int64_t dsv4_cache_slot_bytes(const DSV4Cfg *cfg)
{
    DSV4Cache tmp;
    memset(&tmp, 0, sizeof tmp);
    dsv4_cache_expert_geometry(&tmp, cfg);
    return tmp.slot_bytes;
}

typedef struct {
    int     expert;
    int     shard;
    int64_t off;
} PfKey;

static int pf_key_before(const PfKey *a, const PfKey *b)
{
    if (a->shard != b->shard) return a->shard < b->shard;
    if (a->off != b->off) return a->off < b->off;
    return a->expert < b->expert;
}

void dsv4_cache_sort_experts(const DSV4Cache *c, int layer, int *experts, int n)
{
    if (!c || !c->st || !experts || n <= 1) return;
    /* The caller uses the grouped fallback for batches this large; keep this
     * helper's stack array fixed. */
    if (n > DSV4_MAX_PREFETCH) return;

    PfKey key[DSV4_MAX_PREFETCH];
    for (int i = 0; i < n; i++) {
        const int e = experts[i];
        const DSV4Loc *loc = dsv4_st_expert(c->st, layer, e);
        key[i].expert = e;
        key[i].shard = loc ? loc[DSV4_EXP_W1W].shard : INT_MAX;
        key[i].off = loc ? loc[DSV4_EXP_W1W].off : INT64_MAX;
    }

    /* At most 256 entries: insertion sort avoids a qsort context and keeps the
     * equal-offset tie on expert id deterministic. */
    for (int i = 1; i < n; i++) {
        const PfKey k = key[i];
        int j = i - 1;
        while (j >= 0 && pf_key_before(&k, &key[j])) {
            key[j + 1] = key[j];
            j--;
        }
        key[j + 1] = k;
    }
    for (int i = 0; i < n; i++) experts[i] = key[i].expert;
}

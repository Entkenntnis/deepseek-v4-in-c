/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind.c - slot tables and the byte-to-field binding loop. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_bind.h"
#include "dsv4_trunk.h"
#include "dsv4_util.h"

/* ---- model-level slots ---- */
enum {
    M_EMBED = 1, M_NORM, M_HEAD,
    M_HC_FN, M_HC_BASE, M_HC_SCALE
};

typedef struct {
    int      slot;
    uint16_t dst;
    uint8_t  wide;      /* 1 = widen to f32, 0 = keep raw bytes */
} ModelSlot;

static const ModelSlot model_slots[] = {
    { M_EMBED,    offsetof(DSV4ModelW, embed),          0 },
    { M_NORM,     offsetof(DSV4ModelW, norm),           1 },
    { M_HEAD,     offsetof(DSV4ModelW, head),           0 },
    { M_HC_FN,    offsetof(DSV4ModelW, hc_head.fn),     1 },
    { M_HC_BASE,  offsetof(DSV4ModelW, hc_head.base),   1 },
    { M_HC_SCALE, offsetof(DSV4ModelW, hc_head.scale),  1 },
};

static int64_t model_bytes(const DSV4St *s, int want_head)
{
    int64_t total = 0;
    for (size_t i = 0; i < sizeof model_slots / sizeof model_slots[0]; i++) {
        const ModelSlot *ms = &model_slots[i];
        if (ms->slot == M_HEAD && !want_head) continue;
        const DSV4ModelLoc *loc = dsv4_st_model(s, ms->slot);
        if (!loc) {
            fprintf(stderr, "dsv4_bind: model slot %d is missing\n", ms->slot);
            return -1;
        }
        int64_t bytes = loc->nbytes;
        if (ms->wide) {
            const int esz = dsv4_st_elemsize((DSV4Dtype)loc->dtype);
            if (esz <= 0 || loc->nbytes % esz) return -1;
            bytes = (loc->nbytes / esz) * 4;
        }
        total = (total + bytes + 7) & ~(int64_t)7;
    }
    return total;
}

int dsv4_bind_model(const DSV4St *s, const DSV4Cfg *c, int want_head,
                    DSV4ModelBind *m)
{
    memset(m, 0, sizeof *m);
    const int64_t need = model_bytes(s, want_head);
    if (need < 0) return -1;

    unsigned char *blob = (unsigned char *)malloc(need ? (size_t)need : 1);
    if (!blob) return -1;
    dsv4_hugepages(blob, (size_t)need);

    size_t at = 0;
    for (size_t i = 0; i < sizeof model_slots / sizeof model_slots[0]; i++) {
        const ModelSlot *ms = &model_slots[i];
        if (ms->slot == M_HEAD && !want_head) continue;
        const DSV4ModelLoc *loc = dsv4_st_model(s, ms->slot);
        if (!loc) { free(blob); return -1; }

        unsigned char *dst = blob + at;
        if (ms->wide) {
            const int esz = dsv4_st_elemsize((DSV4Dtype)loc->dtype);
            const int64_t n = loc->nbytes / esz;
            if (dsv4_st_read_f32(s, loc->shard, loc->off, loc->nbytes,
                                 loc->dtype, (float *)dst) != n) {
                free(blob);
                return -1;
            }
            at += (size_t)n * sizeof(float);
        } else {
            if (dsv4_st_read(s, loc->shard, loc->off, loc->nbytes, dst)
                != loc->nbytes) {
                free(blob);
                return -1;
            }
            at += (size_t)loc->nbytes;
        }
        at = (at + 7) & ~(size_t)7;
        *(const void **)((unsigned char *)&m->w + ms->dst) = dst;
    }

    m->blob = blob;
    m->w.wdt = DSV4_WBF16;
    (void)c;
    return 0;
}

void dsv4_bind_model_free(DSV4ModelBind *m)
{
    if (!m) return;
    free(m->blob);
    memset(m, 0, sizeof *m);
}

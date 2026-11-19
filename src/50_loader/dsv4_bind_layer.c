/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind_layer.c - slot -> DSV4LayerW mapping and the bind loop. */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "dsv4_bind.h"
#include "dsv4_trunk.h"

/* ---- layer slots ---- */
enum {
    L_ATTN_NORM = 1, L_FFN_NORM, L_ATTN_SINK, L_Q_NORM, L_KV_NORM,
    L_WQ_A = 10, L_WQ_A_S, L_WQ_B, L_WQ_B_S, L_WKV, L_WKV_S,
    L_WO_A, L_WO_A_S, L_WO_B, L_WO_B_S,
    L_COMP_APE = 20, L_COMP_NORM, L_COMP_WKV, L_COMP_WGATE,
    L_IDX_WQ_B = 24, L_IDX_WQ_B_S, L_IDX_WP,
    L_IDX_COMP_APE, L_IDX_COMP_NORM, L_IDX_COMP_WKV, L_IDX_COMP_WGATE,
    L_HC_ATTN_FN = 31, L_HC_ATTN_BASE, L_HC_ATTN_SCALE,
    L_HC_FFN_FN, L_HC_FFN_BASE, L_HC_FFN_SCALE,
    L_GATE = 40, L_TID2EID, L_BIAS,
    L_SH_W1 = 50, L_SH_W1_S, L_SH_W2, L_SH_W2_S, L_SH_W3, L_SH_W3_S,
    L_SLOT_MAX = 64
};

enum { SK_NONE, SK_VEC, SK_I64, SK_QMAT, SK_SCALE };

typedef struct {
    uint16_t dst;      /* offsetof(DSV4LayerW, field) */
    uint8_t  kind;
} LayerSlot;

static const LayerSlot layer_slots[L_SLOT_MAX] = {
    [L_ATTN_NORM]  = { offsetof(DSV4LayerW, attn_norm), SK_VEC },
    [L_FFN_NORM]   = { offsetof(DSV4LayerW, ffn_norm),  SK_VEC },
    [L_ATTN_SINK]  = { offsetof(DSV4LayerW, attn.sink), SK_VEC },
    [L_Q_NORM]     = { offsetof(DSV4LayerW, attn.q_norm), SK_VEC },
    [L_KV_NORM]    = { offsetof(DSV4LayerW, attn.kv_norm), SK_VEC },

    [L_WQ_A]   = { offsetof(DSV4LayerW, attn.wq_a), SK_QMAT },
    [L_WQ_A_S] = { offsetof(DSV4LayerW, attn.wq_a), SK_SCALE },
    [L_WQ_B]   = { offsetof(DSV4LayerW, attn.wq_b), SK_QMAT },
    [L_WQ_B_S] = { offsetof(DSV4LayerW, attn.wq_b), SK_SCALE },
    [L_WKV]    = { offsetof(DSV4LayerW, attn.wkv),  SK_QMAT },
    [L_WKV_S]  = { offsetof(DSV4LayerW, attn.wkv),  SK_SCALE },
    [L_WO_A]   = { offsetof(DSV4LayerW, attn.wo_a), SK_QMAT },
    [L_WO_A_S] = { offsetof(DSV4LayerW, attn.wo_a), SK_SCALE },
    [L_WO_B]   = { offsetof(DSV4LayerW, attn.wo_b), SK_QMAT },
    [L_WO_B_S] = { offsetof(DSV4LayerW, attn.wo_b), SK_SCALE },

    [L_COMP_APE]   = { offsetof(DSV4LayerW, attn.comp.ape),  SK_VEC },
    [L_COMP_NORM]  = { offsetof(DSV4LayerW, attn.comp.norm), SK_VEC },
    [L_COMP_WKV]   = { offsetof(DSV4LayerW, attn.comp.wkv),  SK_QMAT },
    [L_COMP_WGATE] = { offsetof(DSV4LayerW, attn.comp.wgate), SK_QMAT },

    [L_IDX_WQ_B]        = { offsetof(DSV4LayerW, attn.idx.wq_b), SK_QMAT },
    [L_IDX_WQ_B_S]      = { offsetof(DSV4LayerW, attn.idx.wq_b), SK_SCALE },
    [L_IDX_WP]          = { offsetof(DSV4LayerW, attn.idx.weights_proj), SK_QMAT },
    [L_IDX_COMP_APE]    = { offsetof(DSV4LayerW, attn.idx.comp.ape),  SK_VEC },
    [L_IDX_COMP_NORM]   = { offsetof(DSV4LayerW, attn.idx.comp.norm), SK_VEC },
    [L_IDX_COMP_WKV]    = { offsetof(DSV4LayerW, attn.idx.comp.wkv),  SK_QMAT },
    [L_IDX_COMP_WGATE]  = { offsetof(DSV4LayerW, attn.idx.comp.wgate), SK_QMAT },

    [L_HC_ATTN_FN]    = { offsetof(DSV4LayerW, hc_attn.fn), SK_VEC },
    [L_HC_ATTN_BASE]  = { offsetof(DSV4LayerW, hc_attn.base), SK_VEC },
    [L_HC_ATTN_SCALE] = { offsetof(DSV4LayerW, hc_attn.scale), SK_VEC },
    [L_HC_FFN_FN]     = { offsetof(DSV4LayerW, hc_ffn.fn), SK_VEC },
    [L_HC_FFN_BASE]   = { offsetof(DSV4LayerW, hc_ffn.base), SK_VEC },
    [L_HC_FFN_SCALE]  = { offsetof(DSV4LayerW, hc_ffn.scale), SK_VEC },

    [L_GATE]    = { offsetof(DSV4LayerW, moe.gate), SK_QMAT },
    [L_TID2EID] = { offsetof(DSV4LayerW, moe.tid2eid), SK_I64 },
    [L_BIAS]    = { offsetof(DSV4LayerW, moe.bias), SK_VEC },

    [L_SH_W1]   = { offsetof(DSV4LayerW, moe.shared.w1), SK_QMAT },
    [L_SH_W1_S] = { offsetof(DSV4LayerW, moe.shared.w1), SK_SCALE },
    [L_SH_W2]   = { offsetof(DSV4LayerW, moe.shared.w2), SK_QMAT },
    [L_SH_W2_S] = { offsetof(DSV4LayerW, moe.shared.w2), SK_SCALE },
    [L_SH_W3]   = { offsetof(DSV4LayerW, moe.shared.w3), SK_QMAT },
    [L_SH_W3_S] = { offsetof(DSV4LayerW, moe.shared.w3), SK_SCALE },
};

static int widen_vector(const uint8_t *src, int64_t nbytes, int dtype,
                        float *dst, size_t cap, size_t *woff)
{
    const int esz = dsv4_st_elemsize((DSV4Dtype)dtype);
    if (esz <= 0 || nbytes % esz) return -1;
    const int64_t n = nbytes / esz;
    const size_t need = (size_t)n * sizeof(float);
    if (*woff + need > cap) return -1;
    if (dsv4_widen_f32(dst, src, nbytes, dtype) != 0) return -1;
    *woff = (*woff + need + 7) & ~(size_t)7;
    return 0;
}

int dsv4_bind_layer(const DSV4Cfg *c, int layer,
                    const struct DSV4TrunkLayer *lay,
                    const unsigned char *run, DSV4LayerW *w,
                    unsigned char *widen, size_t widen_cap)
{
    memset(w, 0, sizeof *w);
    w->layer = layer;
    w->compress_ratio = dsv4_compress_ratio(c, layer);
    w->has_comp = w->compress_ratio != 0;
    w->has_idx = dsv4_has_indexer(c, layer);

    size_t woff = 0;
    for (int i = 0; i < lay->nt; i++) {
        const DSV4TrunkTensor *t = &lay->t[i];
        if (t->slot <= 0 || t->slot >= L_SLOT_MAX ||
            layer_slots[t->slot].kind == SK_NONE) {
            fprintf(stderr, "dsv4_bind: layer %d has unknown slot %d\n",
                    layer, t->slot);
            return -1;
        }
        const LayerSlot *ls = &layer_slots[t->slot];
        unsigned char *dst = (unsigned char *)w + ls->dst;

        switch (ls->kind) {
        case SK_VEC: {
            const void *p = run + t->off;
            if (t->dtype != DSV4_DT_F32) {
                if (!widen) {
                    fprintf(stderr, "dsv4_bind: layer %d slot %d needs a "
                                    "widen buffer\n", layer, t->slot);
                    return -1;
                }
                unsigned char *at = widen + woff;
                if (widen_vector(run + t->off, t->nbytes, t->dtype,
                                 (float *)at, widen_cap, &woff) != 0) {
                    fprintf(stderr, "dsv4_bind: layer %d slot %d cannot "
                                    "widen (dtype %d, %lld bytes)\n",
                            layer, t->slot, t->dtype, (long long)t->nbytes);
                    return -1;
                }
                p = at;
            }
            *(const void **)dst = p;
            break;
        }
        case SK_I64:
            *(const void **)dst = run + t->off;
            break;
        case SK_QMAT: {
            DSV4QMat *q = (DSV4QMat *)dst;
            q->wdt = t->wdt;
            q->rows = t->rows;
            q->cols = t->cols;
            q->blk_r = t->blk_r;
            q->blk_c = t->blk_c;
            q->w = run + t->off;
            break;
        }
        case SK_SCALE:
            ((DSV4QMat *)dst)->s = run + t->off;
            break;
        default:
            return -1;
        }
    }
    return 0;
}

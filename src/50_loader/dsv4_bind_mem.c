/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind_mem.c - binding from a trunk run: point narrow tensors in place,
 * widen only the small vectors. See README.md. */
#include <stdio.h>
#include <string.h>

#include "dsv4_bind.h"
#include "dsv4_bind_int.h"

/* Only BF16 and F32 ever reach here: plan_layer marks quantised tensors narrow
 * because they cannot be widened without their scale partner. */
static int widen_into(float *dst, const unsigned char *src, int64_t n, int dtype)
{
    if (dtype == DSV4_DT_F32) {
        memcpy(dst, src, (size_t)n * sizeof(float));
        return 0;
    }
    if (dtype == DSV4_DT_BF16) {
        const uint16_t *p = (const uint16_t *)src;
        for (int64_t i = 0; i < n; i++) {
            union { uint32_t u; float f; } v;
            v.u = (uint32_t)p[i] << 16;
            dst[i] = v.f;
        }
        return 0;
    }
    fprintf(stderr, "dsv4_bind: cannot widen dtype %d from a trunk run\n", dtype);
    return -1;
}

size_t dsv4_bind_widen_bytes(const DSV4Cfg *c)
{
    /* Trunk slots are uniform, so this must be the MAXIMUM over every layer.
     * Computed by planning all of them: the obvious pick (the ratio-4 indexed
     * layer) is wrong, because `ape` is ratio*coff*head_dim and a ratio-128
     * layer needs sixteen times more. n_layers is at most 61, and planning is
     * pure name formatting, so the exhaustive answer costs nothing. */
    size_t worst = 0;
    for (int L = 0; L < c->n_layers; L++) {
        DSV4BindPlan p; memset(&p, 0, sizeof p);
        DSV4LayerW tmp;
        dsv4_bind_plan_layer(&p, c, L, &tmp);

        size_t total = 0;
        for (int i = 0; i < p.n; i++)
            if (!p.r[i].narrow)
                total = dsv4_bind_align8(total +
                                         (size_t)p.r[i].want * sizeof(float));
        if (total > worst) worst = total;
    }
    return worst;
}

int dsv4_bind_layer_mem(const DSV4Cfg *c, int layer, DSV4LayerBind *b,
                        const unsigned char *run, const DSV4MemSrc *src,
                        unsigned char *widen, size_t widen_cap,
                        size_t *widen_used)
{
    memset(b, 0, sizeof *b);
    b->layer = layer;
    b->blob = NULL;                 /* the run is borrowed, not owned */

    DSV4BindPlan p; memset(&p, 0, sizeof p);
    dsv4_bind_plan_layer(&p, c, layer, &b->w);

    size_t woff = 0;
    for (int i = 0; i < p.n; i++) {
        DSV4BindReq *q = &p.r[i];
        int64_t off = 0, nbytes = 0;
        int dtype = 0;
        const int found = src->find_idx
            ? src->find_idx(src->ctx, i, q->name, &off, &nbytes, &dtype)
            : src->find(src->ctx, q->name, &off, &nbytes, &dtype);
        if (found != 0) {
            fprintf(stderr, "dsv4_bind: %s is not in the trunk run for layer %d\n",
                    q->name, layer);
            return -1;
        }

        const int esz = dsv4_st_elemsize((DSV4Dtype)dtype);
        if (esz <= 0) {
            fprintf(stderr, "dsv4_bind: %s has unusable dtype %d\n", q->name, dtype);
            return -1;
        }
        const int64_t have = nbytes / esz;
        /* The same element-count check the shard path makes, so a trunk packed
         * from a different checkpoint cannot bind silently. */
        if (q->want >= 0 && have != q->want) {
            fprintf(stderr, "dsv4_bind: %s holds %lld elements in the trunk, "
                            "config implies %lld\n", q->name,
                    (long long)have, (long long)q->want);
            return -1;
        }

        if (q->narrow) {
            /* No copy: point straight into the run. */
            *q->dest = run + off;
        } else {
            const size_t need = (size_t)have * sizeof(float);
            if (woff + need > widen_cap) {
                fprintf(stderr, "dsv4_bind: widen area too small for layer %d "
                                "(need %zu, have %zu)\n", layer, woff + need,
                        widen_cap);
                return -1;
            }
            if (widen_into((float *)(widen + woff), run + off, have, dtype) != 0)
                return -1;
            *q->dest = widen + woff;
            woff = dsv4_bind_align8(woff + need);
        }
    }

    if (widen_used) *widen_used = woff;
    return 0;
}

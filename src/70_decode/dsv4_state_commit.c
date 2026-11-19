/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_state_commit.c - all-or-nothing copy of a validated payload. */
#include <string.h>

#include "dsv4_state_file_int.h"

int kv_commit(const DSV4Cfg *c, DSV4LayerState *lst, int32_t *ids,
              const int32_t *file_ids, int n_ids, const int *ncomp,
              const float *buf)
{
    const size_t wn = kv_layer_window_floats(c);

    /* Validate every destination pointer before the first memcpy. A commit
     * must be all-or-nothing even if a caller hands us an uninitialised state. */
    for (int L = 0; L < c->n_layers; L++) {
        const int ratio = kv_layer_ratio(c, L);
        if (!lst[L].kv_cache) return -1;
        if (ratio && (!lst[L].comp_kv || !lst[L].comp_score)) return -1;
    }

    size_t at = 0;
    for (int L = 0; L < c->n_layers; L++) {
        DSV4LayerState *st = &lst[L];
        const int ratio = kv_layer_ratio(c, L);
        const size_t cn = kv_layer_cn(c, L);
        memcpy(st->kv_cache, buf + at, wn * sizeof(float));
        at += wn;
        if (ratio) {
            const size_t rows = (size_t)ncomp[L] * (size_t)c->head_dim;
            memcpy(st->kv_cache + wn, buf + at, rows * sizeof(float));
            at += rows;
            memcpy(st->comp_kv, buf + at, cn * sizeof(float));
            at += cn;
            memcpy(st->comp_score, buf + at, cn * sizeof(float));
            at += cn;
            st->n_compressed = ncomp[L];
        } else {
            st->n_compressed = 0;
        }
    }
    memcpy(ids, file_ids, (size_t)n_ids * sizeof *ids);
    return 0;
}

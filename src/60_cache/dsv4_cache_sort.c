/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_sort.c - put one layer's expert requests into checkpoint order.
 *
 * The cache resolves six tensors per expert and the reader pool could issue
 * them in any order. First-use order is routing order, which on a 200-token
 * block jumps around the file. The store itself is laid out expert by expert,
 * so sorting by (shard, byte offset of w1.weight) turns the layer's reads into
 * one increasing sweep while leaving the slot decision content unchanged.
 * The math is unaffected: buckets still accumulate into their own k slots and
 * the final weighted sum still runs in k order. */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#include "dsv4_cache.h"

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
        char name[224];
        const DSV4Tensor *t;

        snprintf(name, sizeof name, "layers.%d.ffn.experts.%d.w1.weight",
                 layer, e);
        t = dsv4_st_find(c->st, name);
        if (!t) {
            snprintf(name, sizeof name, "layers.%d.ffn.experts.%d.w1.scale",
                     layer, e);
            t = dsv4_st_find(c->st, name);
        }
        key[i].expert = e;
        key[i].shard = t ? t->shard : INT_MAX;
        key[i].off = t ? t->off : INT64_MAX;
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

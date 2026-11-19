/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_topk.c - top-k selection for the CSA indexer.
 *
 * Kept apart from the score pass so the heap can be tested and replaced
 * without touching the dot-product loop.
 */
#include <stdio.h>
#include <stdlib.h>

#include "dsv4_kernels.h"

/* The indexer never selects more than index_topk rows (512 on Flash, 1024 on
 * Pro). A fixed bound lets the heap live on the stack and removes the
 * allocation-failure branch entirely. */
#define DSV4_TOPK_MAX 1024

/* Top-k by score, descending, ties broken toward the LOWER index.
 *
 * torch.topk's tie-breaking is what a checkpoint's behaviour was fixed against,
 * and near the selection boundary ties are not rare once scores are quantised.
 * Writes k indices; k is clamped to n. Returns the number written.
 *
 * THE OLD LOOP WAS O(k*n*k). For every one of k selections it rescanned n
 * candidates and, for each, rescanned the already-chosen prefix to see whether
 * it was taken. At the indexer's real geometry (n up to a few thousand, k=512)
 * that is far too many comparisons per layer; it only looked cheap on short
 * benchmark prompts. Measured in the old microbenchmarks.
 *
 * A bounded min-heap of the current best k is O(n log k). The root is the entry
 * that would be dropped first under the reference ordering: lower score, or
 * equal score and higher index. Replacing the root with a better candidate and
 * sifting down gives exactly the same set and the same order as the old scan,
 * including its tie rule; test_indexer compares the two. */
typedef struct {
    float s;
    int   i;
} DSV4TopkEntry;

static int topk_worse(const DSV4TopkEntry *a, const DSV4TopkEntry *b)
{
    if (a->s < b->s) return 1;
    if (a->s > b->s) return 0;
    return a->i > b->i;
}

static int topk_better(const DSV4TopkEntry *a, const DSV4TopkEntry *b)
{
    if (a->s > b->s) return 1;
    if (a->s < b->s) return 0;
    return a->i < b->i;
}

static void topk_sift_down(DSV4TopkEntry *h, int n, int i)
{
    for (;;) {
        const int l = 2 * i + 1, r = l + 1;
        int m = i;
        if (l < n && topk_worse(&h[l], &h[m])) m = l;
        if (r < n && topk_worse(&h[r], &h[m])) m = r;
        if (m == i) break;
        const DSV4TopkEntry t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

static void topk_sift_up(DSV4TopkEntry *h, int i)
{
    while (i > 0) {
        const int p = (i - 1) / 2;
        if (!topk_worse(&h[i], &h[p])) break;
        const DSV4TopkEntry t = h[i]; h[i] = h[p]; h[p] = t;
        i = p;
    }
}

int dsv4_topk(int *idxs, const float *scores, int n, int k)
{
    if (k > n) k = n;
    if (k <= 0 || n <= 0) return k;
    if (k > DSV4_TOPK_MAX) {
        fprintf(stderr, "dsv4_topk: k=%d exceeds the %d-row bound\n",
                k, DSV4_TOPK_MAX);
        abort();
    }

    DSV4TopkEntry h[DSV4_TOPK_MAX];
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        const DSV4TopkEntry e = { scores[i], i };
        if (cnt < k) {
            h[cnt] = e;
            topk_sift_up(h, cnt);
            cnt++;
        } else if (topk_better(&e, &h[0])) {
            h[0] = e;
            topk_sift_down(h, cnt, 0);
        }
    }

    /* Descending, ties to the lower index. k is the indexer's top-k (512 on
     * Flash, 1024 on Pro), so insertion sort is the right tool here. */
    for (int i = 1; i < cnt; i++) {
        const DSV4TopkEntry t = h[i];
        int j = i;
        while (j > 0 && topk_better(&t, &h[j - 1])) {
            h[j] = h[j - 1];
            j--;
        }
        h[j] = t;
    }
    for (int i = 0; i < cnt; i++) idxs[i] = h[i].i;
    return k;
}

/* Shift selected compressed positions into the attention cache's coordinates.
 *
 * The attention kv_cache is laid out as [ window_size | compressed rows ], so a
 * compressed index t lives at t + offset. Entries of -1 are masked slots and
 * must stay -1 rather than becoming offset-1. */

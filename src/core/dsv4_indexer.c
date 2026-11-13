/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_indexer.c - CSA: the lightning indexer.
 *
 * Source: model.py Indexer. Present ONLY where compress_ratio == 4 (invariant 2),
 * confirmed in the checkpoint: layer 2 carries attn.indexer.*, layer 3 does not.
 *
 * WHAT IT IS FOR
 *   Attention cannot afford to look at every compressed position at long
 *   context, so a cheap scorer picks index_topk of them (512 on Flash, 1024 on
 *   Pro) and only those enter the real attention. The indexer runs in its own
 *   narrower space -- index_n_heads = 64 heads of index_head_dim = 128, against
 *   attention's 64 heads of 512 -- and keeps its own compressed KV cache built
 *   by its own Compressor.
 *
 * FOUR THINGS THAT MATTER
 *
 * 1. THERE IS A ReLU ON THE PER-HEAD SCORES, BEFORE THE HEAD WEIGHTING:
 *        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
 *    Every negative per-head score is discarded, not merely down-weighted, so a
 *    head can only ever vote FOR a position. Dropping the relu leaves a
 *    correctly-shaped score that ranks positions differently, and the model
 *    still generates.
 *
 * 2. THE SCALE IS NOT ATTENTION'S. Here softmax_scale = index_head_dim^-0.5
 *    (128), and it is multiplied by a further n_heads^-0.5 (64) before being
 *    applied to the head weights:
 *        weights = weights_proj(x) * (index_head_dim^-0.5 * index_n_heads^-0.5)
 *    Reusing attention's head_dim^-0.5 (512) is a plausible and wrong constant.
 *
 * 3. THE HADAMARD ROTATION CANCELS IN EXACT ARITHMETIC. Both q and the
 *    indexer's compressed kv go through rotate_activation, which is an
 *    orthogonal Hadamard transform. Orthogonal maps preserve inner products, so
 *    q.k is unchanged by applying it to both sides. It exists to spread
 *    outliers before FP4 quantisation, not to change the score. This file
 *    computes in f32 and therefore omits BOTH the rotation and the FP4
 *    simulation -- see the scope note below.
 *
 * 4. THE INDICES ARE OFFSET. The attention kv_cache holds the sliding window
 *    first and the compressed rows after it, so the indexer's positions are
 *    shifted by `offset` before being concatenated with the window indices.
 *    Forgetting it points every compressed selection at a window slot.
 *
 * SCOPE: the FP4 activation simulation (fp4_act_quant on q and on the indexer's
 * compressed kv) is NOT implemented. It is quantisation-aware-training
 * emulation: it perturbs the scores slightly and so can reorder positions near
 * the top-k boundary. Output here will therefore agree with the reference on
 * ranking but not necessarily on the exact membership of the last few selected
 * positions. That must be measured against the oracle, not assumed away.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_dot.h"

/* Score every candidate compressed position for one query.
 *
 *   q        [n_heads][head_dim]      already RoPE'd
 *   kv       [n_pos][head_dim]        the indexer's own compressed cache
 *   weights  [n_heads]                weights_proj(x), UNSCALED
 *   out      [n_pos]
 *
 * The caller passes weights straight from the projection; the two-part scale is
 * applied here so it cannot be forgotten at a call site. */
void dsv4_indexer_score(float *out, const float *q, const float *kv,
                        const float *weights, int n_heads, int head_dim,
                        int n_pos)
{
    const float scale = (1.0f / sqrtf((float)head_dim))
                      * (1.0f / sqrtf((float)n_heads));

    /* Positions are independent: out[t] depends only on q, weights and kv[t].
     * Parallelising over t therefore changes no reduction order and cannot
     * change a result, which is why this one is unconditional above a small
     * threshold. The dot itself uses the shared 4-lane double tree. */
#pragma omp parallel for schedule(static) if (n_pos > 32)
    for (int t = 0; t < n_pos; t++) {
        const float *kt = kv + (size_t)t * head_dim;
        double total = 0.0;
        for (int h = 0; h < n_heads; h++) {
            const float *qh = q + (size_t)h * head_dim;
            const double dot = dsv4_dot_f32d(qh, kt, head_dim);
            /* ReLU FIRST, then weight. A head that scores a position negatively
             * contributes nothing at all rather than voting against it. */
            if (dot > 0.0) total += dot * (double)weights[h];
        }
        out[t] = (float)total * scale;
    }
}

/* Top-k by score, descending, ties broken toward the LOWER index.
 *
 * torch.topk's tie-breaking is what a checkpoint's behaviour was fixed against,
 * and near the selection boundary ties are not rare once scores are quantised.
 * Writes k indices; k is clamped to n. Returns the number written.
 *
 * THE OLD LOOP WAS O(k*n*k). For every one of k selections it rescanned n
 * candidates and, for each, rescanned the already-chosen prefix to see whether
 * it was taken. At the indexer's real geometry (n up to a few thousand, k=512)
 * that is hundreds of millions of comparisons per layer; it only looked cheap
 * because the benchmark prompt leaves the context at ~23 tokens.
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

/* The reference scan, kept only as the allocation-failure fallback. It is the
 * definition the heap is checked against, and it must never be reached on a
 * working machine. */
static int topk_scan(int *idxs, const float *scores, int n, int k)
{
    for (int i = 0; i < k; i++) {
        int best = -1;
        for (int c = 0; c < n; c++) {
            int taken = 0;
            for (int j = 0; j < i; j++) if (idxs[j] == c) { taken = 1; break; }
            if (taken) continue;
            if (best < 0 || scores[c] > scores[best]) best = c;
        }
        idxs[i] = best;
    }
    return k;
}

int dsv4_topk(int *idxs, const float *scores, int n, int k)
{
    if (k > n) k = n;
    if (k <= 0 || n <= 0) return k;

    DSV4TopkEntry *h = (DSV4TopkEntry *)malloc((size_t)k * sizeof *h);
    if (!h) return topk_scan(idxs, scores, n, k);

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
    free(h);
    return k;
}

/* Shift selected compressed positions into the attention cache's coordinates.
 *
 * The attention kv_cache is laid out as [ window_size | compressed rows ], so a
 * compressed index t lives at t + offset. Entries of -1 are masked slots and
 * must stay -1 rather than becoming offset-1. */
void dsv4_indexer_offset(int *idxs, int n, int offset)
{
    for (int i = 0; i < n; i++)
        if (idxs[i] >= 0) idxs[i] += offset;
}

/* How many compressed rows exist at absolute position `pos`, and hence how many
 * the indexer may choose from: (pos + 1) / ratio. The top-k is
 * min(index_topk, that), so early in a sequence every row is selected and the
 * indexer is a no-op. */
int dsv4_indexer_navail(int pos, int ratio) { return (pos + 1) / ratio; }

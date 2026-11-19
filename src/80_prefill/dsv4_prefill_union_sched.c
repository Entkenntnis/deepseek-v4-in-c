/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_union_sched.c - GPU/CPU bucket placement while reads land. */
#include <limits.h>

#include "dsv4.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

static int union_bucket_count(const DSV4PFillCtx *f, int j)
{
    return f->p->ex_count[f->order[j]];
}

/* Number of not-done 9+ buckets in this group. The in-flight GPU bucket is
 * included because it is not marked done until after its submit/wait returns. */
static int union_pending_high(const DSV4PFillCtx *f)
{
    int n = 0;
    for (int j = 0; j < f->ng; j++)
        if (!f->done[j] && union_bucket_count(f, j) >= DSV4_PF_GPU_ALWAYS_MIN)
            n++;
    return n;
}

/* CPU's contested-band ceiling. With fewer than DSV4_PF_MID_BACKLOG high
 * buckets pending it stays at 4; each extra high bucket raises it by one until
 * it reaches 8. This is the "meet in the middle" split. */
static int union_cpu_mid_limit(const DSV4PFillCtx *f)
{
    const int extra = union_pending_high(f) - (DSV4_PF_MID_BACKLOG - 1);
    int lim = DSV4_PF_CPU_ALWAYS_MAX + (extra > 0 ? extra : 0);
    if (lim > DSV4_PREFILL_FILL_MAX) lim = DSV4_PREFILL_FILL_MAX;
    return lim;
}

/* Highest ready 9+ bucket; if none, highest ready 5..8 bucket. Never picks a
 * 1..4 bucket: those belong to the CPU. Ties keep queue order. */
static int union_pick_gpu(const DSV4PFillCtx *f)
{
    if (!f->p->vk) return -1;
    int best_high = -1, best_mid = -1;
    int high_count = -1, mid_count = -1;
    for (int i = *f->qh; i < *f->qt; i++) {
        const int j = f->q[i];
        if (j < 0 || f->done[j] || !f->ex[j]) continue;
        const int count = union_bucket_count(f, j);
        if (count >= DSV4_PF_GPU_ALWAYS_MIN) {
            if (count > high_count) { best_high = i; high_count = count; }
        } else if (count > DSV4_PF_CPU_ALWAYS_MAX && count > mid_count) {
            best_mid = i; mid_count = count;
        }
    }
    return best_high >= 0 ? best_high : best_mid;
}

/* Smallest ready bucket with count <= max_count. Ties keep queue order. */
static int union_pick_cpu(const DSV4PFillCtx *f, int max_count)
{
    int best = -1, best_count = INT_MAX;
    for (int i = *f->qh; i < *f->qt; i++) {
        const int j = f->q[i];
        if (j < 0 || f->done[j] || !f->ex[j]) continue;
        const int count = union_bucket_count(f, j);
        if (count > max_count) continue;
        if (count < best_count) { best = i; best_count = count; }
    }
    return best;
}

/* Run one ready CPU bucket while the GPU fence is pending. Low buckets always
 * win; 5..8 only become eligible as the pending high-bucket backlog grows. */
static int union_fill_cpu(void *ctx)
{
    DSV4PFillCtx *f = (DSV4PFillCtx *)ctx;
    DSV4Prefill *p = f->p;
    if (dsv4_prefill_cancelled(p)) return 0;

    /* Readers keep publishing while the GPU tile runs. Pull in whatever has
     * landed since the last window; this is non-blocking and is what lets the
     * later w3/w2 windows find CPU buckets that were not ready for w1. */
    if (!f->io_done) {
        int k;
        while (dsv4_cache_try(p->cache, &k)) {
            if (k >= 0 && k < f->ng && !f->done[k] && f->ex[k])
                f->q[(*f->qt)++] = k;
        }
    }

    const int best = union_pick_cpu(f, union_cpu_mid_limit(f));
    if (best < 0) return 0;

    const int j = f->q[best];
    f->q[best] = -1;
    const int e = f->order[j];
    dsv4_prefill_expert_bucket_fill(p, f->ex[j], e, p->ex_count[e],
                                    p->ex_tok + p->ex_off[e],
                                    p->ex_k + p->ex_off[e]);
    f->done[j] = 1;
    (*f->remaining)--;
    return 1;
}

/* Run at most one ready bucket: GPU class first, then CPU class. */
int dsv4_pf_union_run_one_ready(DSV4Prefill *p, DSV4PFillCtx *f)
{
    int pick = union_pick_gpu(f);
    if (pick >= 0) {
        const int j = f->q[pick];
        f->q[pick] = -1;
        const int e = f->order[j];
        dsv4_prefill_expert_bucket(p, f->ex[j], e, p->ex_count[e],
                                   p->ex_tok + p->ex_off[e],
                                   p->ex_k + p->ex_off[e],
                                   DSV4_EXPERT_ROUTE_GPU,
                                   union_fill_cpu, f);
        f->done[j] = 1;
        (*f->remaining)--;
        return 1;
    }

    pick = union_pick_cpu(f, INT_MAX);
    if (pick >= 0) {
        const int j = f->q[pick];
        f->q[pick] = -1;
        const int e = f->order[j];
        dsv4_prefill_expert_bucket(p, f->ex[j], e, p->ex_count[e],
                                   p->ex_tok + p->ex_off[e],
                                   p->ex_k + p->ex_off[e],
                                   DSV4_EXPERT_ROUTE_CPU, NULL, NULL);
        f->done[j] = 1;
        (*f->remaining)--;
        return 1;
    }
    return 0;
}

/* Consume one batch. The GPU takes the largest ready 9+ bucket first (or a
 * 5..8 bucket if no high bucket is ready); while it runs, fill_cpu() works the
 * CPU end from the smallest bucket upward. 5..8 move to the CPU only as the
 * pending 9+ backlog grows. If no ready bucket remains, the scheduler asks
 * wait_any() for the next one, so the batch's I/O tail still overlaps compute.
 * end() drains whatever wait_any() did not consume, and the trailing pass puts
 * resident hits back through the same policy.
 *
 * The batch's slot claims are released by end(); on a cancel we still drain it
 * before returning so the reset that follows cannot race a reader. */

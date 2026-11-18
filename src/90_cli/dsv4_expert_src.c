/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_expert_src.c - the DSV4ExpertSrc the engine hands to every layer. */
#include <stdio.h>

#include "dsv4_serve_int.h"

/* Expert source: the cache, carried through ctx. The engine only uses the
 * overlapped begin()/end() pair; get() is the gate path and stays NULL. */
static DSV4CacheBatch g_batch;

static int run_begin_experts(void *ctx, int layer, const int *e, int n,
                             const DSV4ExpertW **out)
{
    return dsv4_cache_get_many_begin((DSV4Cache *)ctx, layer, e, n, out,
                                     &g_batch);
}

static int run_end_experts(void *ctx, const DSV4ExpertW **out)
{
    return dsv4_cache_get_many_end((DSV4Cache *)ctx, out, &g_batch);
}

static int run_wait_any(void *ctx, int *k)
{
    return dsv4_cache_wait_any((DSV4Cache *)ctx, &g_batch, k);
}

static int run_try_any(void *ctx, int *k)
{
    return dsv4_cache_try_any((DSV4Cache *)ctx, &g_batch, k);
}

static void run_sort_experts(void *ctx, int layer, int *experts, int n)
{
    dsv4_cache_sort_experts((const DSV4Cache *)ctx, layer, experts, n);
}

static int run_expert_stats(void *ctx, DSV4ExpertStats *out)
{
    const DSV4Cache *c = (const DSV4Cache *)ctx;
    out->hits = c->hits;
    out->misses = c->misses;
    out->bytes_read = c->bytes_read;
    out->read_n = c->read_n;
    out->read_s = c->read_s;
    out->read_max_s = c->read_max_s;
    for (int i = 0; i < 8; i++) {
        out->lat_n[i] = c->lat_n[i];
        out->lat_s[i] = c->lat_s[i];
    }
    out->slots = c->nslot;
    out->readers = c->nrth;
    out->evictions = c->evictions;
    return 0;
}

/* 0 = old end() barrier; >0 = miss reads in flight per layer. */
static int g_stream_miss = 6;

void dsv4_set_stream_miss(int n)
{
    g_stream_miss = n;
}

int dsv4_stream_miss(void)
{
    return g_stream_miss;
}


int dsv4_make_expert_src(DSV4ExpertSrc *src, DSV4Cache *cache)
{
    *src = (DSV4ExpertSrc){
        .begin = run_begin_experts, .end = run_end_experts,
        .wait_any = run_wait_any, .try_any = run_try_any,
        .stats = run_expert_stats, .sort = run_sort_experts, .ctx = cache,
    };
    return 0;
}

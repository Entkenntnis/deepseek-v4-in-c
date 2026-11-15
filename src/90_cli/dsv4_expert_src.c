/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_expert_src.c - the DSV4ExpertSrc the engine hands to every layer. */
#include <stdio.h>

#include "dsv4_serve_int.h"

/* Expert source: the cache, carried through ctx. */
static const DSV4ExpertW *run_get_expert(void *ctx, int layer, int expert)
{
    return dsv4_cache_get((DSV4Cache *)ctx, layer, expert);
}

static int run_get_experts(void *ctx, int layer, const int *e, int n,
                           const DSV4ExpertW **out)
{
    return dsv4_cache_get_many((DSV4Cache *)ctx, layer, e, n, out);
}

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

static int run_expert_stats(void *ctx, DSV4ExpertStats *out)
{
    const DSV4Cache *c = (const DSV4Cache *)ctx;
    out->hits = c->hits;
    out->misses = c->misses;
    out->evictions = c->evictions;
    out->bytes_read = c->bytes_read;
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
    *src = (DSV4ExpertSrc){ run_get_expert, run_get_experts,
                            run_begin_experts, run_end_experts,
                            run_wait_any, run_expert_stats, cache };
    return 0;
}

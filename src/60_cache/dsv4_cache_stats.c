/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_stats.c - counters reset and the human-readable report. */
#include <stdio.h>

#include "dsv4_cache.h"

void dsv4_cache_reset_stats(DSV4Cache *c)
{
    c->hits = c->misses = c->evictions = c->bytes_read = 0;
    c->pf_requests = c->pf_hits = c->pf_loads = c->pf_fail = c->pf_bytes = 0;
    c->pf_wait_s = 0.0;
}

void dsv4_cache_report(const DSV4Cache *c, const char *label)
{
    const uint64_t n = c->hits + c->misses;
    printf("expert cache %s: %d slots x %.2f MB = %.2f GB\n", label, c->nslot,
           (double)c->expert_bytes / 1048576.0,
           (double)c->nslot * (double)c->expert_bytes / 1073741824.0);
    printf("  %d reader threads, %d staging buffers\n", c->nrth, c->nbounce);
    printf("  %llu requests, %llu hits (%.1f%%), %llu misses, %llu evictions\n",
           (unsigned long long)n, (unsigned long long)c->hits,
           n ? 100.0 * (double)c->hits / (double)n : 0.0,
           (unsigned long long)c->misses, (unsigned long long)c->evictions);
    printf("  %.2f GB read from disk\n",
           (double)(c->bytes_read + c->pf_bytes) / 1073741824.0);
    if (c->stream_cap > 0)
        printf("  miss streaming: up to %d reads in flight, consumed as they "
               "land\n", c->stream_cap);
    if (c->pf_enabled && c->pf_requests > 0)
        printf("  prefetch: %llu requests, %llu hits, %llu loads, "
               "%llu failed, %.2f GB, %.1f ms waiting\n",
               (unsigned long long)c->pf_requests,
               (unsigned long long)c->pf_hits,
               (unsigned long long)c->pf_loads,
               (unsigned long long)c->pf_fail,
               (double)c->pf_bytes / 1073741824.0,
               c->pf_wait_s * 1e3);
}

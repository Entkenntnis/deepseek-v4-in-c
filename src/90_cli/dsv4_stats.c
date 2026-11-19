/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_stats.c - the flat stats fields the default views actually display.
 *
 * TUI: RSS, disk read, cache hit-rate/size, trunk pinned/resident.
 * e2e/prefill: cache hits, misses and read GB.
 * No profiling buckets, latencies or prefetch counters are kept.
 */
#include <stdio.h>

#include "dsv4_serve_int.h"

void dsv4_write_sys(const Engine *e)
{
    (void)e;
    printf(" sys_rss_mb=%.1f sys_io_read_gb=%.4f",
           dsv4_proc_rss_mb(), dsv4_proc_read_gb());
}

void dsv4_write_cache(const Engine *e)
{
    const DSV4Cache *c = &e->cache;
    const uint64_t n = c->hits + c->misses;
    printf(" cache_slots=%d cache_hit=%llu cache_miss=%llu "
           "cache_hit_rate=%.4f cache_read_gb=%.4f cache_resident_gb=%.3f "
           "cache_slot_mb=%.2f",
           c->nslot,
           (unsigned long long)c->hits, (unsigned long long)c->misses,
           n ? (double)c->hits / (double)n : 0.0,
           (double)(c->bytes_read + c->pf_bytes) / 1073741824.0,
           (double)c->slot_bytes * (double)c->nslot / 1073741824.0,
           (double)c->slot_bytes / 1048576.0);
}

void dsv4_write_trunk(const Engine *e)
{
    const DSV4Trunk *t = &e->tr;
    printf(" trunk_layers=%d trunk_pinned=%d trunk_resident_gb=%.3f",
           t->n_layers, t->npin,
           (double)dsv4_trunk_resident_bytes(t) / 1073741824.0);
}

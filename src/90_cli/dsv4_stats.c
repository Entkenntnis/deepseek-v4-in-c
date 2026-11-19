/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_stats.c - the flat stats fields every event carries.
 *
 * The wire has no nesting: `sys_rss_mb` and `cache_hit` are just fields. The
 * Python client re-nests them by prefix into the dictionaries the TUI uses.
 */
#include <inttypes.h>
#include <stdio.h>

#include "dsv4_serve_int.h"

/* Process truth. wall_s is uptime since the engine finished loading. */
void dsv4_write_sys_full(const Engine *e)
{
    double rss, peak;
    dsv4_proc_mem_mb(&rss, &peak);
    printf(" sys_rss_mb=%.1f sys_peak_mb=%.1f sys_io_read_gb=%.4f "
           "sys_cpu_s=%.3f sys_wall_s=%.3f",
           rss, peak, dsv4_proc_read_gb(), dsv4_proc_cpu_s(),
           dsv4_now_s() - e->t_start);
}

void dsv4_write_cache(const Engine *e)
{
    const DSV4Cache *c = &e->cache;
    const uint64_t n = c->hits + c->misses;
    printf(" cache_slots=%d cache_hit=%llu cache_miss=%llu cache_evict=%llu "
           "cache_hit_rate=%.4f cache_read_gb=%.4f cache_resident_gb=%.3f "
           "cache_slot_mb=%.2f cache_pf=%d cache_pf_hit=%llu "
           "cache_pf_load=%llu cache_pf_read_gb=%.4f cache_pf_wait_ms=%.1f",
           c->nslot,
           (unsigned long long)c->hits, (unsigned long long)c->misses,
           (unsigned long long)c->evictions,
           n ? (double)c->hits / (double)n : 0.0,
           (double)(c->bytes_read + c->pf_bytes) / 1073741824.0,
           (double)c->slot_bytes * (double)c->nslot / 1073741824.0,
           (double)c->slot_bytes / 1048576.0,
           c->pf_enabled,
           (unsigned long long)c->pf_hits,
           (unsigned long long)c->pf_loads,
           (double)c->pf_bytes / 1073741824.0,
           c->pf_wait_s * 1e3);
}

void dsv4_write_trunk(const Engine *e)
{
    const DSV4Trunk *t = &e->tr;
    const uint64_t n = t->hits + t->misses;
    printf(" trunk_layers=%d trunk_pinned=%d trunk_slots=%d "
           "trunk_slot_mb=%.2f trunk_hit=%llu trunk_miss=%llu "
           "trunk_hit_rate=%.4f trunk_read_gb=%.4f trunk_load_s=%.3f "
           "trunk_resident_gb=%.3f",
           t->n_layers, t->npin, t->nslot,
           (double)t->slot_bytes / 1048576.0,
           (unsigned long long)t->hits, (unsigned long long)t->misses,
           n ? (double)t->hits / (double)n : 0.0,
           (double)t->bytes_read / 1073741824.0, t->load_seconds,
           (double)dsv4_trunk_resident_bytes(t) / 1073741824.0);
}

void dsv4_write_prof(const Engine *e)
{
    const DSV4Prof *p = &dsv4_prof;
    const double hc  = p->hc        - e->prof_base.hc;
    const double at  = p->attn      - e->prof_base.attn;
    const double ga  = p->gate      - e->prof_base.gate;
    const double io  = p->expert_io - e->prof_base.expert_io;
    const double mm  = p->expert_mm - e->prof_base.expert_mm;
    const double sh  = p->shared    - e->prof_base.shared;
    const double hd  = p->head      - e->prof_base.head;
    const double bd  = p->bind      - e->prof_base.bind;
    const double ix  = p->idxer     - e->prof_base.idxer;
    const double ti  = p->trunk_io  - e->prof_base.trunk_io;
    printf(" prof_hc=%.4f prof_attn=%.4f prof_idxer=%.4f "
           "prof_gate=%.4f prof_expert_io=%.4f prof_expert_mm=%.4f "
           "prof_shared=%.4f prof_head=%.4f prof_bind=%.4f "
           "prof_trunk_io=%.4f prof_accounted=%.4f",
           hc, at, ix, ga, io, mm, sh, hd, bd, ti,
           hc + at + ga + io + mm + sh + hd + bd + ti);
}

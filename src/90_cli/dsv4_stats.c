/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_stats.c - the JSON fragments every event carries. */
#include <inttypes.h>
#include <stdio.h>

#include "dsv4_serve_int.h"

/* Process truth. wall_s is uptime since the engine finished loading. */
void dsv4_write_sys_full(const Engine *e)
{
    double rss, peak;
    dsv4_proc_mem_mb(&rss, &peak);
    printf("\"sys\":{\"rss_mb\":%.1f,\"peak_mb\":%.1f,\"io_read_gb\":%.4f,"
           "\"cpu_s\":%.3f,\"wall_s\":%.3f}",
           rss, peak, dsv4_proc_read_gb(), dsv4_proc_cpu_s(),
           dsv4_now_s() - e->t_start);
}

void dsv4_write_cache(const Engine *e)
{
    const DSV4Cache *c = &e->cache;
    const uint64_t n = c->hits + c->misses;
    printf("\"cache\":{\"slots\":%d,\"hit\":%llu,\"miss\":%llu,\"evict\":%llu,"
           "\"hit_rate\":%.4f,\"read_gb\":%.4f,\"resident_gb\":%.3f,"
           "\"slot_mb\":%.2f,\"pf\":%d,\"pf_hit\":%llu,\"pf_load\":%llu,"
           "\"pf_read_gb\":%.4f,\"pf_wait_ms\":%.1f}",
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
    printf("\"trunk\":{\"layers\":%d,\"pinned\":%d,\"slots\":%d,"
           "\"slot_mb\":%.2f,\"hit\":%llu,\"miss\":%llu,\"hit_rate\":%.4f,"
           "\"read_gb\":%.4f,\"load_s\":%.3f,\"resident_gb\":%.3f}",
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
    printf("\"prof\":{\"hc\":%.4f,\"attn\":%.4f,\"idxer\":%.4f,"
           "\"gate\":%.4f,\"expert_io\":%.4f,\"expert_mm\":%.4f,"
           "\"shared\":%.4f,\"head\":%.4f,\"bind\":%.4f,"
           "\"trunk_io\":%.4f,\"accounted\":%.4f}",
           hc, at, ix, ga, io, mm, sh, hd, bd, ti,
           hc + at + ga + io + mm + sh + hd + bd + ti);
}

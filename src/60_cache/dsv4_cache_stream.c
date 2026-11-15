/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache_stream.c - the stream cap and wait_any(). */
#include "dsv4_cache.h"
#include "dsv4_cache_int.h"
#include "dsv4_prof.h"
#include "dsv4_trace.h"

void dsv4_cache_set_stream_cap(DSV4Cache *c, int cap)
{
    if (!c) return;
    if (cap < 0) cap = 0;
    if (cap > DSV4_MAX_TOPK) cap = DSV4_MAX_TOPK;
    c->stream_cap = cap;
}

int dsv4_cache_wait_any(DSV4Cache *c, DSV4CacheBatch *b, int *k)
{
    if (!c || !b || !k) return 0;
    if (!b->streaming) return 0;

    const int tr = dsv4_trace_active();
    if (tr)
        dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_WAIT, .layer = b->layer });
    const double tw = tr ? dsv4_now_s() : 0.0;

    pthread_mutex_lock(&c->rmx);
    for (;;) {
        int expert = -1;
        if (dsv4_cache_take_ready(c, b, k, &expert)) {
            pthread_mutex_unlock(&c->rmx);
            if (tr)
                dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_WAKE,
                                              .layer = b->layer, .k = *k,
                                              .expert = expert });
            return 1;
        }
        if (b->rdone >= b->ntodo && !dsv4_cache_waits_left(b)) {
            pthread_mutex_unlock(&c->rmx);
            if (tr)
                dsv4_trace_ev(&(DSV4TraceEv){ .kind = DSV4_TR_WAIT_END,
                                              .layer = b->layer,
                                              .ms = (dsv4_now_s() - tw) * 1000.0 });
            return 0;                      /* everything consumed */
        }
        pthread_cond_wait(&c->rdone, &c->rmx);
    }
}

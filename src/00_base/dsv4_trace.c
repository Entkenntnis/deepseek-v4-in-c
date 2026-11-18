/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trace.c - the ring of scheduling events and its file writer.
 *
 * LOCK ORDER: cache rmx -> trace mx. A reader publishes while holding rmx and
 * calls dsv4_trace_ev() there; nothing takes rmx inside this file. The token
 * writer holds only the trace mutex, so it cannot deadlock against a reader. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_prof.h"
#include "dsv4_trace.h"

/* A decode token is ~1,700 events. A prefill layer unions every distinct
 * expert and can emit a BATCH/Q/R/P per expert plus a bucket event and the
 * stage spans, so one call can reach ~150k including per-token sequential
 * spans. Keep enough room for a full block; the allocation only happens when
 * tracing is enabled. */
#define DSV4_TRACE_CAP 262144

typedef struct {
    double    t;
    int       kind, layer, k, expert, slot, reader;
    int       n, ntodo;
    long long bytes;
    double    ms;
    char      tag[20];
} TraceEv;

enum { BLOCK_NONE = 0, BLOCK_DECODE, BLOCK_PREFILL };

static struct {
    int             on, live;
    int             block;          /* which begin() opened the live block */
    int             decode_on, prefill_on;
    FILE           *f;
    pthread_mutex_t mx;
    TraceEv        *ev;
    int             n, dropped, cap;
    double          t0;
    long            token;          /* 1-based count of decode tokens */
    long            from, count;
    long            pf_call;        /* 1-based count of prefill calls */
    long            pf_from, pf_count;
    int             pf_tokens;
    int             pos, lo, hi;
    char            phase[12];
} g = { .lo = -1, .hi = -1, .decode_on = 1 };

static void tag_copy(char *d, const char *s)
{
    size_t i = 0;
    if (s) for (; s[i] && i < sizeof g.ev[0].tag - 1; i++) d[i] = s[i];
    d[i] = 0;
}

int dsv4_trace_open(const char *path)
{
    if (!path || !*path) return 0;
    if (g.on) return 0;

    g.f = fopen(path, "w");
    if (!g.f) {
        fprintf(stderr, "dsv4_trace: cannot open %s\n", path);
        return -1;
    }
    g.ev = (TraceEv *)calloc(DSV4_TRACE_CAP, sizeof *g.ev);
    if (!g.ev) {
        fclose(g.f);
        g.f = NULL;
        return -1;
    }
    if (pthread_mutex_init(&g.mx, NULL) != 0) {
        free(g.ev);
        g.ev = NULL;
        fclose(g.f);
        g.f = NULL;
        return -1;
    }
    g.cap = DSV4_TRACE_CAP;
    g.n = g.dropped = 0;
    g.token = g.pf_call = 0;
    g.live = 0;
    g.block = BLOCK_NONE;
    g.on = 1;
    fprintf(stderr, "dsv4_trace: writing %s\n", path);
    return 0;
}

void dsv4_trace_close(void)
{
    if (!g.on) return;
    g.on = 0;
    g.live = 0;
    g.block = BLOCK_NONE;
    fflush(g.f);
    fclose(g.f);
    g.f = NULL;
    pthread_mutex_destroy(&g.mx);
    free(g.ev);
    g.ev = NULL;
}

void dsv4_trace_phases(int mask)
{
    if (mask <= 0) mask = DSV4_TRACE_PHASE_DECODE;
    g.decode_on  = (mask & DSV4_TRACE_PHASE_DECODE) != 0;
    g.prefill_on = (mask & DSV4_TRACE_PHASE_PREFILL) != 0;
}

void dsv4_trace_window(long from, long count)
{
    g.from = from < 0 ? 0 : from;
    g.count = count < 0 ? 0 : count;
}

void dsv4_trace_prefill_window(long from, long count)
{
    g.pf_from = from < 0 ? 0 : from;
    g.pf_count = count < 0 ? 0 : count;
}

void dsv4_trace_layers(int lo, int hi)
{
    g.lo = lo;
    g.hi = hi;
}

int dsv4_trace_active(void)
{
    return g.on && g.live;
}

void dsv4_trace_ev(const DSV4TraceEv *in)
{
    if (!g.on || !g.live) return;
    if (g.lo >= 0 && (in->layer < g.lo || in->layer > g.hi)) return;

    const double now = in->at > 0.0 ? in->at : dsv4_now_s();
    pthread_mutex_lock(&g.mx);
    if (g.n < g.cap) {
        TraceEv *e = &g.ev[g.n++];
        e->t      = now - g.t0;
        e->kind   = in->kind;
        e->layer  = in->layer;
        e->k      = in->k;
        e->expert = in->expert;
        e->slot   = in->slot;
        e->reader = in->reader;
        e->n      = in->n;
        e->ntodo  = in->ntodo;
        e->bytes  = in->bytes;
        e->ms     = in->ms;
        tag_copy(e->tag, in->tag);
    } else {
        g.dropped++;
    }
    pthread_mutex_unlock(&g.mx);
}

void dsv4_trace_token_begin(int pos, const char *phase)
{
    if (!g.on || !g.decode_on) return;
    g.token++;
    const long idx = g.token - 1;
    const int live = idx >= g.from && (g.count <= 0 || idx < g.from + g.count);

    pthread_mutex_lock(&g.mx);
    g.live = live;
    g.block = live ? BLOCK_DECODE : BLOCK_NONE;
    g.n = 0;
    g.dropped = 0;
    g.t0 = dsv4_now_s();
    g.pos = pos;
    g.pf_tokens = 0;
    tag_copy(g.phase, phase);
    pthread_mutex_unlock(&g.mx);
}

static void trace_flush_locked(const char *header)
{
    fprintf(g.f, "%s", header);
    for (int i = 0; i < g.n; i++) {
        const TraceEv *e = &g.ev[i];
        fprintf(g.f, "%.6f %c %d %d %d %d %d %lld %.3f %d %d %s\n",
                e->t, e->kind, e->layer, e->k, e->expert, e->slot, e->reader,
                e->bytes, e->ms, e->n, e->ntodo, e->tag);
    }
    fflush(g.f);
    g.n = 0;
    g.live = 0;
    g.block = BLOCK_NONE;
}

void dsv4_trace_token_end(void)
{
    if (!g.on || !g.live || g.block != BLOCK_DECODE) return;

    char header[128];
    pthread_mutex_lock(&g.mx);
    snprintf(header, sizeof header,
             "# token %ld pos %d phase %s events %d dropped %d\n",
             g.token - 1, g.pos, g.phase, g.n, g.dropped);
    trace_flush_locked(header);
    pthread_mutex_unlock(&g.mx);
}

void dsv4_trace_prefill_begin(int pos, int tokens)
{
    if (!g.on || !g.prefill_on) return;
    g.pf_call++;
    const long idx = g.pf_call - 1;
    const int live = idx >= g.pf_from &&
                     (g.pf_count <= 0 || idx < g.pf_from + g.pf_count);

    pthread_mutex_lock(&g.mx);
    g.live = live;
    g.block = live ? BLOCK_PREFILL : BLOCK_NONE;
    g.n = 0;
    g.dropped = 0;
    g.t0 = dsv4_now_s();
    g.pos = pos;
    g.pf_tokens = tokens;
    tag_copy(g.phase, "prefill");
    pthread_mutex_unlock(&g.mx);
}

void dsv4_trace_prefill_end(void)
{
    if (!g.on || !g.live || g.block != BLOCK_PREFILL) return;

    char header[160];
    pthread_mutex_lock(&g.mx);
    snprintf(header, sizeof header,
             "# prefill call=%ld pos=%d tokens=%d phase=%s events=%d "
             "dropped=%d\n",
             g.pf_call - 1, g.pos, g.pf_tokens, g.phase, g.n, g.dropped);
    trace_flush_locked(header);
    pthread_mutex_unlock(&g.mx);
}

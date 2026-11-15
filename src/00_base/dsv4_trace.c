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

/* ~43 layers x ~40 events is ~1,700 per token; 8,192 leaves room for a
 * pathological layer and is still only half a megabyte. */
#define DSV4_TRACE_CAP 8192

typedef struct {
    double    t;
    int       kind, layer, k, expert, slot, reader;
    int       n, ntodo;
    long long bytes;
    double    ms;
    char      tag[12];
} TraceEv;

static struct {
    int             on, live;
    FILE           *f;
    pthread_mutex_t mx;
    TraceEv        *ev;
    int             n, dropped, cap;
    double          t0;
    long            token;          /* 1-based count of decode tokens */
    long            from, count;
    int             pos, lo, hi;
    char            phase[12];
} g = { .lo = -1, .hi = -1 };

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
    g.token = 0;
    g.live = 0;
    g.on = 1;
    fprintf(stderr, "dsv4_trace: writing %s\n", path);
    return 0;
}

void dsv4_trace_close(void)
{
    if (!g.on) return;
    g.on = 0;
    g.live = 0;
    fflush(g.f);
    fclose(g.f);
    g.f = NULL;
    pthread_mutex_destroy(&g.mx);
    free(g.ev);
    g.ev = NULL;
}

void dsv4_trace_window(long from, long count)
{
    g.from = from < 0 ? 0 : from;
    g.count = count < 0 ? 0 : count;
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
    if (!g.on) return;
    g.token++;
    const long idx = g.token - 1;
    const int live = idx >= g.from && (g.count <= 0 || idx < g.from + g.count);

    pthread_mutex_lock(&g.mx);
    g.live = live;
    g.n = 0;
    g.dropped = 0;
    g.t0 = dsv4_now_s();
    g.pos = pos;
    tag_copy(g.phase, phase);
    pthread_mutex_unlock(&g.mx);
}

void dsv4_trace_token_end(void)
{
    if (!g.on || !g.live) return;

    pthread_mutex_lock(&g.mx);
    fprintf(g.f, "# token %ld pos %d phase %s events %d dropped %d\n",
            g.token - 1, g.pos, g.phase, g.n, g.dropped);
    for (int i = 0; i < g.n; i++) {
        const TraceEv *e = &g.ev[i];
        fprintf(g.f, "%.6f %c %d %d %d %d %d %lld %.3f %d %d %s\n",
                e->t, e->kind, e->layer, e->k, e->expert, e->slot, e->reader,
                e->bytes, e->ms, e->n, e->ntodo, e->tag);
    }
    fflush(g.f);
    g.n = 0;
    g.live = 0;
    pthread_mutex_unlock(&g.mx);
}

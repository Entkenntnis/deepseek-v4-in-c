/* SPDX-License-Identifier: Apache-2.0 */
/* cases_elementwise.c - RMSNorm, SwiGLU and the router. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "dsv4_kernels.h"


typedef struct { float *y, *x, *w; int n; } TernCase;

static void rms_run(void *p)
{
    TernCase *c = (TernCase *)p;
    dsv4_rmsnorm(c->y, c->x, c->w, c->n, 1e-6f);
}

static void swi_run(void *p)
{
    TernCase *c = (TernCase *)p;
    dsv4_swiglu(c->y, c->x, c->w, c->n, 10.0f);
}

static void bench_simple(const char *name, const char *shape, int n,
                         RunFn fn, int kind)
{
    if (bench_skip(name)) return;

    TernCase c;
    memset(&c, 0, sizeof c);
    c.n = n;
    c.y = (float *)bench_xalloc((size_t)n * sizeof(float));
    c.x = (float *)bench_xalloc((size_t)n * sizeof(float));
    c.w = (float *)bench_xalloc((size_t)n * sizeof(float));
    bench_fill_f32(c.x, (size_t)n, 1.0f);
    bench_fill_f32(c.w, (size_t)n, 1.0f);

    int reps = 0;
    const double sec = bench_measure(fn, &c, 40, &reps);
    const double flops = (kind == 0 ? 3.0 : 6.0) * (double)n;
    const double bytes = 3.0 * (double)n * 4.0;
    bench_report(name, shape, "f32", sec, reps, 1, flops, bytes);

    free(c.y); free(c.x); free(c.w);
}

typedef struct {
    int   *idx;
    float *wts, *scores, *orig, *bias, *base;
    int    n, k;
} RouteCase;

static void route_run(void *p)
{
    RouteCase *r = (RouteCase *)p;
    memcpy(r->scores, r->base, (size_t)r->n * sizeof(float));
    dsv4_route(r->idx, r->wts, r->scores, r->orig, r->bias, NULL, 0,
               r->n, r->k, 2.5f);
}

void bench_route(void)
{
    if (bench_skip("route")) return;

    RouteCase r;
    memset(&r, 0, sizeof r);
    r.n = 256;
    r.k = 6;
    r.idx    = (int *)bench_xalloc(8u * sizeof(int));
    r.wts    = (float *)bench_xalloc(8u * sizeof(float));
    r.scores = (float *)bench_xalloc(256u * sizeof(float));
    r.orig   = (float *)bench_xalloc(256u * sizeof(float));
    r.bias   = (float *)bench_xalloc(256u * sizeof(float));
    r.base   = (float *)bench_xalloc(256u * sizeof(float));
    bench_fill_f32(r.base, 256, 4.0f);
    bench_fill_f32(r.bias, 256, 0.1f);

    int reps = 0;
    const double sec = bench_measure(route_run, &r, 40, &reps);
    const double flops = 8.0 * 256.0 + 6.0 * 256.0;
    const double bytes = (4.0 * 256.0 + 2.0 * 8.0 + 2.0 * 256.0) * 4.0;
    bench_report("route", "256 -> 6", "f32", sec, reps, 1, flops, bytes);

    free(r.idx); free(r.wts); free(r.scores); free(r.orig);
    free(r.bias); free(r.base);
}


static void named_simple(const char *prefix, int n, RunFn fn, int kind)
{
    char name[32], shape[32];
    snprintf(name, sizeof name, "%s_%d", prefix, n);
    snprintf(shape, sizeof shape, "n=%d", n);
    bench_simple(name, shape, n, fn, kind);
}

void bench_rmsnorm(int n) { named_simple("rmsnorm", n, rms_run, 0); }
void bench_swiglu(int n)  { named_simple("swiglu", n, swi_run, 1); }

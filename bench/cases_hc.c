/* SPDX-License-Identifier: Apache-2.0 */
/* cases_hc.c - mHC Sinkhorn split, reduce and expand. */
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "dsv4_kernels.h"


typedef struct {
    float *y, *post, *comb, *mixes, *x;
    float *fn, *base, *scale;
    DSV4HcW w;
    int    hc, hidden, iters;
} HcPreCase;

static void hcpre_run(void *p)
{
    HcPreCase *c = (HcPreCase *)p;
    dsv4_hc_pre(c->y, c->post, c->comb, c->mixes, c->x, &c->w,
                c->hc, c->hidden, c->iters, 1e-6f, 1e-6f);
}

void bench_hc_pre(void)
{
    if (bench_skip("hc_pre")) return;

    const int hc = 4, hidden = 4096, mix = (2 + hc) * hc;
    const int hcdim = hc * hidden;

    HcPreCase c;
    memset(&c, 0, sizeof c);
    c.hc = hc; c.hidden = hidden; c.iters = 20;
    c.y      = (float *)bench_xalloc((size_t)hidden * sizeof(float));
    c.post   = (float *)bench_xalloc((size_t)hc * sizeof(float));
    c.comb   = (float *)bench_xalloc((size_t)hc * hc * sizeof(float));
    c.mixes  = (float *)bench_xalloc((size_t)mix * sizeof(float));
    c.x      = (float *)bench_xalloc((size_t)hcdim * sizeof(float));
    c.fn     = (float *)bench_xalloc((size_t)mix * hcdim * sizeof(float));
    c.base   = (float *)bench_xalloc((size_t)mix * sizeof(float));
    c.scale  = (float *)bench_xalloc(3u * sizeof(float));
    bench_fill_f32(c.x, (size_t)hcdim, 1.0f);
    bench_fill_f32(c.fn, (size_t)mix * hcdim, 0.02f);
    bench_fill_f32(c.base, (size_t)mix, 0.1f);
    c.scale[0] = c.scale[1] = c.scale[2] = 1.0f;
    c.w.fn = c.fn; c.w.base = c.base; c.w.scale = c.scale;

    int reps = 0;
    const double sec = bench_measure(hcpre_run, &c, 40, &reps);
    const double flops = 2.0 * (double)mix * (double)hcdim
                       + 2.0 * (double)hc * (double)hidden;
    const double bytes = ((double)mix * hcdim + hcdim + hidden
                        + mix + hc * hc) * 4.0;
    bench_report("hc_pre", "hc4 hidden4096", "f32/f64", sec, reps, 1, flops, bytes);

    free(c.y); free(c.post); free(c.comb); free(c.mixes); free(c.x);
    free(c.fn); free(c.base); free(c.scale);
}

typedef struct {
    float *y, *x, *resid, *post, *comb;
    int hc, hidden;
} HcPostCase;

static void hcpost_run(void *p)
{
    HcPostCase *c = (HcPostCase *)p;
    dsv4_hc_post(c->y, c->x, c->resid, c->post, c->comb, c->hc, c->hidden);
}

void bench_hc_post(void)
{
    if (bench_skip("hc_post")) return;

    const int hc = 4, hidden = 4096;
    HcPostCase c;
    memset(&c, 0, sizeof c);
    c.hc = hc; c.hidden = hidden;
    c.y     = (float *)bench_xalloc((size_t)hc * hidden * sizeof(float));
    c.x     = (float *)bench_xalloc((size_t)hidden * sizeof(float));
    c.resid = (float *)bench_xalloc((size_t)hc * hidden * sizeof(float));
    c.post  = (float *)bench_xalloc((size_t)hc * sizeof(float));
    c.comb  = (float *)bench_xalloc((size_t)hc * hc * sizeof(float));
    bench_fill_f32(c.x, (size_t)hidden, 1.0f);
    bench_fill_f32(c.resid, (size_t)hc * hidden, 1.0f);
    bench_fill_f32(c.post, (size_t)hc, 1.0f);
    bench_fill_f32(c.comb, (size_t)hc * hc, 0.25f);

    int reps = 0;
    const double sec = bench_measure(hcpost_run, &c, 40, &reps);
    const double flops = 2.0 * (double)hc * (double)hidden * (double)(hc + 1);
    const double bytes = ((double)hc * hidden * 2 + hidden
                        + hc + hc * hc) * 4.0;
    bench_report("hc_post", "hc4 hidden4096", "f32/f64", sec, reps, 1, flops, bytes);

    free(c.y); free(c.x); free(c.resid); free(c.post); free(c.comb);
}

typedef struct {
    float *pre, *post, *comb, *mixes, *scale, *base;
    int hc, iters;
} SinkCase;

static void sink_run(void *p)
{
    SinkCase *c = (SinkCase *)p;
    dsv4_hc_split_sinkhorn(c->pre, c->post, c->comb, c->mixes,
                           c->scale, c->base, c->hc, c->iters, 1e-6f);
}

void bench_sinkhorn(void)
{
    if (bench_skip("hc_sinkhorn")) return;

    const int hc = 4, mix = (2 + hc) * hc;
    SinkCase c;
    memset(&c, 0, sizeof c);
    c.hc = hc; c.iters = 20;
    c.pre   = (float *)bench_xalloc((size_t)hc * sizeof(float));
    c.post  = (float *)bench_xalloc((size_t)hc * sizeof(float));
    c.comb  = (float *)bench_xalloc((size_t)hc * hc * sizeof(float));
    c.mixes = (float *)bench_xalloc((size_t)mix * sizeof(float));
    c.scale = (float *)bench_xalloc(3u * sizeof(float));
    c.base  = (float *)bench_xalloc((size_t)mix * sizeof(float));
    bench_fill_f32(c.mixes, (size_t)mix, 0.1f);
    bench_fill_f32(c.base, (size_t)mix, 0.1f);
    c.scale[0] = c.scale[1] = c.scale[2] = 1.0f;

    int reps = 0;
    const double sec = bench_measure(sink_run, &c, 40, &reps);
    bench_report("hc_sinkhorn", "hc4 iters20", "f32/f64", sec, reps, 1, 0.0, 0.0);

    free(c.pre); free(c.post); free(c.comb); free(c.mixes);
    free(c.scale); free(c.base);
}


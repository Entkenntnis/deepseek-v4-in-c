/* SPDX-License-Identifier: Apache-2.0 */
/* cases_compress.c - the HCA pooling step. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "dsv4_kernels.h"


typedef struct {
    float *out, *kvin, *scin, *ape, *kvs, *scs;
    int    pos, ratio, hd;
} CompCase;

static void comp_run(void *p)
{
    CompCase *c = (CompCase *)p;
    dsv4_compress_step(c->out, c->kvin, c->scin, c->ape, c->kvs, c->scs,
                       c->pos, c->ratio, c->hd);
}

void bench_compress(const char *name, int ratio)
{
    if (bench_skip(name)) return;

    const int hd = 512;
    const int coff = (ratio == 4) ? 2 : 1;
    const int width = coff * hd;
    const int nslot = coff * ratio;

    CompCase c;
    memset(&c, 0, sizeof c);
    c.hd = hd; c.ratio = ratio; c.pos = ratio - 1;
    c.out  = (float *)bench_xalloc((size_t)hd * sizeof(float));
    c.kvin = (float *)bench_xalloc((size_t)width * sizeof(float));
    c.scin = (float *)bench_xalloc((size_t)width * sizeof(float));
    c.ape  = (float *)bench_xalloc((size_t)ratio * width * sizeof(float));
    c.kvs  = (float *)bench_xalloc((size_t)nslot * width * sizeof(float));
    c.scs  = (float *)bench_xalloc((size_t)nslot * width * sizeof(float));
    bench_fill_f32(c.kvin, (size_t)width, 1.0f);
    bench_fill_f32(c.scin, (size_t)width, 1.0f);
    bench_fill_f32(c.ape, (size_t)ratio * width, 0.1f);
    bench_fill_f32(c.kvs, (size_t)nslot * width, 1.0f);
    for (size_t i = 0; i < (size_t)nslot * width; i++) c.scs[i] = -INFINITY;

    int reps = 0;
    const double sec = bench_measure(comp_run, &c, 40, &reps);
    const double flops = 3.0 * (double)nslot * (double)width;
    const double bytes = 2.0 * (double)nslot * (double)width * 4.0;
    char shape[32];
    snprintf(shape, sizeof shape, "hd512 ratio=%d", ratio);
    bench_report(name, shape, "f32", sec, reps, 1, flops, bytes);

    free(c.out); free(c.kvin); free(c.scin); free(c.ape);
    free(c.kvs); free(c.scs);
}


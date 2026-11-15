/* SPDX-License-Identifier: Apache-2.0 */
/* cases_rope.c - YaRN table application over one layer's q heads. */
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "dsv4_kernels.h"


typedef struct {
    float *v, *cs, *sn;
    int hd, rd, h, pos, half;
} RopeCase;

static void rope_run(void *p)
{
    RopeCase *r = (RopeCase *)p;
    for (int i = 0; i < r->h; i++)
        dsv4_rope_apply(r->v + (size_t)i * (size_t)r->hd, r->hd, r->rd,
                        r->cs, r->sn, r->pos, r->half, 0);
}

void bench_rope(void)
{
    if (bench_skip("rope_q")) return;

    const int hd = 512, rd = 64, h = 64, seq = 2048, half = rd / 2;
    RopeCase c;
    memset(&c, 0, sizeof c);
    c.hd = hd; c.rd = rd; c.h = h; c.pos = 1000; c.half = half;
    c.v  = (float *)bench_xalloc((size_t)h * hd * sizeof(float));
    c.cs = (float *)bench_xalloc((size_t)seq * half * sizeof(float));
    c.sn = (float *)bench_xalloc((size_t)seq * half * sizeof(float));
    bench_fill_f32(c.v, (size_t)h * hd, 1.0f);
    dsv4_rope_table(c.cs, c.sn, rd, seq, 0, 10000.0, 16.0, 32.0, 1.0);

    int reps = 0;
    const double sec = bench_measure(rope_run, &c, 40, &reps);
    const double flops = 3.0 * (double)h * (double)rd;
    const double bytes = 2.0 * (double)h * (double)rd * 4.0;
    bench_report("rope_q", "64x512 rd64", "f32", sec, reps, 1, flops, bytes);

    free(c.v); free(c.cs); free(c.sn);
}

/* ------------------------------------------------------------- attention -- */

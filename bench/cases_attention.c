/* SPDX-License-Identifier: Apache-2.0 */
/* cases_attention.c - sparse attention, CSA scoring and top-k. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "dsv4_kernels.h"


typedef struct {
    float *o, *q, *kv, *sink, *scratch;
    int   *idxs;
    int    h, d, n;
} AttnCase;

static void attn_run(void *p)
{
    AttnCase *a = (AttnCase *)p;
    dsv4_sparse_attn(a->o, a->q, a->kv, a->sink, a->idxs, a->h, a->d, a->n,
                     1.0f / sqrtf((float)a->d), a->scratch);
}

void bench_attn(const char *name, int n)
{
    if (bench_skip(name)) return;

    const int h = 64, d = 512;
    AttnCase a;
    memset(&a, 0, sizeof a);
    a.h = h; a.d = d; a.n = n;
    a.o       = (float *)bench_xalloc((size_t)h * d * sizeof(float));
    a.q       = (float *)bench_xalloc((size_t)h * d * sizeof(float));
    a.kv      = (float *)bench_xalloc((size_t)n * d * sizeof(float));
    a.sink    = (float *)bench_xalloc((size_t)h * sizeof(float));
    a.scratch = (float *)bench_xalloc((size_t)h * n * sizeof(float));
    a.idxs    = (int *)bench_xalloc((size_t)n * sizeof(int));
    bench_fill_f32(a.q, (size_t)h * d, 1.0f);
    bench_fill_f32(a.kv, (size_t)n * d, 1.0f);
    bench_fill_f32(a.sink, (size_t)h, 1.0f);
    for (int i = 0; i < n; i++) a.idxs[i] = i;

    int reps = 0;
    const double sec = bench_measure(attn_run, &a, 40, &reps);
    const double flops = 4.0 * (double)h * (double)n * (double)d;
    const double bytes = (double)h * (double)n * (double)d * 4.0;
    char shape[32];
    snprintf(shape, sizeof shape, "64x512 n=%d", n);
    bench_report(name, shape, "f32/f64", sec, reps, 1, flops, bytes);

    free(a.o); free(a.q); free(a.kv); free(a.sink);
    free(a.scratch); free(a.idxs);
}

typedef struct {
    float *out, *q, *kv, *weights;
    int    nh, hd, n;
} IdxCase;

static void idx_run(void *p)
{
    IdxCase *c = (IdxCase *)p;
    dsv4_indexer_score(c->out, c->q, c->kv, c->weights, c->nh, c->hd, c->n);
}

void bench_indexer(void)
{
    if (bench_skip("indexer_score")) return;

    const int nh = 64, hd = 128, n = 2048;
    IdxCase c;
    memset(&c, 0, sizeof c);
    c.nh = nh; c.hd = hd; c.n = n;
    c.out     = (float *)bench_xalloc((size_t)n * sizeof(float));
    c.q       = (float *)bench_xalloc((size_t)nh * hd * sizeof(float));
    c.kv      = (float *)bench_xalloc((size_t)n * hd * sizeof(float));
    c.weights = (float *)bench_xalloc((size_t)nh * sizeof(float));
    bench_fill_f32(c.q, (size_t)nh * hd, 1.0f);
    bench_fill_f32(c.kv, (size_t)n * hd, 1.0f);
    bench_fill_f32(c.weights, (size_t)nh, 1.0f);

    int reps = 0;
    const double sec = bench_measure(idx_run, &c, 40, &reps);
    const double flops = 2.0 * (double)nh * (double)n * (double)hd;
    const double bytes = (double)n * hd * 4.0;
    bench_report("indexer_score", "64x128 n=2048", "f32/f64", sec, reps, 1,
           flops, bytes);

    free(c.out); free(c.q); free(c.kv); free(c.weights);
}

typedef struct {
    int   *idx;
    float *scores;
    int    n, k;
} TopkCase;

static void topk_run(void *p)
{
    TopkCase *t = (TopkCase *)p;
    dsv4_topk(t->idx, t->scores, t->n, t->k);
}

void bench_topk(void)
{
    if (bench_skip("topk_2048")) return;

    TopkCase t;
    memset(&t, 0, sizeof t);
    t.n = 2048; t.k = 512;
    t.idx    = (int *)bench_xalloc((size_t)t.k * sizeof(int));
    t.scores = (float *)bench_xalloc((size_t)t.n * sizeof(float));
    bench_fill_f32(t.scores, (size_t)t.n, 1.0f);

    int reps = 0;
    const double sec = bench_measure(topk_run, &t, 40, &reps);
    bench_report("topk_2048_512", "n=2048 k=512", "f32", sec, reps, 1, 0.0, 0.0);

    free(t.idx); free(t.scores);
}


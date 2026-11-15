/* SPDX-License-Identifier: Apache-2.0 */
/* cases_qmat.c - packed GEMV/GEMM cases, including one routed expert. */
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "dsv4_kernels.h"
#include "dsv4_matmul.h"


typedef struct {
    QFix *f;
    int   ncalls, ncopies, nt, cur;
    float *x, *y;
} QCase;

static void qcase_run(void *p)
{
    QCase *q = (QCase *)p;
    const int base = (q->cur % q->ncopies) * q->ncalls;
    q->cur++;
    for (int c = 0; c < q->ncalls; c++)
        dsv4_mmq_n(q->y, q->x, &q->f[base + c].m, q->nt);
}

void bench_qmat(const char *name, const char *shape, int wdt,
                       int rows, int cols, int nt, int ncalls, int ncopies,
                       int target_ms)
{
    if (bench_skip(name)) return;

    QCase q;
    memset(&q, 0, sizeof q);
    q.ncalls  = ncalls;
    q.ncopies = ncopies;
    q.nt      = nt;
    q.f = (QFix *)calloc((size_t)ncalls * (size_t)ncopies, sizeof(QFix));
    if (!q.f) bench_die("allocation failed");
    for (int i = 0; i < ncalls * ncopies; i++)
        qfix_init(&q.f[i], wdt, rows, cols);

    q.x = (float *)bench_xalloc((size_t)nt * (size_t)cols * sizeof(float));
    q.y = (float *)bench_xalloc((size_t)nt * (size_t)rows * sizeof(float));
    bench_fill_f32(q.x, (size_t)nt * (size_t)cols, 1.0f);

    /* Touch every rotated copy once. Without this the first timed pass over a
     * copy pays its page faults and the average depends on how many copies
     * fit in the measurement window. */
    for (int i = 0; i < ncopies; i++) qcase_run(&q);

    int reps = 0;
    const double sec = bench_measure(qcase_run, &q, target_ms, &reps);
    const double flops = 2.0 * (double)nt * (double)rows * (double)cols
                       * (double)ncalls;
    const double bytes = qfix_bytes(&q.f[0], nt) * (double)ncalls;
    bench_report(name, shape, bench_fmt_name(wdt), sec, reps, ncalls, flops, bytes);

    for (int i = 0; i < ncalls * ncopies; i++) qfix_free(&q.f[i]);
    free(q.f);
    free(q.x);
    free(q.y);
}

/* One routed expert: gate/up matmuls, clamped SwiGLU, down matmul. */
typedef struct {
    QFix   w1, w3, w2;
    float *x, *g, *u, *mid, *y;
} TrioCase;

static void trio_run(void *p)
{
    TrioCase *t = (TrioCase *)p;
    dsv4_mmq(t->g, t->x, &t->w1.m);
    dsv4_mmq(t->u, t->x, &t->w3.m);
    dsv4_swiglu(t->mid, t->g, t->u, 2048, 10.0f);
    dsv4_mmq(t->y, t->mid, &t->w2.m);
}

void bench_expert_trio(void)
{
    if (bench_skip("expert_trio")) return;

    TrioCase t;
    memset(&t, 0, sizeof t);
    qfix_init(&t.w1, DSV4_WFP4, 2048, 4096);
    qfix_init(&t.w3, DSV4_WFP4, 2048, 4096);
    qfix_init(&t.w2, DSV4_WFP4, 4096, 2048);
    t.x   = (float *)bench_xalloc(4096u * sizeof(float));
    t.g   = (float *)bench_xalloc(2048u * sizeof(float));
    t.u   = (float *)bench_xalloc(2048u * sizeof(float));
    t.mid = (float *)bench_xalloc(2048u * sizeof(float));
    t.y   = (float *)bench_xalloc(4096u * sizeof(float));
    bench_fill_f32(t.x, 4096, 1.0f);

    int reps = 0;
    const double sec = bench_measure(trio_run, &t, 40, &reps);
    const double flops = 2.0 * 2048.0 * 4096.0 * 3.0 + 6.0 * 2048.0;
    const double bytes = (double)(t.w1.wb + t.w1.sb + t.w3.wb + t.w3.sb
                                + t.w2.wb + t.w2.sb)
                       + (double)(4096 + 3 * 2048 + 4096) * 4.0;
    bench_report("expert_trio", "3 x fp4", "fp4", sec, reps, 1, flops, bytes);

    qfix_free(&t.w1);
    qfix_free(&t.w3);
    qfix_free(&t.w2);
    free(t.x); free(t.g); free(t.u); free(t.mid); free(t.y);
}


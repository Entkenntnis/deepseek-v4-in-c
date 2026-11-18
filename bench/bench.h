/* SPDX-License-Identifier: Apache-2.0 */
/* bench.h - shared declarations for the compute-only microbenchmarks. */
#ifndef DSV4_BENCH_H
#define DSV4_BENCH_H

#include <stddef.h>
#include <stdint.h>

#include "dsv4.h"

typedef void (*RunFn)(void *);

/* main / timing */
double bench_now_s(void);
double bench_measure(RunFn fn, void *ctx, int target_ms, int *out_reps);
void   bench_report(const char *name, const char *shape, const char *fmt,
                    double sec, int reps, int calls, double flops,
                    double bytes);
int    bench_skip(const char *name);

/* fixtures */
void  *bench_xalloc(size_t n);
void   bench_die(const char *what);
uint32_t bench_rnd32(void);
float  bench_frnd(void);
void   bench_fill_f32(float *p, size_t n, float amp);

typedef struct {
    DSV4QMat m;
    void    *w;
    void    *s;
    size_t   wb, sb;         /* logical payload bytes                        */
    size_t   walloc, salloc; /* allocation bytes, rounded up for alignment   */
} QFix;

void        qfix_init(QFix *q, int wdt, int rows, int cols);
/* Same fixture, but the weight/scale allocations are aligned to `align` and
 * rounded up to a multiple of it. The Vulkan backend imports host pointers and
 * requires the driver's minImportedHostPointerAlignment. */
void        qfix_init_aligned(QFix *q, int wdt, int rows, int cols,
                              size_t align);
void        qfix_free(QFix *q);
double      qfix_bytes(const QFix *q, int nt);
const char *bench_fmt_name(int wdt);

/* cases */
void bench_qmat(const char *name, const char *shape, int wdt,
                int rows, int cols, int nt, int ncalls, int ncopies,
                int target_ms);
void bench_expert_trio(void);
void bench_rmsnorm(int n);
void bench_swiglu(int n);
void bench_route(void);
void bench_hc_pre(void);
void bench_hc_post(void);
void bench_sinkhorn(void);
void bench_attn(const char *name, int n);
void bench_indexer(void);
void bench_topk(void);
void bench_compress(const char *name, int ratio);
void bench_rope(void);

#endif /* DSV4_BENCH_H */

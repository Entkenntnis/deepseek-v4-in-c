/* SPDX-License-Identifier: Apache-2.0 */
/* bench/matmul_variants.c - compare experimental matmul loop variants.
 *
 * Built with -DDSV4_MATMUL_EXPERIMENTS. The variants themselves live in
 * bench/matmul_experiments.c, so src/core/dsv4_matmul.c contains only the
 * kernels the engine ships. Every variant is checked for bitwise equality
 * against the shipped kernel before it is timed; a variant that changes the
 * arithmetic is reported and the benchmark keeps going so the output is useful
 * even then.
 *
 * Progress is deliberately chatty: one line per repetition, per shape, with
 * every variant's running best. A silent benchmark on this machine is not
 * useful because the thermal state moves.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dsv4.h"

typedef void (*gemv_fn)(float *, const float *, const uint8_t *,
                        const uint8_t *, int, int, int, int);

void dsv4_matmul_fp4(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_matmul_fp8(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);

#ifdef DSV4_MATMUL_EXPERIMENTS
void dsv4_matmul_fp4_nodiv(float *, const float *, const uint8_t *,
                           const uint8_t *, int, int, int, int);
void dsv4_matmul_fp4_nodiv_lut(float *, const float *, const uint8_t *,
                               const uint8_t *, int, int, int, int);
void dsv4_matmul_fp4_nodiv2_lut(float *, const float *, const uint8_t *,
                                const uint8_t *, int, int, int, int);
void dsv4_matmul_fp4_32_nd2(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_nodiv(float *, const float *, const uint8_t *,
                           const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_unroll(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_dt(float *, const float *, const uint8_t *,
                        const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_f32(float *, const float *, const uint8_t *,
                         const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_avx2(float *, const float *, const uint8_t *,
                          const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_f32_arith(float *, const float *, const uint8_t *,
                               const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_f32_arith_red(float *, const float *, const uint8_t *,
                                   const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_f32_red(float *, const float *, const uint8_t *,
                             const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_f32_hybrid(float *, const float *, const uint8_t *,
                                const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_f32_rows(float *, const float *, const uint8_t *,
                              const uint8_t *, int, int, int, int);
void dsv4_matmul_fp8_f32_rows_fastsum(float *, const float *, const uint8_t *,
                                      const uint8_t *, int, int, int, int);
void dsv4_matmul_fp4_f32_red(float *, const float *, const uint8_t *,
                             const uint8_t *, int, int, int, int);
#endif

typedef struct {
    const char *name;
    gemv_fn fn;
} variant;

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int nthreads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static unsigned rng = 987654321u;
static unsigned nextr(void) { rng = rng * 1103515245u + 12345u; return rng; }

/* Uniform random bytes are not E4M3 weights. Their 6.25% subnormal rate makes
 * almost every 128-byte block take the hybrid kernel's exactness fallback,
 * while a sampled real wo_b tensor has 0.0145%. `--realistic` reproduces the
 * measured exponent histogram (from layers.0.attn.wo_b.weight, 33,554,432
 * bytes) so the benchmark measures the path the model actually takes. */
static const unsigned e4m3_exp_hist[16] = {
    5198, 5662, 10781, 21676, 43416, 85751, 172439, 343792,
    688449, 1373858, 2735911, 5258131, 8731359, 9623989, 4201159, 252861,
};

static void fill_fp8_realistic(uint8_t *W, size_t n)
{
    static uint8_t lut[4096];
    static int ready;
    if (!ready) {
        const unsigned total = 33554432u;
        unsigned cum = 0, e = 0;
        for (int u = 0; u < 4096; u++) {
            const unsigned target = (unsigned)((uint64_t)u * total / 4096u);
            while (e < 15 && cum + e4m3_exp_hist[e] <= target) {
                cum += e4m3_exp_hist[e];
                e++;
            }
            lut[u] = (uint8_t)e;
        }
        ready = 1;
    }
    for (size_t i = 0; i < n; i++) {
        const unsigned r = nextr();
        const unsigned e = lut[(r >> 20) & 0xFFFu];
        unsigned m = (r >> 17) & 7u;
        const unsigned s = (r >> 16) & 1u;
        if (e == 15 && m == 7) m = 6;   /* no NaNs in the sampled tensor */
        W[i] = (uint8_t)((s << 7) | (e << 3) | m);
    }
}

static void fill_inputs(float *x, int in, uint8_t *W, size_t wbytes,
                        uint8_t *S, size_t sbytes, int fp8, int realistic)
{
    for (int i = 0; i < in; i++)
        x[i] = (float)((int)(nextr() >> 16 & 0xffff) - 32768) * 1e-4f;
    if (fp8 && realistic) {
        fill_fp8_realistic(W, wbytes);
    } else {
        for (size_t i = 0; i < wbytes; i++) {
            uint8_t b = (uint8_t)(nextr() >> 16);
            if (fp8 && (b & 0x7Fu) == 0x7Fu) b &= 0xFEu;   /* no e4m3 NaNs */
            W[i] = b;
        }
    }
    for (size_t i = 0; i < sbytes; i++)
        S[i] = (uint8_t)(124 + (nextr() >> 16) % 7);   /* 2^-3 .. 2^3 */
}

static double l2_rel(const float *a, const float *b, int n)
{
    double e = 0.0, r = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        e += d * d;
        r += (double)a[i] * (double)a[i];
    }
    return sqrt(e / (r > 0.0 ? r : 1.0));
}

static void run_shape(const char *name, int in, int out, int blk_r, int blk_c,
                      int fp8, int realistic, const variant *vars, int nvars,
                      int reps)
{
    const int sc = (in + blk_c - 1) / blk_c;
    const size_t wbytes = fp8 ? (size_t)out * (size_t)in
                              : (size_t)out * (size_t)in / 2;
    const size_t sbytes = (size_t)((out + blk_r - 1) / blk_r) * (size_t)sc;

    printf("\n=== %s (in=%d out=%d block=%dx%d, %s) ===\n",
           name, in, out, blk_r, blk_c, fp8 ? "FP8" : "FP4");
    fflush(stdout);

    float *x = malloc((size_t)in * sizeof *x);
    float *yref = malloc((size_t)out * sizeof *yref);
    float *ytmp = malloc((size_t)out * sizeof *ytmp);
    uint8_t *W = malloc(wbytes), *S = malloc(sbytes);
    if (!x || !yref || !ytmp || !W || !S) {
        printf("  out of memory\n");
        exit(1);
    }
    fill_inputs(x, in, W, wbytes, S, sbytes, fp8, realistic);

    double best[8];
    if (nvars > (int)(sizeof best / sizeof best[0])) {
        printf("  too many variants\n");
        exit(1);
    }
    for (int v = 0; v < nvars; v++) best[v] = 1e30;

    vars[0].fn(yref, x, W, S, in, out, blk_r, blk_c);
    printf("  correctness (bitwise vs baseline):\n");
    for (int v = 0; v < nvars; v++) {
        memset(ytmp, 0, (size_t)out * sizeof *ytmp);
        vars[v].fn(ytmp, x, W, S, in, out, blk_r, blk_c);
        const int same = memcmp(yref, ytmp, (size_t)out * sizeof *yref) == 0;
        printf("    %-18s %s", vars[v].name, same ? "bitwise" : "DIFFERS");
        if (!same) printf("  (L2 rel %.3e)", l2_rel(yref, ytmp, out));
        printf("\n");
    }
    fflush(stdout);

    const double flop = 2.0 * (double)in * (double)out;
    for (int r = 0; r < reps; r++) {
        const int start = r % nvars;
        printf("  rep %2d/%d:", r + 1, reps);
        for (int k = 0; k < nvars; k++) {
            const int v = (start + k) % nvars;
            const double t0 = now();
            vars[v].fn(ytmp, x, W, S, in, out, blk_r, blk_c);
            const double t1 = now();
            if (t1 - t0 < best[v]) best[v] = t1 - t0;
            printf("  %s %.1f", vars[v].name, flop / best[v] * 1e-9);
        }
        printf("  GF/s\n");
        fflush(stdout);
    }

    printf("  best:\n");
    for (int v = 0; v < nvars; v++)
        printf("    %-18s %8.1f GF/s   %.2fx\n",
               vars[v].name, flop / best[v] * 1e-9, best[0] / best[v]);
    fflush(stdout);

    free(x); free(yref); free(ytmp); free(W); free(S);
}

int main(int argc, char **argv)
{
    int reps = 10, realistic = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--realistic")) realistic = 1;
        else reps = atoi(argv[i]);
    }
    printf("matmul loop variants, %d thread(s), %s E4M3 input\n",
           nthreads(), realistic ? "realistic" : "uniform");

#ifdef DSV4_MATMUL_EXPERIMENTS
    const variant fp4[] = {
        { "baseline",       dsv4_matmul_fp4 },
        { "nodiv2+lut",     dsv4_matmul_fp4_nodiv2_lut },
        { "rowblock",       dsv4_matmul_fp4_32_nd2 },
        { "fast-hsum",      dsv4_matmul_fp4_f32_red },
    };
    const variant fp8[] = {
        { "f32inner",       dsv4_matmul_fp8_f32 },
        { "fast-hsum",      dsv4_matmul_fp8_f32_red },
        { "arith",          dsv4_matmul_fp8_f32_arith },
        { "arith+fastsum",  dsv4_matmul_fp8_f32_arith_red },
        { "hybrid",         dsv4_matmul_fp8_f32_hybrid },
        { "rowscales",      dsv4_matmul_fp8_f32_rows },
        { "rows+fastsum",   dsv4_matmul_fp8_f32_rows_fastsum },
    };
    run_shape("fp4 expert w1 2048x4096", 4096, 2048, 1, 32, 0, realistic,
              fp4, (int)(sizeof fp4 / sizeof fp4[0]), reps);
    run_shape("fp4 expert w2 4096x2048", 2048, 4096, 1, 32, 0, realistic,
              fp4, (int)(sizeof fp4 / sizeof fp4[0]), reps);
    run_shape("fp8 attn wo_b 4096x8192", 8192, 4096, 128, 128, 1, realistic,
              fp8, (int)(sizeof fp8 / sizeof fp8[0]), reps);
    run_shape("fp8 attn wq_b 32768x1024", 1024, 32768, 128, 128, 1, realistic,
              fp8, (int)(sizeof fp8 / sizeof fp8[0]), reps);
#else
    printf("built without -DDSV4_MATMUL_EXPERIMENTS\n");
#endif
    return 0;
}

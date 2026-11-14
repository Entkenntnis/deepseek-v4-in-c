/* SPDX-License-Identifier: Apache-2.0 */
/* bench/prefill_gemm.c - what does a dense matmul gain from a 32-token batch?
 *
 * THE QUESTION
 *   The dense prefill path wants to feed a whole prompt chunk through one
 *   weight matrix. A GEMV reads the weights once and does one multiply-add per
 *   weight; a batched kernel should read them once and do nt of them. This
 *   bench measures the per-token cost of that change on the real Flash shapes.
 *
 * THE TRAP THIS BENCH AVOIDS
 *   Timing one matrix in a tight loop keeps it in L3 and measures a program
 *   nobody runs. The engine touches ~165 MB of other weights between two uses
 *   of the same matrix, so every call here reads a DIFFERENT cold copy. Copies
 *   are sized to exceed this machine's L3 by a comfortable margin.
 *
 * REALISTIC FP8
 *   Uniform random bytes are 6.25% subnormal and send almost every 128-byte
 *   block through the exactness fallback. A sampled wo_b tensor is 0.0145%.
 *   The histogram below is the one bench/matmul_variants uses, so the measured
 *   path is the one the checkpoint actually takes.
 *
 * WHAT IT DOES NOT CLAIM
 *   This is a kernel measurement. It says nothing about disk, expert routing,
 *   attention state, or end-to-end TTFT.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "dsv4_prefill_gemm.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static unsigned rng = 20261114u;
static unsigned nextr(void) { rng = rng * 1103515245u + 12345u; return rng; }

/* Measured E4M3 exponent histogram of layers.0.attn.wo_b.weight. */
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

static int nthreads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static void run_shape(const char *name, int in, int out, int fp4, int bf16,
                      const int *NT, int nnt, int reps)
{
    const size_t wbytes = bf16 ? (size_t)out * in * 2
                        : fp4  ? (size_t)out * in / 2
                               : (size_t)out * in;
    const int sc = bf16 ? 0 : (in + (fp4 ? 31 : 127)) / (fp4 ? 32 : 128);
    const int srows = bf16 ? 0 : (out + (fp4 ? 0 : 127)) / (fp4 ? 1 : 128);
    const size_t sbytes = (size_t)srows * sc;

    /* Enough distinct copies that a call cannot find the previous one in L3. */
    size_t want = (size_t)64 << 20;
    int ncopy = (int)(want / (wbytes ? wbytes : 1));
    if (ncopy < 4) ncopy = 4;
    if (ncopy > 16) ncopy = 16;

    uint8_t **W = calloc((size_t)ncopy, sizeof *W);
    uint8_t **S = calloc((size_t)ncopy, sizeof *S);
    if (!W || !S) { printf("  out of memory\n"); free(W); free(S); return; }

    for (int c = 0; c < ncopy; c++) {
        W[c] = malloc(wbytes ? wbytes : 1);
        S[c] = sbytes ? malloc(sbytes) : NULL;
        if (!W[c] || (sbytes && !S[c])) {
            printf("  out of memory\n");
            for (int j = 0; j <= c; j++) { free(W[j]); free(S[j]); }
            free(W); free(S);
            return;
        }
        if (bf16) {
            uint16_t *p = (uint16_t *)W[c];
            for (size_t i = 0; i < (size_t)out * in; i++)
                p[i] = (uint16_t)(0x3800u | (nextr() & 0x03FFu));
        } else if (fp4) {
            for (size_t i = 0; i < wbytes; i++) W[c][i] = (uint8_t)(nextr() >> 16);
        } else {
            fill_fp8_realistic(W[c], wbytes);
        }
        for (size_t i = 0; i < sbytes; i++)
            S[c][i] = (uint8_t)(124 + (nextr() >> 16) % 7);
    }

    const int maxnt = NT[nnt - 1];
    float *x = malloc((size_t)maxnt * in * sizeof *x);
    float *y = malloc((size_t)maxnt * out * sizeof *y);
    float *yg = malloc((size_t)maxnt * out * sizeof *yg);
    if (!x || !y || !yg) { printf("  out of memory\n"); return; }
    for (int t = 0; t < maxnt; t++)
        for (int i = 0; i < in; i++)
            x[(size_t)t * in + i] =
                (float)((int)((nextr() >> 16) & 0x3ff) - 512) * 0.001f;

    printf("\n%s  (%d x %d, %s, %d copies of %.1f MB)\n",
           name, out, in, bf16 ? "BF16" : fp4 ? "FP4" : "FP8",
           ncopy, (double)wbytes / 1048576.0);
    printf("  %4s  %10s  %10s  %8s  %s\n",
           "nt", "GEMV/tok", "batch/tok", "speedup", "verdict");

    for (int k = 0; k < nnt; k++) {
        const int nt = NT[k];
        DSV4QMat m = { W[0], S[0], bf16 ? DSV4_WBF16 : fp4 ? DSV4_WFP4 : DSV4_WFP8,
                       out, in, fp4 ? 1 : (bf16 ? 0 : 128),
                       fp4 ? 32 : (bf16 ? 0 : 128) };

        double best_g = 1e30, best_b = 1e30;
        for (int r = 0; r < reps; r++) {
            const int c = (r + 1) % ncopy;
            m.w = W[c];
            m.s = S[c];

            double t0 = now();
            for (int t = 0; t < nt; t++)
                dsv4_mmq(y + (size_t)t * out, x + (size_t)t * in, &m);
            double t1 = now();
            dsv4_mmq_n(yg, x, &m, nt);
            double t2 = now();

            if (t1 - t0 < best_g) best_g = t1 - t0;
            if (t2 - t1 < best_b) best_b = t2 - t1;
        }

        const double gt = best_g / (double)nt;      /* per token */
        const double bt = best_b / (double)nt;
        const double gf = 2.0 * (double)in * (double)out / gt * 1e-9;
        const double bf = 2.0 * (double)in * (double)out / bt * 1e-9;
        printf("  %4d  %8.1f GF  %8.1f GF  %7.2fx  %s\n",
               nt, gf, bf, gt / bt,
               gt / bt >= 1.5 ? "worth it" :
               gt / bt >= 1.2 ? "marginal" : "not worth it");
    }

    free(x); free(y); free(yg);
    for (int c = 0; c < ncopy; c++) { free(W[c]); free(S[c]); }
    free(W); free(S);
}

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    static const int NT[] = { 1, 4, 8, 16, 32, 64 };
    const int nnt = (int)(sizeof NT / sizeof NT[0]);

    printf("dense prefill matmul: per-token GEMV vs batched\n");
    printf("AVX2+FMA required by dsv4.h, %d OpenMP thread(s), "
           "cold weights on every call\n", nthreads());
    printf("Speedup is per token. 1.00x means the batch bought nothing.\n");

    run_shape("fp8 attn wo_b", 8192, 4096, 0, 0, NT, nnt, reps);
    run_shape("fp8 attn wq_b", 1024, 32768, 0, 0, NT, nnt, reps);
    run_shape("fp8 attn wq_a", 4096, 1024, 0, 0, NT, nnt, reps);
    run_shape("fp8 attn wkv",  4096, 512,  0, 0, NT, nnt, reps);
    run_shape("fp8 shared w1", 4096, 2048, 0, 0, NT, nnt, reps);
    run_shape("bf16 router",   4096, 256,  0, 1, NT, nnt, reps);

    return 0;
}

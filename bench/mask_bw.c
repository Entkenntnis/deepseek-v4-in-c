/* SPDX-License-Identifier: Apache-2.0 */
/* mask_bw.c - what the bind-time FP8 special-block mask is worth in the kernel.
 *
 * The production FP8 row kernel used to scan every 128-byte block with four
 * vpcmpeqb/vpand/vpor/vptest groups before choosing between the closed-form
 * decode and the exact gather path. DSV4QMat.special now carries that answer as
 * one byte per block, built once at bind time.
 *
 * This benchmark isolates that change: same matrix, same special distribution,
 * same dsv4_mmq call, only `special` is NULL or not. It also checks that both
 * paths produce the same floats, because a mask that disagrees with the scan is
 * a wrong kernel, not a faster one.
 *
 * The full-model gain is much smaller than the kernel number here (see
 * BENCHMARK.md): in a decode the scan's loads and integer ops hide in the
 * shadow of the weight-streaming latency, so this harness answers "how much
 * work is the scan", not "how much faster is the model".
 *
 * Run with OMP_NUM_THREADS=4 and the four P-cores, like matmul_bw:
 *
 *     OMP_NUM_THREADS=4 taskset -c 0-3 ./bin/mask_bw 8192 4096 200 4000
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4.h"

void dsv4_mmq(float *y, const float *x, const DSV4QMat *m);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static unsigned rng = 1234567u;
static unsigned nextr(void) { rng = rng * 1103515245u + 12345u; return rng; }

int main(int argc, char **argv)
{
    const int in     = argc > 1 ? atoi(argv[1]) : 8192;
    const int out    = argc > 2 ? atoi(argv[2]) : 4096;
    const int reps   = argc > 3 ? atoi(argv[3]) : 200;
    const int inject = argc > 4 ? atoi(argv[4]) : 4000;  /* special bytes */

    const size_t wb = (size_t)out * in;
    const int sc = in / 128;
    const size_t sb = (size_t)(out / 128) * (size_t)sc;
    float *x = (float *)malloc((size_t)in * sizeof *x);
    float *ya = (float *)malloc((size_t)out * sizeof *ya);
    float *yb = (float *)malloc((size_t)out * sizeof *yb);
    uint8_t *W = (uint8_t *)malloc(wb), *S = (uint8_t *)malloc(sb);
    uint8_t *mask = (uint8_t *)calloc((size_t)out * (size_t)sc, 1);
    if (!x || !ya || !yb || !W || !S || !mask) {
        printf("  out of memory\n");
        return 1;
    }

    for (int i = 0; i < in; i++)
        x[i] = (float)((int)(nextr() >> 16 & 0xffff) - 32768) * 1e-3f;
    /* Normal E4M3 values first, then a controlled number of zeros/NaNs. The
     * real checkpoint is 0.02% special bytes; uniform random bytes would be
     * 6.25% and would make almost every block take the gather path. */
    memset(W, 0x38, wb);                       /* 1.0 */
    memset(S, 127, sb);                        /* 2^0 */
    for (int i = 0; i < inject; i++) {
        const size_t p = (size_t)(nextr() >> 8) % wb;
        W[p] = (nextr() & 1) ? 0x00 : 0x7F;    /* zero or NaN */
    }
    for (int o = 0; o < out; o++) {
        for (int b = 0; b < sc; b++) {
            const uint8_t *p = W + (size_t)o * in + (size_t)b * 128;
            int sp = 0;
            for (int j = 0; j < 128; j++)
                if ((p[j] & 0x78u) == 0 || (p[j] & 0x7Fu) == 0x7Fu) {
                    sp = 1;
                    break;
                }
            mask[(size_t)o * (size_t)sc + b] = (uint8_t)sp;
        }
    }
    size_t spblocks = 0;
    for (size_t i = 0; i < (size_t)out * (size_t)sc; i++) spblocks += mask[i];

    DSV4QMat scan = { W, S, DSV4_WFP8, out, in, 128, 128, NULL };
    DSV4QMat masked = scan;
    masked.special = mask;

    double best[2] = { 1e30, 1e30 };
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        dsv4_mmq(ya, x, &scan);
        double t1 = now();
        dsv4_mmq(yb, x, &masked);
        double t2 = now();
        if (t1 - t0 < best[0]) best[0] = t1 - t0;
        if (t2 - t1 < best[1]) best[1] = t2 - t1;
    }

    int bad = 0;
    for (int i = 0; i < out; i++)
        if (memcmp(&ya[i], &yb[i], sizeof ya[i]) != 0) bad++;

    const double flop = 2.0 * (double)in * (double)out;
    printf("%dx%d  special blocks %zu/%zu (%.2f%%)  bitwise %s\n",
           in, out, spblocks, (size_t)out * (size_t)sc,
           100.0 * (double)spblocks / (double)((size_t)out * (size_t)sc),
           bad ? "DIFFERS" : "same");
    printf("scan   %6.1f GF/s\n", flop / best[0] * 1e-9);
    printf("masked %6.1f GF/s   %.3fx\n",
           flop / best[1] * 1e-9, best[0] / best[1]);

    free(x); free(ya); free(yb); free(W); free(S); free(mask);
    return bad != 0;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* bench.c - compute-only microbenchmarks for DeepSeek-V4-Flash kernels.
 *
 * One process, one table, no disk. Every weight matrix is allocated in RAM and
 * filled with a deterministic, realistic fixture before it is timed. The
 * benchmark links against src/10_compute and src/20_runtime only; it never opens a
 * checkpoint, a trunk or the expert cache. A regression here is a CPU/kernel
 * regression, not an I/O one.
 *
 * The working set of each GEMV/GEMM is rotated over several matrix copies so
 * it exceeds the 12 MB L3 of the target CPU; otherwise the benchmark would
 * report LLC bandwidth instead of the RAM-fed decode path.
 *
 * USAGE
 *   ./bin/bench [--filter SUBSTR] [--threads N] [--pin off|all|p|e]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "dsv4_threads.h"

static const char *g_filter;

double bench_measure(RunFn fn, void *ctx, int target_ms, int *out_reps)
{
    fn(ctx);
    const double t0 = bench_now_s();
    int reps = 0;
    double elapsed;
    do {
        fn(ctx);
        reps++;
        elapsed = bench_now_s() - t0;
    } while ((elapsed * 1000.0 < (double)target_ms || reps < 3)
             && reps < 200000);
    *out_reps = reps;
    return elapsed;
}

void bench_report(const char *name, const char *shape, const char *fmt,
                  double sec, int reps, int calls, double flops, double bytes)
{
    const double per_ms = sec / ((double)reps * (double)calls) * 1e3;
    const double gfs = flops > 0.0 ? flops * (double)reps / sec / 1e9 : 0.0;
    const double gbs = bytes > 0.0 ? bytes * (double)reps / sec / 1e9 : 0.0;

    printf("%-30s %-20s %-5s %10.4f", name, shape, fmt, per_ms);
    if (flops > 0.0) printf(" %9.1f", gfs); else printf(" %9s", "-");
    if (bytes > 0.0) printf(" %9.2f", gbs); else printf(" %9s", "-");
    printf("\n");
    fflush(stdout);
}

int bench_skip(const char *name)
{
    return g_filter && !strstr(name, g_filter);
}

static void usage(const char *argv0)
{
    fprintf(stderr, "usage: %s [--filter SUBSTR] [--threads N] "
                    "[--pin off|all|p|e]\n", argv0);
}

int main(int argc, char **argv)
{
    int pin = DSV4_PIN_P;
    int threads = 4;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--filter") && i + 1 < argc) {
            g_filter = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pin") && i + 1 < argc) {
            const int p = dsv4_threads_parse_pin(argv[++i]);
            if (p < 0) { usage(argv[0]); return 1; }
            pin = p;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    const DSV4ThreadOpts to = { pin, threads, 0, DSV4_PIN_OFF };
    dsv4_threads_apply(&to);

    printf("DeepSeek-V4 Flash microbenchmarks (compute only, weights resident)\n");
    printf("%-30s %-20s %-5s %10s %9s %9s\n",
           "kernel", "shape", "fmt", "ms/call", "GF/s", "GB/s");

    const double t0 = bench_now_s();

    /* routed expert, decode */
    bench_qmat("expert_w1",   "2048x4096", DSV4_WFP4, 2048, 4096, 1, 1, 4, 40);
    bench_qmat("expert_w2",   "4096x2048", DSV4_WFP4, 4096, 2048, 1, 1, 4, 40);
    bench_expert_trio();

    /* attention projections, decode */
    bench_qmat("wq_a",        "1024x4096", DSV4_WFP8, 1024, 4096, 1, 1, 4, 40);
    bench_qmat("wq_b",        "32768x1024", DSV4_WFP8, 32768, 1024, 1, 1, 2, 40);
    bench_qmat("wkv",         "512x4096",  DSV4_WFP8, 512, 4096, 1, 1, 8, 40);
    bench_qmat("wo_a_group",  "1024x4096", DSV4_WFP8, 1024, 4096, 1, 1, 4, 40);
    bench_qmat("wo_a_8groups", "8x1024x4096", DSV4_WFP8, 1024, 4096, 1, 8, 1, 40);
    bench_qmat("wo_b",        "4096x8192", DSV4_WFP8, 4096, 8192, 1, 1, 2, 40);

    /* shared expert, decode */
    bench_qmat("shared_gate_up", "2048x4096", DSV4_WFP8, 2048, 4096, 1, 1, 2, 40);
    bench_qmat("shared_down", "4096x2048", DSV4_WFP8, 4096, 2048, 1, 1, 2, 40);

    /* indexer projections + router + head */
    bench_qmat("indexer_wq_b", "8192x1024", DSV4_WFP8, 8192, 1024, 1, 1, 2, 40);
    bench_qmat("indexer_weights_proj", "64x4096", DSV4_WBF16, 64, 4096, 1, 1, 8, 40);
    bench_qmat("gate",        "256x4096", DSV4_WBF16, 256, 4096, 1, 1, 8, 40);
    bench_qmat("head",        "129280x4096", DSV4_WBF16, 129280, 4096, 1, 1, 1, 40);

    /* prefill GEMM, at the token counts the engine actually tiles with: a
     * routed expert sees ~5 tokens of the block (54 tokens x top-6 / 256
     * experts, deduped), the dense projections see the whole 54-token block. */
    bench_qmat("expert_w1_nt5", "2048x4096 nt5", DSV4_WFP4, 2048, 4096, 5, 1, 4, 40);
    bench_qmat("expert_w1_nt16", "2048x4096 nt16", DSV4_WFP4, 2048, 4096, 16, 1, 4, 40);
    bench_qmat("expert_w1_nt54", "2048x4096 nt54", DSV4_WFP4, 2048, 4096, 54, 1, 4, 40);
    bench_qmat("expert_w1_nt128", "2048x4096 nt128", DSV4_WFP4, 2048, 4096, 128, 1, 4, 40);
    bench_qmat("expert_w1_nt204", "2048x4096 nt204", DSV4_WFP4, 2048, 4096, 204, 1, 2, 40);
    bench_qmat("expert_w2_nt54", "4096x2048 nt54", DSV4_WFP4, 4096, 2048, 54, 1, 4, 40);
    bench_qmat("expert_w2_nt204", "4096x2048 nt204", DSV4_WFP4, 4096, 2048, 204, 1, 2, 40);
    bench_qmat("wq_a_nt54",   "1024x4096 nt54", DSV4_WFP8, 1024, 4096, 54, 1, 4, 40);
    bench_qmat("wq_b_nt16",   "32768x1024 nt16", DSV4_WFP8, 32768, 1024, 16, 1, 2, 40);
    bench_qmat("wq_b_nt54",   "32768x1024 nt54", DSV4_WFP8, 32768, 1024, 54, 1, 2, 40);
    bench_qmat("wo_a_nt16",   "1024x4096 nt16", DSV4_WFP8, 1024, 4096, 16, 1, 4, 40);
    bench_qmat("wo_a_nt54",   "1024x4096 nt54", DSV4_WFP8, 1024, 4096, 54, 1, 4, 40);
    bench_qmat("wo_b_nt54",   "4096x8192 nt54", DSV4_WFP8, 4096, 8192, 54, 1, 2, 40);
    bench_qmat("shared_gate_up_nt54", "2048x4096 nt54", DSV4_WFP8, 2048, 4096, 54, 1, 2, 40);
    bench_qmat("shared_down_nt54", "4096x2048 nt54", DSV4_WFP8, 4096, 2048, 54, 1, 2, 40);

    /* BF16 at the prefill batch size: router and indexer weights. */
    bench_qmat("gate_nt54", "256x4096 nt54", DSV4_WBF16, 256, 4096, 54, 1, 4, 40);
    bench_qmat("weights_proj_nt54", "64x4096 nt54", DSV4_WBF16, 64, 4096, 54, 1, 8, 40);

    /* Same shapes at the 204-token block the engine uses for longer prompts. */
    bench_qmat("wq_a_nt204", "1024x4096 nt204", DSV4_WFP8, 1024, 4096, 204, 1, 2, 40);
    bench_qmat("wo_a_nt204", "1024x4096 nt204", DSV4_WFP8, 1024, 4096, 204, 1, 2, 40);
    bench_qmat("wo_b_nt204", "4096x8192 nt204", DSV4_WFP8, 4096, 8192, 204, 1, 2, 40);
    bench_qmat("shared_gate_up_nt204", "2048x4096 nt204", DSV4_WFP8, 2048, 4096, 204, 1, 1, 40);

    /* attention / indexer / compressor */
    bench_attn("sparse_attn_idx", 640);
    bench_attn("sparse_attn_win", 128);
    bench_indexer();
    bench_topk();
    bench_compress("compress_r4", 4);
    bench_compress("compress_r128", 128);

    /* elementwise */
    bench_rmsnorm(4096);
    bench_rmsnorm(1024);
    bench_swiglu(2048);
    bench_route();
    bench_hc_pre();
    bench_hc_post();
    bench_sinkhorn();
    bench_rope();

    printf("total: %.3f s\n", bench_now_s() - t0);
    return 0;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* expert_bench.c - routed-expert bucket microbenchmark, CPU versus Vulkan.
 *
 * The unit is one expert applied to `n` tokens: the same gather, w1/w3,
 * clamped SwiGLU, w2 and scatter that dsv4_prefill_expert_bucket() runs. It is
 * deliberately not a bare GEMV/GEMM benchmark: below the Vulkan dispatch
 * threshold the fixed costs (host copies, descriptor update, command-buffer
 * record, submit and fence wait) are a large part of the answer.
 *
 * The benchmark builds a minimal DSV4Prefill context with a synthetic Flash
 * geometry and resident FP4 expert weights. No checkpoint, trunk or cache is
 * opened. The Vulkan path imports those host weight allocations through
 * VK_EXT_external_memory_host, exactly as the prefill backend does.
 *
 * Output:
 *   - CPU: the production bucket seam, forced onto the CPU kernels.
 *   - GPU: the fused expert kernel called directly for every n, including the
 *     n < 9 sizes that the automatic route keeps on the CPU.
 *   - dispatch: the production seam with the Vulkan backend available and the
 *     automatic route, i.e. what the engine chooses without scheduler state.
 *   - overlap: the fused kernel with CPU fill buckets run during its fence
 *     wait, matching the union scheduler's GPU/CPU overlap.
 *
 * USAGE
 *   ./bin/bench_expert [--filter SUBSTR] [--threads N] [--pin off|all|p|e]
 *                      [--copies N] [--target-ms N] [--no-gpu] [--breakdown]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"
#include "dsv4_prefill_vk_int.h"
#include "dsv4_threads.h"

#define EB_HIDDEN 4096
#define EB_INTER  2048
#define EB_MAXTOK 128
#define EB_TOPK   6

typedef struct {
    QFix       q1, q3, q2;
    DSV4ExpertW ew;
} ExpertFix;

typedef struct {
    DSV4Prefill *p;
    ExpertFix   *ex;
    int          ncopies;
    int          cur;
    int          n;
    int          tok[EB_MAXTOK];
    int          ks[EB_MAXTOK];

    /* Overlap fill queue. Counts stay inside the contested band so the fill
     * mirrors union_fill_cpu()'s CPU buckets. */
    int          fill_q;
    int          fill_nq;
    long         fill_total;
} Case;

typedef struct {
    DSV4PrefillStats    st;
    DSV4PrefillVkStats  vk[DSV4_VK_KIND_COUNT];
} Snap;

typedef struct {
    double cpu_ms, gpu_ms, disp_ms, ovl_ms;
    double cpu_mm, cpu_sw, cpu_pack;
    double vk_desc, vk_cin, vk_rec, vk_sub, vk_wait, vk_cout, vk_total;
    double vk_fill;
    double ovl_absorbed;
} Row;

static const char *g_filter;
static int g_breakdown;

static int skip_case(const char *name)
{
    return g_filter && !strstr(name, g_filter);
}

static void die(const char *what)
{
    fprintf(stderr, "bench_expert: %s\n", what);
    exit(1);
}

/* ------------------------------------------------------------- fixtures --- */

static void expert_fix_init(ExpertFix *e, size_t align)
{
    qfix_init_aligned(&e->q1, DSV4_WFP4, EB_INTER, EB_HIDDEN, align);
    qfix_init_aligned(&e->q3, DSV4_WFP4, EB_INTER, EB_HIDDEN, align);
    qfix_init_aligned(&e->q2, DSV4_WFP4, EB_HIDDEN, EB_INTER, align);
    e->ew.w1 = e->q1.m;
    e->ew.w3 = e->q3.m;
    e->ew.w2 = e->q2.m;
}

static void expert_fix_free(ExpertFix *e)
{
    qfix_free(&e->q1);
    qfix_free(&e->q3);
    qfix_free(&e->q2);
}

/* ------------------------------------------------------------- context ---- */

/* The prefill context only validates that begin/end exist; this benchmark
 * calls the bucket seam directly and never runs a whole layer. */
static int src_begin(void *ctx, int layer, const int *experts, int n,
                     const DSV4ExpertW **out)
{
    (void)ctx; (void)layer; (void)experts; (void)n; (void)out;
    return 0;
}

static int src_end(void *ctx, const DSV4ExpertW **out)
{
    (void)ctx; (void)out;
    return 0;
}

static void setup_cfg(DSV4Cfg *c, int *ratios)
{
    memset(c, 0, sizeof *c);
    c->hidden = EB_HIDDEN;
    c->n_layers = 1;
    c->vocab = 129280;
    c->rms_eps = 1e-6f;
    c->max_pos = 8192;

    c->n_heads = 64;
    c->n_kv_heads = 1;
    c->head_dim = 512;
    c->q_lora = 1024;
    c->o_lora = 1024;
    c->o_groups = 8;
    c->qk_rope = 64;
    c->sliding_window = 128;

    c->index_n_heads = 64;
    c->index_head_dim = 128;
    c->index_topk = 512;

    c->n_compress = 1;
    ratios[0] = 0;
    c->compress_ratios = ratios;
    c->compress_rope_theta = 160000.0f;

    c->hc_mult = 4;
    c->hc_sinkhorn_iters = 20;
    c->hc_eps = 1e-6f;

    c->n_experts = 256;
    c->topk = EB_TOPK;
    c->n_shared = 1;
    c->moe_inter = EB_INTER;
    c->routed_scale = 1.5f;
    c->norm_topk = 1;
    c->scoring_func = DSV4_SCORE_SQRTSOFTPLUS;
    c->topk_method = DSV4_TOPK_NOAUX_TC;

    c->swiglu_limit = 10.0f;

    c->rope_theta = 10000.0f;
    c->yarn_factor = 16.0f;
    c->yarn_beta_fast = 32.0f;
    c->yarn_beta_slow = 1.0f;
    c->yarn_orig_ctx = 65536;

    c->wblock_m = 128;
    c->wblock_n = 128;
}

static DSV4Prefill *make_prefill(const DSV4Cfg *c, int max_pos)
{
    DSV4ExpertSrc src;
    memset(&src, 0, sizeof src);
    src.begin = src_begin;
    src.end = src_end;

    const DSV4PrefillOpts opt = { .max_pos = max_pos };
    DSV4Prefill *p = NULL;
    if (dsv4_prefill_init(&p, c, NULL, NULL, &src, NULL, &opt) != 0)
        die("could not create a prefill context");
    if (dsv4_prefill_reserve(p, EB_MAXTOK) != 0)
        die("could not reserve prefill block buffers");
    return p;
}

static int import_qfix(DSV4Prefill *p, QFix *q)
{
    if (!p || !p->vk) return 0;
    if (q->w && dsv4_prefill_vk_region_import(p->vk, q->w, q->walloc) != 0)
        return -1;
    if (q->s && dsv4_prefill_vk_region_import(p->vk, q->s, q->salloc) != 0)
        return -1;
    return 0;
}

static int import_expert(DSV4Prefill *p, ExpertFix *e)
{
    return (import_qfix(p, &e->q1) || import_qfix(p, &e->q3)
            || import_qfix(p, &e->q2)) ? -1 : 0;
}

static void snapshot(DSV4Prefill *p, Snap *s)
{
    s->st = p->st;
    dsv4_prefill_vk_stats(p->vk, s->vk);
}

static double delta_ms(double after, double before, int reps)
{
    return reps > 0 ? (after - before) * 1e3 / (double)reps : 0.0;
}

/* --------------------------------------------------------------- runners -- */

static void gather(Case *c)
{
    DSV4Prefill *p = c->p;
    const int d = p->c->hidden;
    for (int j = 0; j < c->n; j++)
        memcpy(p->ex_x + (size_t)j * d,
               p->x1 + (size_t)c->tok[j] * d, (size_t)d * sizeof(float));
}

static void scatter(Case *c)
{
    DSV4Prefill *p = c->p;
    const int d = p->c->hidden;
    for (int j = 0; j < c->n; j++) {
        DSV4PrefillTok *s = &p->scratch[c->tok[j]];
        memcpy(s->expert_acc + (size_t)c->ks[j] * s->expert_acc_stride,
               p->ex_out + (size_t)j * d, (size_t)d * sizeof(float));
    }
}

static void run_cpu(void *vp)
{
    Case *c = (Case *)vp;
    const int base = c->cur++;
    DSV4ExpertW *ew = &c->ex[base % c->ncopies].ew;
    dsv4_prefill_expert_bucket(c->p, ew, 0, c->n, c->tok, c->ks,
                               DSV4_EXPERT_ROUTE_CPU, NULL, NULL);
}

static void run_gpu(void *vp)
{
    Case *c = (Case *)vp;
    const int base = c->cur++;
    DSV4ExpertW *ew = &c->ex[base % c->ncopies].ew;
    gather(c);
    dsv4_prefill_vk_expert_fused(c->p->vk, c->p->ex_out, c->p->ex_x,
                                 &ew->w1, &ew->w3, &ew->w2, c->n,
                                 c->p->c->moe_inter, c->p->c->swiglu_limit,
                                 NULL, NULL, NULL);
    scatter(c);
}

static void run_dispatch(void *vp)
{
    Case *c = (Case *)vp;
    const int base = c->cur++;
    DSV4ExpertW *ew = &c->ex[base % c->ncopies].ew;
    dsv4_prefill_expert_bucket(c->p, ew, 0, c->n, c->tok, c->ks,
                               DSV4_EXPERT_ROUTE_AUTO, NULL, NULL);
}

static int overlap_fill(void *ctx)
{
    Case *c = (Case *)ctx;
    if (c->fill_q >= c->fill_nq) return 0;
    const int q = c->fill_q++;
    const int count = 1 + (q % 7);
    const int ei = (c->cur + 1 + q) % c->ncopies;
    dsv4_prefill_expert_bucket_fill(c->p, &c->ex[ei].ew, 0, count,
                                    c->tok, c->ks);
    return 1;
}

static void run_gpu_fill(void *vp)
{
    Case *c = (Case *)vp;
    const int base = c->cur++;
    DSV4ExpertW *ew = &c->ex[base % c->ncopies].ew;
    c->fill_q = 0;
    gather(c);
    dsv4_prefill_vk_expert_fused(c->p->vk, c->p->ex_out, c->p->ex_x,
                                 &ew->w1, &ew->w3, &ew->w2, c->n,
                                 c->p->c->moe_inter, c->p->c->swiglu_limit,
                                 overlap_fill, c, NULL);
    c->fill_total += c->fill_q;
    scatter(c);
}

/* ----------------------------------------------------------- measurement -- */

static double measure(Case *c, RunFn fn, int target_ms, int *out_reps,
                      Snap *before, Snap *after, int reset_fill)
{
    c->cur = 0;
    for (int i = 0; i < 5; i++) fn(c);
    if (reset_fill) c->fill_total = 0;
    snapshot(c->p, before);

    const double t0 = bench_now_s();
    int reps = 0;
    double elapsed;
    do {
        fn(c);
        reps++;
        elapsed = bench_now_s() - t0;
    } while ((elapsed * 1000.0 < (double)target_ms || reps < 3)
             && reps < 200000);

    snapshot(c->p, after);
    *out_reps = reps;
    return elapsed;
}

static double case_flops(int n)
{
    return 6.0 * (double)EB_HIDDEN * (double)EB_INTER * (double)n;
}

static double case_bytes(const Case *c, int n)
{
    const int d = c->p->c->hidden, inter = c->p->c->moe_inter;
    /* One expert's packed weights plus the activation rows the bucket moves. */
    return (double)(c->ex[0].q1.wb + c->ex[0].q1.sb
                  + c->ex[0].q3.wb + c->ex[0].q3.sb
                  + c->ex[0].q2.wb + c->ex[0].q2.sb)
         + (double)n * (2.0 * d + 3.0 * inter) * 4.0;
}

static void set_case(Case *c, int n)
{
    c->n = n;
    c->cur = 0;
    for (int j = 0; j < EB_MAXTOK; j++) {
        c->tok[j] = (j * 13) % EB_MAXTOK;  /* scattered token rows */
        c->ks[j] = j % EB_TOPK;
    }
}

int main(int argc, char **argv)
{
    int pin = DSV4_PIN_P;
    int threads = 4;
    int copies = 4;
    int target_ms = 120;
    int use_gpu = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--filter") && i + 1 < argc) {
            g_filter = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pin") && i + 1 < argc) {
            const int p = dsv4_threads_parse_pin(argv[++i]);
            if (p < 0) return 1;
            pin = p;
        } else if (!strcmp(argv[i], "--copies") && i + 1 < argc) {
            copies = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--target-ms") && i + 1 < argc) {
            target_ms = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--no-gpu")) {
            use_gpu = 0;
        } else if (!strcmp(argv[i], "--breakdown")) {
            g_breakdown = 1;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            fprintf(stderr,
                "usage: %s [--filter SUBSTR] [--threads N] "
                "[--pin off|all|p|e] [--copies N] "
                "[--target-ms N] [--no-gpu] [--breakdown]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "bench_expert: unknown argument %s\n", argv[i]);
            return 1;
        }
    }
    if (copies < 1 || copies > 8) copies = 2;
    if (target_ms < 10) target_ms = 10;

    const DSV4ThreadOpts to = { pin, threads, 0, DSV4_PIN_OFF };
    dsv4_threads_apply(&to);

    DSV4Cfg cfg;
    int ratios[1];
    setup_cfg(&cfg, ratios);
    DSV4Prefill *p = make_prefill(&cfg, cfg.max_pos);

    if (!use_gpu && p->vk) {
        dsv4_prefill_vk_free(p->vk);
        p->vk = NULL;
    }

    /* The driver's import alignment can be larger than 4096. Allocate the
     * fixture at exactly that alignment so the host-pointer import succeeds. */
    const size_t align = p->vk ? (size_t)p->vk->import_align : 64u;
    ExpertFix *ex = (ExpertFix *)calloc((size_t)copies, sizeof *ex);
    if (!ex) die("allocation failed");
    for (int i = 0; i < copies; i++) expert_fix_init(&ex[i], align);

    if (p->vk) {
        for (int i = 0; i < copies; i++) {
            if (import_expert(p, &ex[i]) != 0) {
                fprintf(stderr, "bench_expert: host import failed; "
                                "continuing CPU-only\n");
                dsv4_prefill_vk_free(p->vk);
                p->vk = NULL;
                break;
            }
        }
    }

    /* Fill every token row so the gather is not a page-fault benchmark. */
    bench_fill_f32(p->x1, (size_t)EB_MAXTOK * (size_t)EB_HIDDEN, 1.0f);
    for (int t = 0; t < EB_MAXTOK; t++)
        memset(p->scratch[t].expert_acc, 0,
               (size_t)EB_TOPK * (size_t)EB_HIDDEN * sizeof(float));

    printf("DeepSeek-V4 Flash routed-expert bucket microbenchmark\n");
    printf("weights=%d copies, hidden=%d moe_inter=%d, Vulkan=%s\n",
           copies, EB_HIDDEN, EB_INTER,
           p->vk ? p->vk->devname : "unavailable/disabled");
    printf("%-30s %-20s %-5s %10s %9s %9s\n",
           "kernel", "shape", "fmt", "ms/call", "GF/s", "GB/s");

    Row rows[EB_MAXTOK + 1];
    memset(rows, 0, sizeof rows);

    DSV4PrefillVk *vk = p->vk;
    Case c;
    memset(&c, 0, sizeof c);
    c.p = p;
    c.ex = ex;
    c.ncopies = copies;
    c.fill_nq = 64;

    for (int n = 1; n <= 10; n++) {
        set_case(&c, n);
        char shape[32];
        snprintf(shape, sizeof shape, "n=%d", n);

        /** CPU: force the production seam onto the CPU kernels. */
        if (!skip_case("expert_bucket_cpu")) {
            p->vk = NULL;
            Snap a, b;
            int reps = 0;
            const double sec = measure(&c, run_cpu, target_ms, &reps, &a, &b, 0);
            rows[n].cpu_ms = sec / (double)reps * 1e3;
            rows[n].cpu_mm = delta_ms(b.st.expert_mm, a.st.expert_mm, reps);
            rows[n].cpu_sw = delta_ms(b.st.expert_swiglu, a.st.expert_swiglu, reps);
            rows[n].cpu_pack = delta_ms(b.st.expert_pack, a.st.expert_pack, reps);
            bench_report("expert_bucket_cpu", shape, "fp4", sec, reps, 1,
                         case_flops(n), case_bytes(&c, n));
            p->vk = vk;
        }

        if (!vk) continue;

        if (!skip_case("expert_bucket_gpu")) {
            Snap a, b;
            int reps = 0;
            const double sec = measure(&c, run_gpu, target_ms, &reps, &a, &b, 0);
            rows[n].gpu_ms = sec / (double)reps * 1e3;
            const DSV4PrefillVkStats *va = &a.vk[DSV4_VK_KIND_EXPERT];
            const DSV4PrefillVkStats *vb = &b.vk[DSV4_VK_KIND_EXPERT];
            rows[n].vk_desc = delta_ms(vb->desc_s, va->desc_s, reps);
            rows[n].vk_cin = delta_ms(vb->copy_in_s, va->copy_in_s, reps);
            rows[n].vk_rec = delta_ms(vb->record_s, va->record_s, reps);
            rows[n].vk_sub = delta_ms(vb->submit_s, va->submit_s, reps);
            rows[n].vk_wait = delta_ms(vb->wait_s, va->wait_s, reps);
            rows[n].vk_cout = delta_ms(vb->copy_out_s, va->copy_out_s, reps);
            rows[n].vk_total = delta_ms(vb->total_s, va->total_s, reps);
            bench_report("expert_bucket_gpu", shape, "fp4", sec, reps, 1,
                         case_flops(n), case_bytes(&c, n));
        }

        if (!skip_case("expert_bucket_dispatch")) {
            Snap a, b;
            int reps = 0;
            const double sec = measure(&c, run_dispatch, target_ms, &reps,
                                       &a, &b, 0);
            rows[n].disp_ms = sec / (double)reps * 1e3;
            bench_report("expert_bucket_dispatch", shape, "fp4", sec, reps, 1,
                         case_flops(n), case_bytes(&c, n));
        }

        if (!skip_case("expert_bucket_gpu_fill")) {
            Snap a, b;
            int reps = 0;
            const double sec = measure(&c, run_gpu_fill, target_ms, &reps,
                                       &a, &b, 1);
            rows[n].ovl_ms = sec / (double)reps * 1e3;
            rows[n].ovl_absorbed = (double)c.fill_total / (double)reps;
            rows[n].vk_fill = delta_ms(
                b.vk[DSV4_VK_KIND_EXPERT].fill_s,
                a.vk[DSV4_VK_KIND_EXPERT].fill_s, reps);
            bench_report("expert_bucket_gpu_fill", shape, "fp4", sec, reps, 1,
                         case_flops(n), case_bytes(&c, n));
        }
    }

    printf("\ncomparison (ms per bucket; winner is isolated latency)\n");
    printf("%3s %10s %10s %10s %10s %8s %s\n",
           "n", "cpu", "gpu", "dispatch", "gpu+fill", "gpu/cpu", "winner");
    for (int n = 1; n <= 10; n++) {
        const Row *r = &rows[n];
        const char *win = "-";
        if (r->cpu_ms > 0.0 && r->gpu_ms > 0.0)
            win = r->gpu_ms < r->cpu_ms ? "gpu" : "cpu";
        printf("%3d %10.4f", n, r->cpu_ms);
        if (r->gpu_ms > 0.0) printf(" %10.4f", r->gpu_ms);
        else                 printf(" %10s", "-");
        if (r->disp_ms > 0.0) printf(" %10.4f", r->disp_ms);
        else                  printf(" %10s", "-");
        if (r->ovl_ms > 0.0) printf(" %10.4f", r->ovl_ms);
        else                 printf(" %10s", "-");
        if (r->cpu_ms > 0.0 && r->gpu_ms > 0.0)
            printf(" %8.2f", r->cpu_ms / r->gpu_ms);
        else
            printf(" %8s", "-");
        printf(" %s\n", win);
    }

    int first_gpu = -1;
    for (int n = 1; n <= 10; n++)
        if (rows[n].cpu_ms > 0.0 && rows[n].gpu_ms > 0.0
            && rows[n].gpu_ms < rows[n].cpu_ms) {
            first_gpu = n;
            break;
        }
    if (first_gpu > 0)
        printf("isolated crossover: GPU first wins at n=%d\n", first_gpu);
    if (vk)
        printf("automatic route: n>=%d uses Vulkan; 5..8 are scheduler "
               "contested (see dispatch rows)\n", DSV4_PF_GPU_ALWAYS_MIN);

    if (g_breakdown) {
        printf("\nbreakdown (ms per bucket)\n");
        printf("%3s | %7s %7s %7s %7s | %7s %7s %7s %7s %7s %7s %7s | %7s %6s\n",
               "n", "cpu_mm", "cpu_sw", "cpu_pack", "cpu_tot",
               "gpu_desc", "gpu_cin", "gpu_rec", "gpu_sub", "gpu_wait",
               "gpu_cout", "gpu_tot", "fill_s", "absorb");
        for (int n = 1; n <= 10; n++) {
            const Row *r = &rows[n];
            printf("%3d | %7.4f %7.4f %7.4f %7.4f | "
                   "%7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f | "
                   "%7.4f %6.1f\n",
                   n, r->cpu_mm, r->cpu_sw, r->cpu_pack, r->cpu_ms,
                   r->vk_desc, r->vk_cin, r->vk_rec, r->vk_sub, r->vk_wait,
                   r->vk_cout, r->vk_total, r->vk_fill, r->ovl_absorbed);
        }
    }

    for (int i = 0; i < copies; i++) expert_fix_free(&ex[i]);
    free(ex);
    dsv4_prefill_free(p);
    return 0;
}

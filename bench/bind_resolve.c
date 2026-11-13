/* SPDX-License-Identifier: Apache-2.0 */
/* bind_resolve.c - what name resolution costs on the per-token bind path.
 *
 * The trunk used to resolve each plan request by scanning one layer's ~46
 * manifest entries with strcmp, 43 times per token. The fast path now indexes
 * a map built once at open. This benchmark compares the two on the real
 * manifest so the change has a number attached to it, independent of the noisy
 * full-model wall clock.
 *
 * It does no reads and no widening: only the lookup loop is timed.
 *
 *   ./bin/bind_resolve [model_dir] [trunk_dir]
 *
 * Defaults are the same ~/colibri-models paths as bench_decode.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4_cfg.h"
#include "dsv4_trunk.h"

#define MAXN 96
#define REPS 20000

typedef struct { char name[MAXN][224]; int n; } NameList;

static int emit_name(void *ctx, int i, const char *name)
{
    NameList *nl = (NameList *)ctx;
    if (i >= MAXN) return -1;
    snprintf(nl->name[i], sizeof nl->name[i], "%s", name);
    nl->n = i + 1;
    return 0;
}

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static volatile long sink;

int main(int argc, char **argv)
{
    const char *home = getenv("HOME");
    char model[1024], trunk[1024];
    if (argc > 1) snprintf(model, sizeof model, "%s", argv[1]);
    else snprintf(model, sizeof model, "%s/colibri-models/deepseek-v4-flash", home);
    if (argc > 2) snprintf(trunk, sizeof trunk, "%s", argv[2]);
    else snprintf(trunk, sizeof trunk, "%s/colibri-models/dsv4-trunk-flash", home);

    char cfg[1200];
    snprintf(cfg, sizeof cfg, "%s/config.json", model);
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, cfg)) return 1;

    DSV4Trunk tr;
    if (dsv4_trunk_open(&tr, trunk, &c, 8LL << 30) != 0) return 1;

    printf("bind name resolution: %d layers, linear scan vs precomputed map\n",
           tr.n_layers);
    printf("%-6s %-6s %10s %10s %10s %12s\n",
           "layer", "tensors", "linear ns", "map ns", "saved ns", "map/token ms");

    double total_lin = 0.0, total_map = 0.0;
    for (int L = 0; L < tr.n_layers; L++) {
        DSV4TrunkLayer *lay = &tr.lay[L];
        if (lay->nt == 0) continue;

        NameList nl = { .n = 0 };
        const int n = dsv4_bind_layer_plan(&c, L, emit_name, &nl);
        if (n < 0 || n != nl.n) {
            fprintf(stderr, "layer %d: plan walk failed\n", L);
            return 1;
        }

        for (int i = 0; i < n; i++) {
            const int ti = lay->bind_of[i];
            if (ti < 0 || ti >= lay->nt || strcmp(lay->t[ti].name, nl.name[i])) {
                fprintf(stderr, "layer %d request %d maps wrong\n", L, i);
                return 1;
            }
        }

        const double t0 = now_ns();
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < n; i++) {
                int found = -1;
                for (int j = 0; j < lay->nt; j++)
                    if (!strcmp(lay->t[j].name, nl.name[i])) { found = j; break; }
                sink += found;
            }
        const double t1 = now_ns();
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < n; i++)
                sink += lay->bind_of[i];
        const double t2 = now_ns();

        const double lin = (t1 - t0) / (double)REPS;
        const double map = (t2 - t1) / (double)REPS;
        total_lin += lin;
        total_map += map;
        if (L < 6 || lin - map > 100.0)
            printf("%-6d %-6d %10.0f %10.0f %10.0f %12.3f\n",
                   L, n, lin, map, lin - map,
                   (lin - map) * (double)tr.n_layers / 1e6);
    }

    printf("\nper token over all %d layers: linear %.1f us, map %.1f us, "
           "saved %.1f us (%.3f ms)\n",
           tr.n_layers, total_lin / 1e3, total_map / 1e3,
           (total_lin - total_map) / 1e3,
           (total_lin - total_map) / 1e6);

    dsv4_trunk_close(&tr);
    printf("(sink %ld)\n", sink);
    return 0;
}

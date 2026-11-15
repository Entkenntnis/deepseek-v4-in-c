/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prof.c - see dsv4_prof.h. */
#include <stdio.h>
#include <time.h>

#include "dsv4_prof.h"

DSV4Prof dsv4_prof;

double dsv4_now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

void dsv4_prof_report(double wall, int passes)
{
    const DSV4Prof *p = &dsv4_prof;
    const double acc = p->hc + p->attn + p->gate + p->expert_io
                     + p->expert_mm + p->shared + p->head
                     + p->trunk_io + p->bind;
    printf("where the time went: %.1f s of %.1f s wall (%.0f%% accounted)\n",
           acc, wall, 100.0 * acc / wall);
    const struct { const char *n; double v; const char *note; } row[] = {
        { "routed experts (matmul)", p->expert_mm, "FP4, CPU-bound"     },
        { "routed experts (disk)",   p->expert_io, "NVMe, cannot move"  },
        { "attention",               p->attn,      "CSA/HCA/RoPE"       },
        { "  of which indexer",      p->idxer,     "score+top-k (subset)"},
        { "mHC + norms",             p->hc,        "Sinkhorn"           },
        { "shared expert",           p->shared,    "FP8"                },
        { "lm head + final norm",    p->head,      "BF16, per token"    },
        { "trunk I/O (read)",        p->trunk_io,  "NVMe, on layer miss" },
        { "trunk bind / widen",      p->bind,      "name resolve + widen"},
        { "router",                  p->gate,      ""                   },
    };
    for (unsigned i = 0; i < sizeof row / sizeof row[0]; i++)
        printf("  %-24s %7.1f s  %5.1f%%  %6.1f ms/pass  %s\n",
               row[i].n, row[i].v, 100.0 * row[i].v / wall,
               1e3 * row[i].v / (passes ? passes : 1), row[i].note);
}

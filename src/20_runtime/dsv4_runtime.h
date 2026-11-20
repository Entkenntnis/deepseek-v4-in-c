/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_runtime.h - runtime-owned progress and cancellation bridge.
 *
 * Prefill reports progress to this layer instead of calling back into the
 * CLI. The CLI installs an optional sink once during engine setup; without a
 * sink, progress reporting is a no-op. This keeps the static dependency
 * direction downward (prefill -> runtime) while still letting the CLI render
 * progress events.
 */
#ifndef DSV4_RUNTIME_H
#define DSV4_RUNTIME_H

typedef struct {
    const char *stage;      /* start | layer | done | head */
    int layer, layers;      /* current layer (1-based) / total layers */
    int tokens_done;        /* estimated from completed layers */
    int tokens_total;       /* tokens in this block */
    double elapsed;         /* seconds since this operation began */
    double progress;        /* 0..1, estimated */
} DSV4Progress;

typedef void (*DSV4ProgressFn)(void *ctx, const DSV4Progress *p);

/* Installed by the composition root (the CLI). Passing NULL disables it. */
void dsv4_runtime_set_progress_sink(DSV4ProgressFn fn, void *ctx);

/* Called by the compute/prefill path. No-op when no sink is installed. */
void dsv4_runtime_report_progress(const DSV4Progress *p);

/* SIGUSR1 requests cancellation; workers poll the flag at safe boundaries. */
void dsv4_runtime_request_cancel(void);
void dsv4_runtime_clear_cancel(void);
int  dsv4_runtime_cancelled(void);

#endif /* DSV4_RUNTIME_H */

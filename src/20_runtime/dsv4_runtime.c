/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_runtime.c - progress sink and cancellation flag. */
#include <signal.h>

#include "dsv4_runtime.h"

static DSV4ProgressFn progress_sink;
static void *progress_ctx;
static volatile sig_atomic_t cancel_requested;

void dsv4_runtime_set_progress_sink(DSV4ProgressFn fn, void *ctx)
{
    progress_sink = fn;
    progress_ctx = ctx;
}

void dsv4_runtime_report_progress(const DSV4Progress *p)
{
    if (progress_sink) progress_sink(progress_ctx, p);
}

void dsv4_runtime_request_cancel(void)
{
    cancel_requested = 1;
}

void dsv4_runtime_clear_cancel(void)
{
    cancel_requested = 0;
}

int dsv4_runtime_cancelled(void)
{
    return cancel_requested != 0;
}

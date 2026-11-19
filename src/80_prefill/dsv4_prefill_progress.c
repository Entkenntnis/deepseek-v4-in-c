/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill_progress.c - the only prefill reporting that remains:
 * coarse progress for the TUI/e2e/prefill default views. */
#include "dsv4_prefill.h"
#include "dsv4_prefill_int.h"

void dsv4_prefill_progress(DSV4Prefill *p, const char *stage, int layer,
                           int tokens_done, int tokens_total,
                           double elapsed, double progress)
{
    if (!p->opt.on_progress) return;
    DSV4PrefillProgress pr;
    pr.stage = stage;
    pr.layer = layer;
    pr.layers = p->c->n_layers;
    pr.tokens_done = tokens_done;
    pr.tokens_total = tokens_total;
    pr.elapsed = elapsed;
    pr.progress = progress < 0.0 ? 0.0 : (progress > 1.0 ? 1.0 : progress);
    p->opt.on_progress(p->opt.progress_ctx, &pr);
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prefill.h - batched prompt prefill with a Vulkan FP8 backend.
 *
 * The decode path is unchanged and remains the bit-exact reference. This path
 * exists only for the prompt: it runs up to DSV4_PV_NTOK tokens through each
 * layer together so a weight matrix is read once and used by the whole batch,
 * and it uses Vulkan for the dense FP8 projections when --gpu-prefill is set.
 *
 * The routed experts are still streamed through the existing cache, but the
 * batch asks for the UNION of its top-k experts once per layer instead of once
 * per token. The expert math itself stays on the CPU in this first cut; the
 * GPU accelerates the dense trunk and shared expert.
 */
#ifndef DSV4_PREFILL_H
#define DSV4_PREFILL_H

#include <stdint.h>

#include "dsv4.h"
#include "dsv4_bind.h"
#include "dsv4_cache.h"
#include "dsv4_layer.h"
#include "dsv4_trunk.h"

typedef struct {
    int gpu;        /* use the Vulkan FP8 backend */
    int batch;      /* tokens per chunk, clamped to DSV4_PV_NTOK */
    int max_pos;    /* same bound the decode scratch/state were built with */
} DSV4PrefillOpts;

typedef struct DSV4Prefill DSV4Prefill;

int  dsv4_prefill_init(DSV4Prefill **out, const DSV4Cfg *c, DSV4Trunk *tr,
                       DSV4LayerState *lst, const DSV4ExpertSrc *src,
                       const DSV4ModelW *mw, const DSV4PrefillOpts *opt);
void dsv4_prefill_free(DSV4Prefill *p);

/* Run prompt tokens ids[0..n-1] through every layer, in chunks. On return
 * `h_out` is the last token's residual stream, ready for the output head. */
int dsv4_prefill_prompt(DSV4Prefill *p, const int32_t *ids, int n,
                        const float *cs_d, const float *sn_d,
                        const float *cs_c, const float *sn_c,
                        float *h_out);

#endif /* DSV4_PREFILL_H */

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_decode.c - the greedy decode loop after a turn has been prefilled. */
#include <math.h>
#include <string.h>

#include "dsv4_kernels.h"
#include "dsv4_matmul.h"
#include "dsv4_serve_int.h"
#include "dsv4_util.h"

void dsv4_decode_loop(Engine *e, int n_added, int start_pos, int snap_ok,
                      int ngen, int eos, double t0, double tp0,
                      int32_t best)
{
    int generated = 0;
    double ttft = 0.0;
    const char *stop = "max";

    while (generated < ngen) {
        if (dsv4_runtime_cancelled()) {
            /* Before the first token there is nothing to keep from this turn,
             * so a stop rolls the context back rather than leaving a prompt
             * with no answer in it. */
            if (generated == 0) {
                dsv4_cancel_prefill_turn(e, snap_ok, start_pos, n_added, t0, tp0);
                return;
            }
            stop = "cancel";
            break;
        }
        if (e->n_ids >= e->cap_ids) {
            dsv4_emit_error("context full (%d tokens) while generating",
                            e->cap_ids);
            stop = "context";
            break;
        }
        const int32_t id = best;
        const int pos = e->n_ids;
        e->ids[e->n_ids++] = id;

        const double el = dsv4_now_s() - t0;
        if (generated == 0) ttft = el;
        dsv4_emit_token(e, id, generated, el, ttft);
        generated++;
        /* The prefix-only context has now become a real completed-prefix turn;
         * the next generate may need the normal EOS handling again. */
        e->kv_prefix = 0;

        if (id == eos) {
            stop = "eos";
            e->pending_forward = 1;   /* forwarded at the start of next turn */
            break;
        }
        if (generated >= ngen) {
            stop = "max";
            e->pending_forward = 1;
            break;
        }

        const int frc = dsv4_forward(e, id, pos, 1, "gen",
                                     dsv4_now_s() - t0, &best);
        if (frc != 0) {
            e->pending_forward = 1;   /* the emitted token is not in the KV yet */
            stop = "error";
            break;
        }
        e->pending_forward = 0;
    }

    const double dt = dsv4_now_s() - t0;
    /* A zero-token turn that still committed its user message (ngen=0) has
     * moved past the bare prefix, so the next turn must close it normally. */
    if (generated == 0 && strcmp(stop, "cancel") != 0 &&
        strcmp(stop, "error") != 0)
        e->kv_prefix = 0;
    dsv4_emit_done(e, n_added, generated, dt, ttft, stop);
}

void dsv4_cancel_prefill_turn(Engine *e, int rollback, int start_pos,
                              int prompt_tokens, double t0, double tp0)
{
    e->prefill_s = dsv4_now_s() - tp0;
    e->prefill_tokens = 0;   /* nothing from this turn is in the context */
    if (rollback) {
        dsv4_state_snapshot_restore(e->snap);
        e->n_ids = start_pos;
        e->pending_forward = 0;
    } else {
        if (dsv4_engine_reset_context(e) != 0) {
            dsv4_emit_error("prefill cancelled; context reset failed");
            dsv4_runtime_clear_cancel();
            return;
        }
        dsv4_emit_reset(e, "cancel");
    }
    dsv4_emit_done(e, prompt_tokens, 0, dsv4_now_s() - t0, 0.0, "cancel");
    dsv4_runtime_clear_cancel();
}

/* The head's own hc reduction. Small model math, kept here because this is the
 * only caller; if a second one appears, it moves into the decode layer. */
static void hc_head(float *y, const float *x, const DSV4HcW *w,
                    int hc, int hidden, float norm_eps, float hc_eps)
{
    const int hcdim = hc * hidden;
    double sq = 0.0;
    for (int i = 0; i < hcdim; i++) sq += (double)x[i] * x[i];
    const float rs = (float)(1.0 / sqrt(sq / (double)hcdim + (double)norm_eps));

    float pre[DSV4_MAX_HC_MULT];
    for (int j = 0; j < hc; j++) {
        const float *row = w->fn + (size_t)j * hcdim;
        double acc = 0.0;
        for (int i = 0; i < hcdim; i++)
            acc = fma((double)row[i], (double)x[i], acc);
        const float m = (float)acc * rs;
        pre[j] = 1.0f / (1.0f + expf(-(m * w->scale[0] + w->base[j]))) + hc_eps;
    }
    for (int d = 0; d < hidden; d++) {
        double acc = 0.0;
        for (int j = 0; j < hc; j++)
            acc += (double)pre[j] * (double)x[(size_t)j * hidden + d];
        y[d] = (float)acc;
    }
}

/* Greedy sample: only the argmax is needed by any default view. */
static int argmax(const float *lg, int n)
{
    int best = 0;
    for (int v = 1; v < n; v++) if (lg[v] > lg[best]) best = v;
    return best;
}

/* Run the output head on whatever residual stream is in e->h. Shared by
 * decode and the batched prefill: the inputs are identical, and two copies
 * would be one more place for them to drift. */
void dsv4_head_from_h(Engine *e, int32_t *best)
{
    const int hc = e->c.hc_mult, d = e->c.hidden;
    const double th = dsv4_now_s();
    hc_head(e->xh, e->h, &e->mb.w.hc_head, hc, d, e->c.rms_eps, e->c.hc_eps);
    dsv4_rmsnorm(e->xh, e->xh, e->mb.w.norm, d, e->c.rms_eps);
    DSV4QMat head = { e->mb.w.head, NULL, DSV4_WBF16, e->c.vocab, d, 0, 0 };
    dsv4_mmq(e->logits, e->xh, &head);
    e->head_s += dsv4_now_s() - th;
    if (best) *best = argmax(e->logits, e->c.vocab);
}

int dsv4_forward(Engine *e, int32_t tid, int pos, int want_logits,
                 const char *phase, double t_turn, int32_t *best)
{
    const int hc = e->c.hc_mult, d = e->c.hidden;

    if (e->cache.pf_enabled)
        dsv4_cache_prefetch_hash_layers(&e->cache, tid);

    dsv4_embed_row(e->xh, e->mb.w.embed, e->mb.w.wdt, tid, d);
    for (int j = 0; j < hc; j++)
        memcpy(e->h + (size_t)j * d, e->xh, (size_t)d * sizeof(float));

    for (int L = 0; L < e->c.n_layers; L++) {
        DSV4LayerW lw;
        if (dsv4_trunk_bind(&e->tr, &e->c, L, &lw) != 0) {
            dsv4_emit_error("trunk bind failed on layer %d at position %d",
                            L, pos);
            return -1;
        }
        const int comp = dsv4_compress_ratio(&e->c, L) != 0;
        dsv4_layer_forward(e->h, &lw, &e->c, &e->scratch, &e->cache,
                           &e->lst[L], comp ? e->cs_c : e->cs_d,
                           comp ? e->sn_c : e->sn_d, pos, tid);
        dsv4_emit_layer(e, phase, L);
    }

    if (want_logits) dsv4_head_from_h(e, best);
    return 0;
}

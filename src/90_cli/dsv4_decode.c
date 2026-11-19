/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_decode.c - the greedy decode loop after a turn has been prefilled. */
#include <string.h>

#include "dsv4_serve_int.h"
#include "dsv4_time.h"

void dsv4_decode_loop(Engine *e, int n_added, int start_pos, int snap_ok,
                      int ngen, int eos, int emit_layers, double t0,
                      double tp0, int32_t best)
{
    int generated = 0;
    double ttft = 0.0;
    const char *stop = "max";

    while (generated < ngen) {
        if (dsv4_cancel) {
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

        const int frc = dsv4_forward(e, id, pos, 1, emit_layers, "gen",
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

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_abort.c - roll back or reset a cancelled prefill turn. */
#include "dsv4_serve_int.h"
#include "dsv4_time.h"

/* Stop a turn before its first token. With a snapshot the layer state and the
 * token array roll back to the end of the previous turn, so the conversation
 * survives and only this new message is dropped. Without one, the partial
 * prefill cannot be resumed and the whole context has to go. */
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
            dsv4_cancel = 0;
            return;
        }
        dsv4_emit_reset(e, "cancel");
    }
    dsv4_emit_done(e, prompt_tokens, 0, dsv4_now_s() - t0, 0.0, "cancel");
    dsv4_cancel = 0;
}


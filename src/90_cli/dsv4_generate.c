/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_generate.c - one generate turn: close, prefill, head, decode.
 *
 * Turn state:
 *   n_ids            committed token IDs plus the prompt just appended
 *   pending_forward  ids[n_ids - 1] has no KV entry yet
 *   kv_prefix        context is a bare prefix, not a finished assistant turn
 *   snap             rollback point for a prefill that writes KV
 *
 * Flow:
 *   close previous turn -> append prompt -> reserve block -> snapshot ->
 *   block prefill -> output head -> decode
 */
#include <stdio.h>

#include "dsv4_serve_int.h"
#include "dsv4_util.h"

/* ---- shared block-prefill wrappers ------------------------------------- */

/* Reserve the prompt block before any KV is written. On failure the caller
 * must roll back the token IDs it has already appended. */
int dsv4_prefill_prepare(Engine *e, int n)
{
    if (!e->pf) {
        fprintf(stderr, "serve: block prefill is unavailable\n");
        return -1;
    }
    if (dsv4_prefill_reserve(e->pf, n) != 0) {
        fprintf(stderr, "serve: cannot allocate the %d-token prefill block\n", n);
        dsv4_prefill_release(e->pf);
        return -1;
    }
    return 0;
}

int dsv4_prefill_run(Engine *e, const int32_t *ids, int n, int pos0)
{
    const int rc = dsv4_prefill_range(e->pf, ids, n, pos0,
                                      e->cs_d, e->sn_d, e->cs_c, e->sn_c,
                                      e->h);
    /* These buffers are per call. Release them now so the longest prompt ever
     * seen cannot become a permanent high-water mark. */
    dsv4_prefill_release(e->pf);
    return rc;
}

/* ---- turn helpers ------------------------------------------------------- */

/* A generate call starts after the previous assistant turn. This loop runs at
 * most twice: first it commits a token left pending by decode, then it closes
 * an unfinished assistant turn with EOS. A loaded prefix is not an assistant
 * turn, so it must not receive an EOS. */
static int close_unfinished_turn(Engine *e, int eos, double turn_start)
{
    for (;;) {
        if (e->pending_forward) {
            const int pos = e->n_ids - 1;
            if (dsv4_forward(e, e->ids[pos], pos, 0, "resume",
                             dsv4_now_s() - turn_start, NULL) != 0)
                return -1;
            e->pending_forward = 0;
            continue;
        }
        if (e->kv_prefix || e->n_ids == 0 ||
            e->ids[e->n_ids - 1] == eos)
            return 0;
        if (e->n_ids >= e->cap_ids) {
            dsv4_emit_error("context full (%d tokens); reset before continuing",
                            e->cap_ids);
            return -1;
        }
        e->ids[e->n_ids++] = eos;
        e->pending_forward = 1;
    }
}

/* Run the block prefill, then the output head. On success, `next_id` is the
 * first generated token and the decode loop can start. Cancellation and
 * errors are reported here; the caller only has to stop. */
static int prefill_turn(Engine *e, int prompt_pos, int n_prompt, int ngen,
                        double turn_start, int *snap_ok,
                        double *prefill_start, int32_t *next_id)
{
    DSV4CacheGuard guard;
    const DSV4PrefillBytes pb = dsv4_prefill_bytes(&e->c, n_prompt);
    dsv4_cache_guard_begin(&e->cache, pb.block_bytes + pb.token_bytes,
                           dsv4_prefill_guard_slack(e->pf), &guard);

    if (dsv4_prefill_prepare(e, n_prompt) != 0) {
        dsv4_cache_guard_end(&e->cache, &guard);
        e->n_ids = prompt_pos;
        dsv4_emit_error("cannot allocate the %d-token prefill block", n_prompt);
        return -1;
    }
    dsv4_emit_start(e, prompt_pos, n_prompt, ngen, n_prompt);

    /* Prefill writes KV and compressor state, so save a rollback point first.
     * The snapshot is model-sized, not prompt-sized. */
    *snap_ok = e->snap && dsv4_state_snapshot_save(e->snap) == 0;

    e->prefill_s = 0.0;
    e->prefill_tokens = 0;
    *prefill_start = dsv4_now_s();
    const int rc = dsv4_prefill_run(e, e->ids + prompt_pos, n_prompt, prompt_pos);
    /* run() released the block buffers, so the cache can reclaim its window
     * before decode starts refilling it. */
    dsv4_cache_guard_end(&e->cache, &guard);

    if (rc == DSV4_PREFILL_CANCELLED || rc == DSV4_PREFILL_ABORTED) {
        dsv4_cancel_prefill_turn(e, *snap_ok, prompt_pos, n_prompt,
                                 turn_start, *prefill_start);
        return -1;
    }
    if (rc != 0) {
        dsv4_emit_error("block prefill failed at position %d; reset the "
                        "context before continuing", prompt_pos);
        return -1;
    }

    e->prefill_s = dsv4_now_s() - *prefill_start;
    e->prefill_tokens = n_prompt;

    /* No token has been emitted yet, so a stop here still rolls back the turn. */
    if (dsv4_runtime_cancelled()) {
        dsv4_cancel_prefill_turn(e, *snap_ok, prompt_pos, n_prompt,
                                 turn_start, *prefill_start);
        return -1;
    }

    /* The prefill bar is done, but the first token is not ready until the head
     * has run. Tell the client which phase is active. */
    const DSV4Progress hp = {
        .stage = "head",
        .layer = e->c.n_layers,
        .layers = e->c.n_layers,
        .tokens_done = n_prompt,
        .tokens_total = n_prompt,
        .elapsed = e->prefill_s,
        .progress = 1.0,
    };
    dsv4_emit_prefill_progress(e, &hp);

    dsv4_head_from_h(e, next_id);
    if (dsv4_runtime_cancelled()) {
        dsv4_cancel_prefill_turn(e, *snap_ok, prompt_pos, n_prompt,
                                 turn_start, *prefill_start);
        return -1;
    }
    return 0;
}

/* ---- the turn ----------------------------------------------------------- */

void dsv4_engine_generate(Engine *e, const int32_t *tokens, int n_tokens,
                          int ngen, int eos)
{
    const double turn_start = dsv4_now_s();
    dsv4_runtime_clear_cancel();
    e->head_s = 0.0;

    if (ngen < 0) ngen = 0;
    if (n_tokens <= 0) {
        dsv4_emit_error("generate needs a non-empty token list");
        return;
    }
    if (eos < 0 || eos >= e->c.vocab) {
        dsv4_emit_error("generate needs a valid end-of-turn token id");
        return;
    }

    if (close_unfinished_turn(e, eos, turn_start) != 0) return;

    const int prompt_pos = e->n_ids;
    const int n_prompt = dsv4_append_ids(e, tokens, n_tokens);
    if (n_prompt < 0) {
        if (n_prompt == -2)
            dsv4_emit_error("context full (%d tokens): this turn needs more "
                            "room. Use /reset or raise --ctx.", e->cap_ids);
        else
            dsv4_emit_error("invalid token id in prompt");
        return;
    }

    int snap_ok = 0;
    double prefill_start = 0.0;
    int32_t next_id = 0;
    if (prefill_turn(e, prompt_pos, n_prompt, ngen, turn_start,
                     &snap_ok, &prefill_start, &next_id) != 0)
        return;

    dsv4_decode_loop(e, n_prompt, prompt_pos, snap_ok, ngen, eos,
                     turn_start, prefill_start, next_id);
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_generate.c - one turn: prompt prefill, then decode until stop. */
#include <stdio.h>

#include "dsv4_serve_int.h"
#include "dsv4_util.h"

int dsv4_prefill_prepare(Engine *e, int n, int *batched)
{
    *batched = 0;
    if (!e->pf || n <= 1) return 0;
    if (dsv4_prefill_reserve(e->pf, n) != 0) {
        fprintf(stderr, "serve: cannot allocate a %d-token prefill block; "
                        "falling back to the per-token path\n", n);
        dsv4_prefill_release(e->pf);
        return 0;
    }
    *batched = 1;
    return 0;
}

int dsv4_prefill_run(Engine *e, const int32_t *ids, int n, int pos0,
                     int batched, const char *phase, double t0)
{
    if (batched) {
        const int rc = dsv4_prefill_range(e->pf, ids, n, pos0,
                                          e->cs_d, e->sn_d, e->cs_c, e->sn_c,
                                          e->h);
        /* This turn is the only user of the block buffers. Drop them now
         * instead of turning the longest prompt ever seen into a permanent
         * high-water mark for the resident engine. */
        dsv4_prefill_release(e->pf);
        return rc;
    }

    for (int i = 0; i < n; i++) {
        if (dsv4_runtime_cancelled()) return DSV4_PREFILL_CANCELLED;
        if (dsv4_forward(e, ids[i], pos0 + i, 0, phase,
                         dsv4_now_s() - t0, NULL) != 0)
            return -1;
    }
    return 0;
}

void dsv4_engine_generate(Engine *e, const int32_t *tokens, int n_tokens,
                          int ngen, int eos)
{
    const double t0 = dsv4_now_s();
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

    /* A restored prefix already contains BOS + system prompt, but it is not a
     * finished assistant turn: do not append EOS in front of the first user
     * message. */
    const int prefix_start = e->kv_prefix;

    /* Close a truncated previous turn, then forward the token that the previous
     * turn deliberately left pending; its KV entry is not in the layer state yet. */
    if (!prefix_start && e->n_ids > 0 && e->ids[e->n_ids - 1] != eos) {
        if (e->n_ids >= e->cap_ids) {
            dsv4_emit_error("context full (%d tokens); reset before continuing",
                            e->cap_ids);
            return;
        }
        e->ids[e->n_ids++] = eos;
        e->pending_forward = 1;
    }
    if (e->pending_forward) {
        const int pos = e->n_ids - 1;
        if (dsv4_forward(e, e->ids[pos], pos, 0, "resume",
                         dsv4_now_s() - t0, NULL) != 0) return;
        e->pending_forward = 0;
    }

    const int start_pos = e->n_ids;
    const int n_added = dsv4_append_ids(e, tokens, n_tokens);
    if (n_added < 0) {
        if (n_added == -2)
            dsv4_emit_error("context full (%d tokens): this turn needs more "
                            "room. Use /reset or raise --ctx.", e->cap_ids);
        else
            dsv4_emit_error("invalid token id in prompt");
        return;
    }

    /* The whole new turn is one prefill block. Its buffers scale with the
     * prompt and sit next to an already-resident expert cache; make room in
     * the cache before reserving them, and keep claims capped until the block
     * is released. */
    DSV4CacheGuard guard;
    const DSV4PrefillBytes pb = dsv4_prefill_bytes(&e->c, n_added);
    dsv4_cache_guard_begin(&e->cache, pb.block_bytes + pb.token_bytes,
                           dsv4_prefill_guard_slack(e->pf), &guard);

    int use_batched = 0;
    dsv4_prefill_prepare(e, n_added, &use_batched);
    if (!use_batched)
        dsv4_cache_guard_end(&e->cache, &guard);
    dsv4_emit_start(e, start_pos, n_added, ngen,
                    use_batched ? n_added : 0);

    /* Save the old context before the first layer can touch it. The snapshot is
     * model-sized, not prompt-sized, and is overwritten every turn. */
    const int snap_ok =
        e->snap && dsv4_state_snapshot_save(e->snap) == 0;

    /* Prefill the new turn. The head runs only on the final prompt token. Once
     * the block path starts it cannot fall back: it has already written KV and
     * compressor state, so an error is terminal for the turn. */
    e->prefill_s = 0.0;
    e->prefill_tokens = 0;
    const double tp0 = dsv4_now_s();
    const int prc = dsv4_prefill_run(e, e->ids + start_pos, n_added, start_pos,
                                     use_batched, "prompt", t0);
    /* run() released the block buffers, so the full cache window can return
     * before the decode loop starts refilling it. */
    if (use_batched)
        dsv4_cache_guard_end(&e->cache, &guard);
    if (prc == DSV4_PREFILL_CANCELLED || prc == DSV4_PREFILL_ABORTED) {
        dsv4_cancel_prefill_turn(e, snap_ok, start_pos, n_added, t0, tp0);
        return;
    }
    if (prc != 0) {
        if (use_batched)
            dsv4_emit_error("block prefill failed at position %d; reset the "
                            "context before continuing", start_pos);
        return;
    }
    e->prefill_s = dsv4_now_s() - tp0;
    e->prefill_tokens = n_added;
    /* No token has been emitted yet, so a stop here still drops this turn and
     * restores the context from before it. */
    if (dsv4_runtime_cancelled()) {
        dsv4_cancel_prefill_turn(e, snap_ok, start_pos, n_added, t0, tp0);
        return;
    }
    if (use_batched) {
        const DSV4Progress hp = {
            .stage = "head",
            .layer = e->c.n_layers,
            .layers = e->c.n_layers,
            .tokens_done = n_added,
            .tokens_total = n_added,
            .elapsed = e->prefill_s,
            .progress = 1.0,
        };
        dsv4_emit_prefill_progress(e, &hp);
    }
    int32_t best = 0;
    dsv4_head_from_h(e, &best);
    if (dsv4_runtime_cancelled()) {
        dsv4_cancel_prefill_turn(e, snap_ok, start_pos, n_added, t0, tp0);
        return;
    }

    dsv4_decode_loop(e, n_added, start_pos, snap_ok, ngen, eos,
                     t0, tp0, best);
}

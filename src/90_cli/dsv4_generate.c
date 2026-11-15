/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_generate.c - one turn: prompt prefill, then decode until stop. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_serve_int.h"

void dsv4_engine_generate(Engine *e, const char *prompt, int ngen, int think,
                          int emit_layers)
{
    const double t0 = dsv4_now_s();
    dsv4_cancel = 0;
    e->prof_base = dsv4_prof;

    if (ngen < 0) ngen = 0;
    if (e->eos_id < 0) {
        dsv4_emit_error("tokenizer has no end-of-turn token; refusing to run");
        return;
    }

    /* Close a truncated previous turn, then forward the token that the previous
     * turn deliberately left pending; its KV entry is not in the layer state yet. */
    if (e->n_ids > 0 && e->ids[e->n_ids - 1] != e->eos_id) {
        if (e->n_ids >= e->cap_ids) {
            dsv4_emit_error("context full (%d tokens); reset before continuing",
                            e->cap_ids);
            return;
        }
        e->ids[e->n_ids++] = e->eos_id;
        e->pending_forward = 1;
    }
    if (e->pending_forward) {
        const int pos = e->n_ids - 1;
        TopK dummy;
        if (dsv4_forward(e, e->ids[pos], pos, 0, emit_layers, "resume",
                         dsv4_now_s() - t0, &dummy) != 0) return;
        e->pending_forward = 0;
    }

    const size_t sys_len = e->system_prompt ? strlen(e->system_prompt) : 0;
    const size_t need = strlen(prompt) + sys_len + 256;
    char *turn = malloc(need);
    if (!turn) { dsv4_emit_error("out of memory formatting turn"); return; }

    if (e->n_ids == 0) {
        snprintf(turn, need, "%s%s%s%s%s%s%s",
                 DSV4_BOS,
                 e->system_prompt ? e->system_prompt : "",
                 e->system_prompt ? "\n" : "",
                 DSV4_USER, prompt, DSV4_ASSISTANT,
                 think ? DSV4_THINK_OPEN : DSV4_THINK_CLOSE);
    } else {
        snprintf(turn, need, "%s%s%s%s",
                 DSV4_USER, prompt, DSV4_ASSISTANT,
                 think ? DSV4_THINK_OPEN : DSV4_THINK_CLOSE);
    }

    const int start_pos = e->n_ids;
    const int n_added = dsv4_append_text(e, turn);
    free(turn);
    if (n_added < 0) {
        if (n_added == -2)
            dsv4_emit_error("context full (%d tokens): this turn needs more "
                            "room. Use /reset or raise --ctx.", e->cap_ids);
        else
            dsv4_emit_error("tokenization failed");
        return;
    }
    if (n_added == 0) {
        dsv4_emit_error("user message encoded to zero tokens");
        return;
    }

    /* The whole new turn is one prefill block. Reserve its buffers before any
     * state moves; a failure here can still take the per-token fallback. */
    int use_batched = (e->pf && n_added > 1);
    if (use_batched && dsv4_prefill_reserve(e->pf, n_added) != 0) {
        fprintf(stderr, "serve: cannot allocate a %d-token prefill block; "
                        "falling back to the per-token path for this turn\n",
                n_added);
        use_batched = 0;
    }
    dsv4_emit_start(e, start_pos, n_added, ngen, think ? "thinking" : "chat",
                    use_batched ? n_added : 0);

    TopK top;
    memset(&top, 0, sizeof top);

    /* Prefill the new turn. The head runs only on the final prompt token. Once
     * the block path starts it cannot fall back: it has already written KV and
     * compressor state, so an error is terminal for the turn. */
    e->prefill_s = 0.0;
    e->prefill_tokens = 0;
    const double tp0 = dsv4_now_s();
    if (use_batched) {
        if (dsv4_prefill_range(e->pf, e->ids + start_pos, n_added, start_pos,
                               e->cs_d, e->sn_d, e->cs_c, e->sn_c,
                               e->h) != 0) {
            dsv4_emit_error("block prefill failed at position %d; reset the "
                            "context before continuing", start_pos);
            return;
        }
    } else {
        for (int i = start_pos; i < e->n_ids; i++)
            if (dsv4_forward(e, e->ids[i], i, 0, emit_layers, "prompt",
                             dsv4_now_s() - t0, &top) != 0) return;
    }
    e->prefill_s = dsv4_now_s() - tp0;
    e->prefill_tokens = n_added;
    if (use_batched) {
        const DSV4PrefillProgress hp = {
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
    dsv4_head_from_h(e, &top);

    int generated = 0;
    double ttft = 0.0;
    const char *stop = "max";
    int best = top.n > 0 ? top.id[0] : 0;

    while (generated < ngen) {
        if (dsv4_cancel) {
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
        dsv4_emit_token(e, id, generated, pos, el, ttft, &top);
        generated++;

        if (id == e->eos_id) {
            stop = "eos";
            e->pending_forward = 1;   /* forwarded at the start of next turn */
            break;
        }
        if (generated >= ngen) {
            stop = "max";
            e->pending_forward = 1;
            break;
        }

        if (dsv4_forward(e, id, pos, 1, emit_layers, "gen",
                         dsv4_now_s() - t0, &top) != 0) {
            e->pending_forward = 1;   /* the emitted token is not in the KV yet */
            stop = "error";
            break;
        }
        e->pending_forward = 0;
        best = top.n > 0 ? top.id[0] : 0;
    }

    const double dt = dsv4_now_s() - t0;
    dsv4_emit_done(e, n_added, generated, dt, ttft, stop);
}

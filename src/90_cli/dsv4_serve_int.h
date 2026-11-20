/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_serve_int.h - the resident engine and seams between the CLI files. */
#ifndef DSV4_SERVE_INT_H
#define DSV4_SERVE_INT_H

#include <signal.h>
#include <stdint.h>

#include "dsv4.h"
#include "dsv4_bind.h"
#include "dsv4_cache.h"
#include "dsv4_layer.h"
#include "dsv4_prefill.h"
#include "dsv4_runtime.h"
#include "dsv4_st.h"
#include "dsv4_trunk.h"

typedef struct {
    const char      *model_dir;
    DSV4Cfg          c;

    DSV4St           st;
    DSV4Trunk        tr;
    DSV4Cache        cache;
    DSV4ModelBind    mb;
    DSV4Scratch      scratch;
    DSV4LayerState  *lst;
    DSV4Prefill     *pf;       /* NULL when block prefill could not initialise */
    DSV4StateSnapshot *snap;   /* rollback point for an in-flight prefill */

    float           *cs_d, *sn_d, *cs_c, *sn_c;
    int              maxpos;

    float           *h, *xh, *logits;

    int32_t         *ids;
    int              n_ids, cap_ids;
    int              pending_forward;
    /* A loaded prefix-cache context starts a fresh turn; it is not a previous
     * assistant turn that needs an EOS appended. Cleared once the first token
     * of that turn is emitted, or by reset. */
    int              kv_prefix;

    /* Last turn's prompt timing, so the TUI can print an honest prefill line
     * without subtracting the output head from ttft. */
    double           prefill_s;
    int              prefill_tokens;
    /* Time spent in the output head this turn. The prefill benchmark's default
     * table subtracts it from TTFT to show where the time went. */
    double           head_s;
    /* Last time dynamic stats were appended to an event. Static cache/trunk
     * fields are emitted only on ready; the live counters are throttled. */
    double           last_stats_s;

    int              gen_default;

    /* Scratch buffer for token ids parsed out of one command line. */
    int32_t         *in_ids;
    int              in_cap;

} Engine;

/* SIGUSR1 requests cancellation through the runtime layer. Prefill polls it
 * between layers and sequential-stage tokens; decode stops at the next token
 * boundary. */

/* ---- proc ---- */
double dsv4_proc_rss_mb(void);
double dsv4_proc_read_gb(void);

/* ---- events ---- */
/* One error event for every failure. KV-cache messages start with "KV: " so
 * the client can treat them as recoverable without a separate event type. */
void dsv4_emit_error(const char *fmt, ...);
void dsv4_emit_ready(Engine *e, const char *event);
void dsv4_emit_start(Engine *e, int start_pos, int prompt_tokens,
                     int gen_max, int prefill_tokens);
void dsv4_emit_layer(const Engine *e, const char *phase, int layer);
void dsv4_emit_token(Engine *e, int32_t id, int index,
                     double t_turn, double ttft);
void dsv4_emit_done(Engine *e, int prompt_tokens, int generated,
                    double seconds, double ttft, const char *stop);
void dsv4_emit_reset(const Engine *e, const char *reason);
void dsv4_emit_prefill_progress(Engine *e, const DSV4Progress *p);
void dsv4_emit_kv_built(int tokens, double seconds, long long bytes);
void dsv4_emit_kv_loaded(int tokens, int context_cap, double seconds);

/* ---- commands ---- */
int  dsv4_handle_command(Engine *e, char *line);

/* ---- engine ---- */
/* A negative budget_gb means "compute one from available memory". */
int  dsv4_engine_init(Engine *e, const char *model, const char *trunkdir,
                      double budget_gb, int ctx, int gen_default);
void dsv4_engine_free(Engine *e);
int  dsv4_engine_reset_context(Engine *e);
int  dsv4_append_ids(Engine *e, const int32_t *ids, int n);
int  dsv4_forward(Engine *e, int32_t tid, int pos, int want_logits,
                  int emit_layers, const char *phase, double t_turn,
                  int32_t *best);
void dsv4_head_from_h(Engine *e, int32_t *best);
void dsv4_cancel_prefill_turn(Engine *e, int rollback, int start_pos,
                              int prompt_tokens, double t0, double tp0);
void dsv4_decode_loop(Engine *e, int n_added, int start_pos, int snap_ok,
                      int ngen, int eos, int emit_layers, double t0,
                      double tp0, int32_t best);
void dsv4_engine_generate(Engine *e, const int32_t *tokens, int n_tokens,
                          int ngen, int eos, int emit_layers);

/* Reserve the block path if it is available. Always returns 0; *batched says
 * whether the following dsv4_prefill_run() will use it. */
int  dsv4_prefill_prepare(Engine *e, int n, int *batched);
/* Run the new tokens through either the block path (when batched) or the
 * per-token path. Returns 0, DSV4_PREFILL_CANCELLED/ABORTED, or -1. */
int  dsv4_prefill_run(Engine *e, const int32_t *ids, int n, int pos0,
                      int batched, int emit_layers, const char *phase,
                      double t0);
void dsv4_engine_kv_build(Engine *e, const int32_t *tokens, int n_tokens,
                          const char *path);
void dsv4_engine_kv_load(Engine *e, const int32_t *tokens, int n_tokens,
                         const char *path);

#endif /* DSV4_SERVE_INT_H */

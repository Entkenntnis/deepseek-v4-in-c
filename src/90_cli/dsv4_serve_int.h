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
#include "dsv4_prof.h"
#include "dsv4_st.h"
#include "dsv4_tok.h"
#include "dsv4_trunk.h"

/* json.h is included only where it is used; this keeps its static helpers out
 * of every CLI translation unit. */
struct jval;

typedef struct {
    const char      *model_dir;
    DSV4Cfg          c;
    int              cr[DSV4_MAX_LAYERS];

    DSV4St           st;
    DSV4Tok          tok;
    DSV4Trunk        tr;
    DSV4Cache        cache;
    DSV4ModelBind    mb;
    DSV4Scratch      scratch;
    DSV4LayerState  *lst;
    DSV4ExpertSrc    src;      /* one source, shared by decode and prefill */
    DSV4Prefill     *pf;       /* NULL when block prefill could not initialise */

    float           *cs_d, *sn_d, *cs_c, *sn_c;
    int              maxpos;

    float           *h, *xh, *logits;

    int32_t         *ids;
    int              n_ids, cap_ids;
    int32_t          eos_id;
    int              pending_forward;

    /* Last turn's prompt timing, so the TUI can print an honest prefill line
     * without subtracting the output head from ttft. */
    double           prefill_s;
    int              prefill_tokens;

    char            *system_prompt;
    int              think_default;
    int              gen_default;
    int              emit_layers_default;

    int32_t         *enc_tmp;
    int              enc_cap;
    unsigned char   *tok_bytes;
    int              tok_cap;
    char            *hexbuf;
    int              hex_cap;

    double           t_start;
    DSV4Prof         prof_base;
} Engine;

typedef struct {
    int    id[5];
    float  logit[5];
    double prob[5];
    int    n;
} TopK;

/* SIGUSR1 sets this; the decode loop stops at the next token boundary. */
extern volatile sig_atomic_t dsv4_cancel;

/* ---- util ---- */
double dsv4_proc_cpu_s(void);
void   dsv4_proc_mem_mb(double *rss_mb, double *peak_mb);
double dsv4_proc_read_gb(void);
void   dsv4_json_escape(const char *s);
void   dsv4_usage(const char *me);
void   dsv4_emit_error(const char *fmt, ...);
const char *dsv4_jstr(struct jval *o, const char *key);
double dsv4_jnum(struct jval *o, const char *key, double dflt);
int    dsv4_jbool(struct jval *o, const char *key, int dflt);
void   dsv4_jfree(struct jval *v);

/* ---- stats ---- */
void dsv4_write_sys_full(const Engine *e);
void dsv4_write_cache(const Engine *e);
void dsv4_write_trunk(const Engine *e);
void dsv4_write_prof(const Engine *e);

/* ---- events ---- */
void dsv4_emit_ready(const Engine *e, const char *event);
void dsv4_emit_start(const Engine *e, int start_pos, int prompt_tokens,
                     int gen_max, const char *mode, int prefill_tokens);
void dsv4_emit_layer(const Engine *e, int pos, const char *phase, int layer,
                     double t_turn);
void dsv4_emit_token(const Engine *e, int32_t id, int index, int pos,
                     double t_turn, double ttft, const TopK *top);
void dsv4_emit_done(const Engine *e, int prompt_tokens, int generated,
                    double seconds, double ttft, const char *stop);
void dsv4_emit_prefill_progress(Engine *e, const DSV4PrefillProgress *p);

/* ---- engine ---- */
int  dsv4_engine_init(Engine *e, const char *model, const char *trunkdir,
                      const char *tokfile, double budget_gb, int budget_auto,
                      int ctx, int gen_default, int think_default,
                      int layers_default);
void dsv4_set_stream_miss(int n);
int  dsv4_stream_miss(void);
int  dsv4_make_expert_src(DSV4ExpertSrc *src, DSV4Cache *cache);
void dsv4_engine_free(Engine *e);
int  dsv4_engine_reset_context(Engine *e);
int  dsv4_append_text(Engine *e, const char *text);
int  dsv4_forward(Engine *e, int32_t tid, int pos, int want_logits,
                  int emit_layers, const char *phase, double t_turn, TopK *top);
void dsv4_head_from_h(Engine *e, TopK *top);
void dsv4_engine_generate(Engine *e, const char *prompt, int ngen, int think,
                          int emit_layers);

#endif /* DSV4_SERVE_INT_H */

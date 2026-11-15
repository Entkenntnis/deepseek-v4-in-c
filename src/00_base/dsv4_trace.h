/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trace.h - opt-in, timestamped record of routed-expert scheduling.
 *
 * WHY THIS EXISTS
 *   expert_io says how LONG the main thread blocked, never WHY or on whom.
 *   The interesting question for any change to the READ side -- per-tensor
 *   publication, more readers, a different stream cap -- is the interleaving:
 *   which expert was resident at begin(), when did each miss land, and how
 *   much arithmetic still ran after the last publish. Those are the numbers
 *   that decide whether starting a matmul earlier can help at all.
 *
 * WHAT IT COSTS
 *   Disabled: one predictable branch per call site. Enabled: a timestamp and a
 *   64-byte struct copy under a small mutex, no formatting and no I/O on the
 *   reader threads. The file is formatted and flushed by the main thread at
 *   the end of a traced token.
 *
 * SCOPE
 *   Decode only. Prefill and hash-layer prefetch may run while tracing, but
 *   their events are dropped unless a decode token is open.
 */
#ifndef DSV4_TRACE_H
#define DSV4_TRACE_H

#include <stdint.h>

/* Event kinds. Upper case is a decision, lower case a completion. */
enum {
    DSV4_TR_BATCH       = 'B',  /* begin() decided: n requests, ntodo misses  */
    DSV4_TR_HIT         = 'H',  /* k usable at begin(): resident/prefetch/dup
                                 * or pf-wait (blocked on a prefetch read)     */
    DSV4_TR_QUEUED      = 'Q',  /* miss handed to the reader FIFO              */
    DSV4_TR_READ_START  = 'R',  /* reader dequeued it and is about to pread()  */
    DSV4_TR_PUBLISHED   = 'P',  /* slot usable: out[k] set, pending cleared    */
    DSV4_TR_WAIT        = 'W',  /* wait_any() entered                          */
    DSV4_TR_WAKE        = 'K',  /* wait_any() returned one k                   */
    DSV4_TR_WAIT_END    = 'Z',  /* wait_any() found nothing left               */
    DSV4_TR_COMPUTE     = 'C',  /* expert_forward() starts                     */
    DSV4_TR_MM          = 'M',  /* one matmul inside expert_forward()          */
    DSV4_TR_SWIGLU      = 'G',  /* silu(gate) * up                             */
    DSV4_TR_COMPUTE_END = 'c',  /* expert_forward() done                       */
    DSV4_TR_SHARED      = 'S',  /* the shared expert finished                  */
    DSV4_TR_BARRIER     = 'F',  /* end() had to wait for outstanding reads     */
    DSV4_TR_LAYER_END   = 'E',  /* moe() of this layer is complete             */
};

/* Field use per kind (unused fields are zero):
 *   B  layer n=requested ntodo=misses k=dups slot=prefetch-waits
 *   H  layer k expert slot tag=resident|prefetch|dup|pf-wait ms=wait
 *   Q  layer k expert slot tag=pf
 *   R  layer k expert slot reader
 *   P  layer k expert slot reader bytes ms tag=read|prefetch|fail
 *   W  layer
 *   K  layer k expert
 *   Z  layer ms
 *   C/c layer k expert tag=hit|miss|late|shared
 *   M  layer k ms tag=w1|w3|w2
 *   G  layer k ms
 *   S  layer ms
 *   F  layer ms
 *   E  layer ms
 */
typedef struct {
    int        kind;
    int        layer, k, expert, slot, reader;
    int        n, ntodo;
    long long  bytes;
    double     ms;
    double     at;      /* absolute start time when emitted at completion,
                         * 0 = stamp at dsv4_trace_ev() entry */
    const char *tag;
} DSV4TraceEv;

/* Open `path` for writing. Returns 0 on success, -1 when the file cannot be
 * created. Opening an empty/NULL path is a no-op and leaves tracing off. */
int  dsv4_trace_open(const char *path);
void dsv4_trace_close(void);

/* Trace on/off and token selection. `from` is a 0-based decode-token index;
 * `count` is how many tokens to record, with 0 meaning "all from `from` on".
 * Call before the first token; the window is evaluated per token. */
void dsv4_trace_window(long from, long count);
void dsv4_trace_layers(int lo, int hi);   /* -1, -1 = every layer */

/* 1 only while a selected decode token is open. Call sites use this one. */
int  dsv4_trace_active(void);

/* Boundaries of one decode token. begin() counts the token and starts the
 * timer; end() formats and flushes. Both are cheap no-ops when off. */
void dsv4_trace_token_begin(int pos, const char *phase);
void dsv4_trace_token_end(void);

void dsv4_trace_ev(const DSV4TraceEv *e);

#endif /* DSV4_TRACE_H */

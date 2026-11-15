/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prof.h - the engine-wide time accounting.
 *
 * One struct, one clock, no dependencies. Every layer may add to it; the CLI
 * snapshots it before a turn and reports the difference. It lives in base/
 * because decode, prefill and the CLI all write it, and none of them owns it.
 *
 * expert_io vs expert_mm is the split that matters: one is a disk read, the
 * other is arithmetic, and a change that helps one can starve the other. The
 * indexer is a subset of attn and must not be added to the total twice.
 */
#ifndef DSV4_PROF_H
#define DSV4_PROF_H

typedef struct {
    double hc;          /* mHC pre/post, Sinkhorn, the two RMSNorms   */
    double attn;        /* CSA + HCA + RoPE + the output projection   */
    double idxer;       /* subset of attn: indexer score + top-k      */
    double gate;        /* router matmul and top-k selection          */
    double expert_io;   /* waiting for the expert cache               */
    double expert_mm;   /* routed-expert FP4 matmuls                  */
    double shared;      /* the shared expert                          */
    double head;        /* hc_head + final norm + vocab-wide GEMV     */
    double trunk_io;    /* time inside read_run()                     */
    double bind;        /* name resolution, widening, pointer rebinds */
} DSV4Prof;

extern DSV4Prof dsv4_prof;

/* Monotonic seconds. One implementation for the whole engine so that timers
 * taken in different modules are directly comparable. */
double dsv4_now_s(void);

#endif /* DSV4_PROF_H */

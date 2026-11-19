/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_time.h - the one monotonic clock the engine still needs.
 *
 * Timings survive only where they feed something a default view displays:
 * token TTFT, prefill progress/duration, kv build/load time and the output
 * head's share of TTFT. Profiling buckets and scheduling traces are gone.
 */
#ifndef DSV4_TIME_H
#define DSV4_TIME_H

/* Monotonic seconds. One implementation for the whole engine so timers taken
 * in different modules are directly comparable. */
double dsv4_now_s(void);

#endif /* DSV4_TIME_H */

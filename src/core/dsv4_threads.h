/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_threads.h - where the compute team and the I/O readers run.
 *
 * EXPLICITLY TUNED FOR THE INTEL CORE ULTRA 7 258V (Lunar Lake), NOT PORTABLE.
 * There is deliberately no runtime hybrid detection. On this machine:
 *
 *     CPUs 0-3   P-cores (Lion Cove), 4.7-4.8 GHz
 *     CPUs 4-7   LP E-cores (Skymont), 3.7 GHz
 *
 * The measurements behind the defaults live in BENCHMARK.md. In short: E-cores
 * are 2.4x slower at the FP4 expert kernel and worse still at the FP64 mHC
 * dot products, while FP8 attention is nearly as fast on them. The compute team
 * therefore runs on the P-cores and the expert-reader pthreads can be sent to
 * the E-cores, where they mostly sleep in pread().
 *
 * If this is built for another CPU, pass `--pin off` (or `--pin all`), or edit
 * DSV4_P_LO..DSV4_E_HI in dsv4_threads.c. The engine still works; it simply
 * runs with the inherited affinity.
 */
#ifndef DSV4_THREADS_H
#define DSV4_THREADS_H

typedef enum {
    DSV4_PIN_OFF = 0,   /* do not call sched_setaffinity at all            */
    DSV4_PIN_ALL,       /* every CPU in the process's inherited mask       */
    DSV4_PIN_P,         /* P-core group (0-3 on the 258V)                  */
    DSV4_PIN_E          /* E-core group (4-7 on the 258V)                  */
} DSV4Pin;

typedef struct {
    int pin;            /* DSV4Pin for the compute/OpenMP team             */
    int threads;        /* omp_set_num_threads; 0 = leave the runtime alone */
    int readers;        /* expert-reader pool width; 0 = cache default (6)  */
    int reader_pin;     /* DSV4Pin for the reader pthreads                 */
} DSV4ThreadOpts;

/* "off"/"none", "all", "p"/"pcores", "e"/"ecores". Returns -1 if unknown. */
int  dsv4_threads_parse_pin(const char *s);

/* Apply the policy. Must be called AFTER argument parsing and BEFORE the first
 * OpenMP region and before dsv4_cache_init(): libgomp worker threads and the
 * cache's reader pthreads inherit the calling thread's affinity mask. Returns
 * 0 on success; an empty requested group falls back to the inherited mask. */
int  dsv4_threads_apply(const DSV4ThreadOpts *o);

/* Re-pin the CALLING thread according to opts.reader_pin. Called by each cache
 * reader thread at startup, because those threads inherited the compute mask. */
void dsv4_threads_pin_current_reader(void);

/* One-line human-readable description of what apply() did, for --help output,
 * logs and the TUI's ready event. Never NULL. */
const char *dsv4_threads_desc(void);

#endif /* DSV4_THREADS_H */

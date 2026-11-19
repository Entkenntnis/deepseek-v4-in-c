/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_threads.h - where the compute team and the I/O readers run.
 *
 * EXPLICITLY TUNED FOR THE INTEL CORE ULTRA 7 258V (Lunar Lake), NOT PORTABLE.
 * There is deliberately no runtime hybrid detection. On this machine:
 *
 *     CPUs 0-3   P-cores (Lion Cove), 4.7-4.8 GHz
 *     CPUs 4-7   LP E-cores (Skymont), 3.7 GHz
 *
 * The defaults follow what this machine does: the E-cores are much slower at
 * the FP4 expert kernel and the FP64 mHC dot products, while FP8 attention is
 * closer. The compute team therefore runs on the P-cores and the expert-reader
 * pthreads can be sent to the E-cores, where they mostly sleep in pread().
 * The decode microbenchmarks that measured this were removed with the other C benchmarks.
 *
 * The policy is fixed at build time; edit the constants in dsv4_threads.c for
 * different silicon.
 */
#ifndef DSV4_THREADS_H
#define DSV4_THREADS_H

/* Apply the fixed policy in dsv4_threads.c. Must be called BEFORE the first
 * OpenMP region and before dsv4_cache_init(): libgomp worker threads and the
 * cache's reader pthreads inherit the calling thread's affinity mask. Returns
 * 0 on success; an empty requested group falls back to the inherited mask. */
int  dsv4_threads_apply(void);

/* Re-pin the CALLING thread according to the fixed reader policy. Called by
 * each cache reader thread at startup, because those threads inherited the
 * compute mask. */
void dsv4_threads_pin_current_reader(void);

/* Fixed reader-pool width (6). The cache uses this directly. */
int  dsv4_threads_readers(void);

#endif /* DSV4_THREADS_H */

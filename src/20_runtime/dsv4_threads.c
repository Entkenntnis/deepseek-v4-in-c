/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_threads.c - see dsv4_threads.h.
 *
 * The CPU groups are compiled in, not discovered. This engine is tuned for one
 * machine (Intel Core Ultra 7 258V) and a sysfs probe would be a portability
 * claim the rest of the file does not make. The constants below are the whole
 * CPU policy; change them for different silicon.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>

#include <omp.h>

#include <sched.h>

#include "dsv4_threads.h"

/* Intel Core Ultra 7 258V: 4 P-cores then 4 LP E-cores, no SMT. */
#define DSV4_P_LO 0
#define DSV4_P_HI 3
#define DSV4_E_LO 4
#define DSV4_E_HI 7

/* The fixed policy. Change these for different silicon. */
#define DSV4_COMPUTE_THREADS 4
#define DSV4_READERS         6

static cpu_set_t g_orig;
static int       g_have_orig;

static void all_cpus(cpu_set_t *s)
{
    CPU_ZERO(s);
    for (int c = 0; c < CPU_SETSIZE; c++) CPU_SET(c, s);
}

static void inherited_mask(cpu_set_t *s)
{
    CPU_ZERO(s);
    if (sched_getaffinity(0, sizeof *s, s) != 0) all_cpus(s);
    if (CPU_COUNT(s) == 0) all_cpus(s);
}

static void group_mask(cpu_set_t *out, int lo, int hi, const cpu_set_t *base)
{
    CPU_ZERO(out);
    for (int c = lo; c <= hi; c++)
        if (CPU_ISSET(c, base)) CPU_SET(c, out);
}

static void mask_str(const cpu_set_t *m, char *out, size_t cap)
{
    size_t n = 0;
    out[0] = '\0';
    for (int c = 0; c < CPU_SETSIZE && n + 8 < cap; c++)
        if (CPU_ISSET(c, m))
            n += (size_t)snprintf(out + n, cap - n, "%s%d", n ? "," : "", c);
}

int dsv4_threads_apply(void)
{
    omp_set_num_threads(DSV4_COMPUTE_THREADS);

    inherited_mask(&g_orig);
    g_have_orig = 1;

    cpu_set_t want;
    group_mask(&want, DSV4_P_LO, DSV4_P_HI, &g_orig);

    char cpus[96];
    mask_str(&g_orig, cpus, sizeof cpus);
    int pinned = 0;
    if (CPU_COUNT(&want) == 0) {
        fprintf(stderr, "dsv4_threads: P-cores are empty in this process's "
                        "CPU mask (%s); leaving affinity alone\n", cpus);
    } else if (sched_setaffinity(0, sizeof want, &want) != 0) {
        perror("dsv4_threads: sched_setaffinity");
    } else {
        pinned = 1;
        mask_str(&want, cpus, sizeof cpus);
    }

    fprintf(stderr,
            "threads: compute %d on CPUs %s%s, readers %d on E-cores\n",
            DSV4_COMPUTE_THREADS, cpus, pinned ? "" : " (no affinity)",
            DSV4_READERS);
    return 0;
}

void dsv4_threads_pin_current_reader(void)
{
    if (!g_have_orig) return;
    cpu_set_t want;
    group_mask(&want, DSV4_E_LO, DSV4_E_HI, &g_orig);
    if (CPU_COUNT(&want) == 0) return;
    if (sched_setaffinity(0, sizeof want, &want) != 0)
        perror("dsv4_threads: reader sched_setaffinity");
}

int dsv4_threads_readers(void)
{
    return DSV4_READERS;
}

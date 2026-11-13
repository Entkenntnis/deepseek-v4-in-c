/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_threads.c - see dsv4_threads.h.
 *
 * The CPU groups are compiled in, not discovered. This engine is tuned for one
 * machine (Intel Core Ultra 7 258V) and a sysfs probe would be a portability
 * claim the rest of the file does not make. The two constants below are the
 * whole CPU policy; change them for different silicon, or use --pin off/all.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <sched.h>
#endif

#include "dsv4_threads.h"

/* Intel Core Ultra 7 258V: 4 P-cores then 4 LP E-cores, no SMT. */
#define DSV4_P_LO 0
#define DSV4_P_HI 3
#define DSV4_E_LO 4
#define DSV4_E_HI 7

static DSV4ThreadOpts g_opt;
static char g_desc[192] = "threads: inherited";

#if defined(__linux__)
static cpu_set_t g_orig;
static int       g_have_orig;
#endif

int dsv4_threads_parse_pin(const char *s)
{
    if (!s || !*s || !strcmp(s, "off") || !strcmp(s, "none")) return DSV4_PIN_OFF;
    if (!strcmp(s, "all")) return DSV4_PIN_ALL;
    if (!strcmp(s, "p") || !strcmp(s, "pcores")) return DSV4_PIN_P;
    if (!strcmp(s, "e") || !strcmp(s, "ecores")) return DSV4_PIN_E;
    return -1;
}

#if defined(__linux__)

static const char *pin_name(int p)
{
    switch (p) {
    case DSV4_PIN_ALL: return "all";
    case DSV4_PIN_P:   return "P-cores";
    case DSV4_PIN_E:   return "E-cores";
    default:           return "off";
    }
}

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

#endif /* __linux__ */

int dsv4_threads_apply(const DSV4ThreadOpts *o)
{
    g_opt = *o;
#ifdef _OPENMP
    if (o->threads > 0) omp_set_num_threads(o->threads);
#endif

#if defined(__linux__)
    inherited_mask(&g_orig);
    g_have_orig = 1;

    int pinned = 0;
    char cpus[96];
    mask_str(&g_orig, cpus, sizeof cpus);

    if (o->pin != DSV4_PIN_OFF) {
        cpu_set_t want;
        switch (o->pin) {
        case DSV4_PIN_ALL: want = g_orig; break;
        case DSV4_PIN_P:   group_mask(&want, DSV4_P_LO, DSV4_P_HI, &g_orig); break;
        case DSV4_PIN_E:   group_mask(&want, DSV4_E_LO, DSV4_E_HI, &g_orig); break;
        default:           CPU_ZERO(&want); break;
        }
        if (CPU_COUNT(&want) == 0) {
            fprintf(stderr, "dsv4_threads: %s is empty in this process's CPU "
                            "mask (%s); leaving affinity alone\n",
                    pin_name(o->pin), cpus);
        } else {
            if (sched_setaffinity(0, sizeof want, &want) != 0) {
                perror("dsv4_threads: sched_setaffinity");
            } else {
                pinned = 1;
                mask_str(&want, cpus, sizeof cpus);
            }
        }
    }

    int nth = o->threads;
#ifdef _OPENMP
    if (nth <= 0) nth = omp_get_max_threads();
#else
    if (nth <= 0) nth = 1;
#endif

    snprintf(g_desc, sizeof g_desc,
             "threads: compute %d on CPUs %s%s, readers %d%s",
             nth, cpus, pinned ? "" : " (no affinity)",
             o->readers > 0 ? o->readers : 6,
             o->reader_pin == DSV4_PIN_OFF ? ""
                 : (o->reader_pin == DSV4_PIN_E ? " on E-cores"
                                                : " on their own group"));
#else
    snprintf(g_desc, sizeof g_desc,
             "threads: no Linux affinity support, compute %d, readers %d",
             o->threads > 0 ? o->threads : 1,
             o->readers > 0 ? o->readers : 6);
#endif

    fprintf(stderr, "%s\n", g_desc);
    return 0;
}

void dsv4_threads_pin_current_reader(void)
{
#if defined(__linux__)
    if (!g_have_orig || g_opt.reader_pin == DSV4_PIN_OFF) return;

    cpu_set_t want;
    switch (g_opt.reader_pin) {
    case DSV4_PIN_ALL: want = g_orig; break;
    case DSV4_PIN_P:   group_mask(&want, DSV4_P_LO, DSV4_P_HI, &g_orig); break;
    case DSV4_PIN_E:   group_mask(&want, DSV4_E_LO, DSV4_E_HI, &g_orig); break;
    default: return;
    }
    if (CPU_COUNT(&want) == 0) return;
    if (sched_setaffinity(0, sizeof want, &want) != 0)
        perror("dsv4_threads: reader sched_setaffinity");
#endif
}

const char *dsv4_threads_desc(void)
{
    return g_desc;
}

int dsv4_threads_readers(void)
{
    return g_opt.readers > 0 ? g_opt.readers : 6;
}

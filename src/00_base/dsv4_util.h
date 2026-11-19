/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_util.h - the monotonic clock and one memory-availability query. */
#ifndef DSV4_UTIL_H
#define DSV4_UTIL_H

#include <stdint.h>
#include <stdio.h>

/* Monotonic seconds. */
double dsv4_now_s(void);

/* Linux MemAvailable in bytes, or -1. A snapshot, not a reservation. */
static inline int64_t dsv4_mem_available_bytes(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long long kb = -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
    fclose(f);
    return kb < 0 ? -1 : (int64_t)kb * 1024;
}

#endif /* DSV4_UTIL_H */

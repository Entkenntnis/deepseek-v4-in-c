/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_mem.h - how much RAM may this process still take?
 *
 * This exists for `--budget auto`, and it deliberately reports MemAvailable
 * rather than MemFree: Linux's MemAvailable already accounts for reclaimable
 * page cache, which MemFree does not. Using MemFree would understate a laptop
 * that has been reading model files all morning and leave most of the machine
 * unused.
 *
 * The value is a snapshot, not a reservation. Another process can consume the
 * memory between this call and the allocations that follow, which is why the
 * caller keeps a reserve on top of it.
 *
 * The engine targets Linux and reads MemAvailable from /proc/meminfo.
 */
#ifndef DSV4_MEM_H
#define DSV4_MEM_H

#include <stdint.h>
#include <stdio.h>

static int64_t dsv4_mem_available_bytes(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;

    char line[256];
    long long kb = -1;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
    }
    fclose(f);

    if (kb < 0) return -1;
    return (int64_t)kb * 1024;
}

#endif /* DSV4_MEM_H */

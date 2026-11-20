/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_util.h - the monotonic clock and a memory-advice helper. */
#ifndef DSV4_UTIL_H
#define DSV4_UTIL_H

#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

/* Monotonic seconds. */
double dsv4_now_s(void);

/* Ask the kernel to back this range with transparent huge pages. This is only
 * advice: an unsupported kernel or a failed madvise simply leaves the range
 * with normal pages. */
static inline void dsv4_hugepages(void *p, size_t bytes)
{
#ifdef MADV_HUGEPAGE
    if (!p || !bytes) return;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    if (!page) return;
    const uintptr_t base = (uintptr_t)p & ~(uintptr_t)(page - 1);
    const uintptr_t end = ((uintptr_t)p + bytes + page - 1)
                        & ~(uintptr_t)(page - 1);
    (void)madvise((void *)base, (size_t)(end - base), MADV_HUGEPAGE);
#else
    (void)p;
    (void)bytes;
#endif
}

#endif /* DSV4_UTIL_H */

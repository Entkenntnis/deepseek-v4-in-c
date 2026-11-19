/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_time.c - the monotonic clock used by the protocol and benchmarks. */
#include <time.h>

#include "dsv4_util.h"

double dsv4_now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

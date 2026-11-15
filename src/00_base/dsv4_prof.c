/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_prof.c - see dsv4_prof.h. */
#include <time.h>

#include "dsv4_prof.h"

DSV4Prof dsv4_prof;

double dsv4_now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

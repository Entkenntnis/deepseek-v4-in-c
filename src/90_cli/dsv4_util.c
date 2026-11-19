/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_util.c - timing, process stats and the cancel flag. */
#include <signal.h>
#include <stdio.h>

#include "dsv4_serve_int.h"

volatile sig_atomic_t dsv4_cancel;

double dsv4_proc_rss_mb(void)
{
    double rss_mb = 0.0;
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            double kb;
            if (sscanf(line, "VmRSS: %lf kB", &kb) == 1) {
                rss_mb = kb / 1024.0;
                break;
            }
        }
        fclose(f);
    }
    return rss_mb;
}

double dsv4_proc_read_gb(void)
{
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return 0.0;
    char line[256];
    unsigned long long bytes = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "read_bytes: %llu", &bytes) == 1) break;
    }
    fclose(f);
    return (double)bytes / 1073741824.0;
}



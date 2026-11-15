/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_util.c - timing, process stats, JSON helpers and the cancel flag. */
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "dsv4_serve_int.h"

volatile sig_atomic_t dsv4_cancel;

double dsv4_proc_cpu_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void dsv4_proc_mem_mb(double *rss_mb, double *peak_mb)
{
    *rss_mb = 0.0;
    *peak_mb = 0.0;
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            double kb;
            if (sscanf(line, "VmRSS: %lf kB", &kb) == 1) *rss_mb = kb / 1024.0;
            else if (sscanf(line, "VmHWM: %lf kB", &kb) == 1) *peak_mb = kb / 1024.0;
        }
        fclose(f);
    }
    if (*peak_mb <= 0.0) {
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0)
            *peak_mb = (double)ru.ru_maxrss / 1024.0;   /* Linux: KiB */
    }
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

void dsv4_json_escape(const char *s)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        switch (*p) {
        case '"':  fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\b': fputs("\\b", stdout); break;
        case '\f': fputs("\\f", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if (*p < 0x20) printf("\\u%04x", *p);
            else putchar(*p);
        }
    }
    putchar('"');
}

void dsv4_emit_error(const char *fmt, ...)
{
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fputs("{\"event\":\"error\",\"message\":", stdout);
    dsv4_json_escape(msg);
    fputs("}\n", stdout);
    fflush(stdout);
}

void dsv4_usage(const char *me)
{
    fprintf(stderr,
            "usage: %s <model_dir> --trunk <dir> --tok <file> "
            "[--budget GB|auto] [--ctx N] [--gen N] [--think] [--no-layers]\n"
            "       [--threads N] [--pin off|all|p|e] [--readers N]\n"
            "       [--reader-pin off|all|p|e] [--stream-miss N (default 6)]\n"
            "\n"
            "Speaks JSON lines on stdin/stdout. Start it from the TUI:\n"
            "  python3 tui/dsv4_tui.py\n", me);
}

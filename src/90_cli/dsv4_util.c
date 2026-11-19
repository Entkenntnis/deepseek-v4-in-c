/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_util.c - timing, process stats, JSON helpers and the cancel flag. */
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* String fields travel as hex. That keeps the line protocol ASCII-only and
 * means a path or an error message needs no quoting, escaping, or Unicode
 * handling on this side. */
void dsv4_emit_hex_field(const char *key, const char *s)
{
    static const char hx[] = "0123456789abcdef";
    printf("%s=hex:", key);
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        putchar(hx[*p >> 4]);
        putchar(hx[*p & 15]);
    }
}

int dsv4_hex_decode(const char *s, char *out, int cap)
{
    int n = 0;
    for (; s[0] && s[1]; s += 2) {
        int hi = -1, lo = -1;
        const char a = s[0], b = s[1];
        if (a >= '0' && a <= '9') hi = a - '0';
        else if (a >= 'a' && a <= 'f') hi = a - 'a' + 10;
        else if (a >= 'A' && a <= 'F') hi = a - 'A' + 10;
        if (b >= '0' && b <= '9') lo = b - '0';
        else if (b >= 'a' && b <= 'f') lo = b - 'a' + 10;
        else if (b >= 'A' && b <= 'F') lo = b - 'A' + 10;
        if (hi < 0 || lo < 0 || n + 1 >= cap) return -1;
        out[n++] = (char)((hi << 4) | lo);
    }
    if (s[0]) return -1;                 /* odd number of hex digits */
    out[n] = 0;
    return n;
}

void dsv4_emit_error(const char *fmt, ...)
{
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fputs("error ", stdout);
    dsv4_emit_hex_field("message", msg);
    putchar('\n');
    fflush(stdout);
}

void dsv4_usage(const char *me)
{
    fprintf(stderr,
            "usage: %s <model_dir> --trunk <dir> "
            "[--budget GB|auto] [--ctx N] [--gen N]\n"
            "\n"
            "Speaks a line protocol on stdin/stdout: one command or event per\n"
            "line, whitespace-separated key=value fields, strings as hex:... .\n"
            "Start it from the TUI:\n"
            "  python3 tui/dsv4_tui.py\n", me);
}

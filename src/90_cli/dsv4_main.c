/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_main.c - argument parsing, process setup and the command loop. */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_serve_int.h"
#include "dsv4_threads.h"

static void usage(const char *me)
{
    fprintf(stderr,
            "usage: %s --model <dir> --trunk <dir> "
            "[--budget GB|auto] [--ctx N] [--gen N]\n"
            "\n"
            "Speaks a line protocol on stdin/stdout: one command or event per\n"
            "line, whitespace-separated key=value fields, strings as hex:... .\n"
            "Start it from the TUI:\n"
            "  python3 tui/dsv4_tui.py\n", me);
}

static void handle_cancel(int sig)
{
    (void)sig;
    dsv4_cancel = 1;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *trunkdir = NULL;
    double budget_gb = 16.0;
    int ctx = 8192, ngen = 128;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc)
            trunkdir = argv[++i];
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) {
            const char *v = argv[++i];
            budget_gb = strcmp(v, "auto") ? atof(v) : -1.0;
        }
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc)
            ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc)
            ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!model || !trunkdir) {
        usage(argv[0]);
        return 2;
    }

    struct sigaction cancel_sa;
    memset(&cancel_sa, 0, sizeof cancel_sa);
    cancel_sa.sa_handler = handle_cancel;
    sigemptyset(&cancel_sa.sa_mask);
    cancel_sa.sa_flags = SA_RESTART;
    if (sigaction(SIGUSR1, &cancel_sa, NULL) != 0)
        fprintf(stderr, "warning: cannot install cancel handler: %s\n",
                strerror(errno));

    /* Thread policy before engine_init(): the cache's reader pthreads inherit
     * the calling thread's mask. This is fixed, not a command-line option. */
    dsv4_threads_apply();

    /* Line buffering is not enough on a pipe: every event must be visible to
     * the TUI the moment it is produced. */
    static char outbuf[1 << 16];
    setvbuf(stdout, outbuf, _IOFBF, sizeof outbuf);

    Engine e;
    if (dsv4_engine_init(&e, model, trunkdir, budget_gb, ctx, ngen) != 0) {
        dsv4_emit_error("engine init failed "
                        "(see stderr for the loader's message)");
        dsv4_engine_free(&e);
        return 1;
    }

    dsv4_emit_ready(&e, "ready");

    char *line = NULL;
    size_t cap = 0;
    for (;;) {
        ssize_t n = getline(&line, &cap, stdin);
        if (n < 0) break;                       /* EOF: parent went away */
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r'))
            line[--n] = 0;
        if (n == 0) continue;
        if (dsv4_handle_command(&e, line)) break;
    }
    free(line);

    /* Readers may still be finishing a prefetch; engine_free() stops the pool. */
    dsv4_engine_free(&e);
    return 0;
}

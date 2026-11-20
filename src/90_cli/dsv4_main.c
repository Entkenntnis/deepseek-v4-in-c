/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_main.c - option parsing, process setup and the command loop. */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_serve_int.h"
#include "dsv4_threads.h"

#define BUDGET_AUTO  (-1.0)
#define EXIT_USAGE   2
#define PARSE_OK     0
#define PARSE_HELP   1
#define PARSE_ERROR  2

typedef struct {
    const char *model_dir;
    const char *trunk_dir;
    double      budget_gb;
    int         budget_auto;
    int         context_tokens;
    int         default_generation;
} Options;

static void usage(FILE *out, const char *prog)
{
    fprintf(out,
            "usage: %s --model <dir> --trunk <dir> "
            "[--budget GB|auto] [--ctx N] [--gen N]\n"
            "\n"
            "Speaks a line protocol on stdin/stdout: one command or event per\n"
            "line, whitespace-separated key=value fields, strings as hex:... .\n"
            "Start it from the TUI:\n"
            "  python3 tui/dsv4_tui.py\n", prog);
}

static void handle_cancel(int sig)
{
    (void)sig;
    dsv4_runtime_request_cancel();
}

static int parse_options(int argc, char **argv, Options *o)
{
    const char *prog = argv[0];

    o->model_dir = NULL;
    o->trunk_dir = NULL;
    o->budget_gb = 16.0;
    o->budget_auto = 0;
    o->context_tokens = 8192;
    o->default_generation = 128;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            o->model_dir = argv[++i];
        } else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) {
            o->trunk_dir = argv[++i];
        } else if (!strcmp(argv[i], "--budget") && i + 1 < argc) {
            const char *v = argv[++i];
            o->budget_auto = !strcmp(v, "auto");
            if (!o->budget_auto) o->budget_gb = atof(v);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            o->context_tokens = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--gen") && i + 1 < argc) {
            o->default_generation = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--help")) {
            usage(stdout, prog);
            return PARSE_HELP;
        } else {
            usage(stderr, prog);
            return PARSE_ERROR;
        }
    }

    if (!o->model_dir || !o->trunk_dir) {
        usage(stderr, prog);
        return PARSE_ERROR;
    }
    return PARSE_OK;
}

static void install_cancel_handler(void)
{
    struct sigaction cancel_sa;
    memset(&cancel_sa, 0, sizeof cancel_sa);
    cancel_sa.sa_handler = handle_cancel;
    sigemptyset(&cancel_sa.sa_mask);
    cancel_sa.sa_flags = SA_RESTART;
    if (sigaction(SIGUSR1, &cancel_sa, NULL) != 0)
        fprintf(stderr, "warning: cannot install cancel handler: %s\n",
                strerror(errno));
}

static void setup_stdout(void)
{
    /* The protocol is line-oriented; line buffering makes events visible as
     * soon as they are written. The emit functions flush explicitly as well. */
    if (setvbuf(stdout, NULL, _IOLBF, 0) != 0)
        fprintf(stderr, "warning: cannot configure line-buffered stdout\n");
}

static void run_command_loop(Engine *e)
{
    char *line = NULL;
    size_t cap = 0;

    while (getline(&line, &cap, stdin) >= 0) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;
        if (dsv4_handle_command(e, line) == DSV4_COMMAND_QUIT) break;
    }
    free(line);
}

int main(int argc, char **argv)
{
    Options opt;
    const int parsed = parse_options(argc, argv, &opt);
    if (parsed == PARSE_HELP) return EXIT_SUCCESS;
    if (parsed != PARSE_OK) return EXIT_USAGE;

    install_cancel_handler();

    /* Thread policy before engine_init(): the cache's reader pthreads inherit
     * the calling thread's mask. This is fixed, not a command-line option. */
    dsv4_threads_apply();
    setup_stdout();

    Engine engine = {0};
    const double budget = opt.budget_auto ? BUDGET_AUTO : opt.budget_gb;
    if (dsv4_engine_init(&engine, opt.model_dir, opt.trunk_dir, budget,
                         opt.context_tokens, opt.default_generation) != 0) {
        dsv4_emit_error("engine init failed "
                        "(see stderr for the loader's message)");
        dsv4_engine_free(&engine);
        return EXIT_FAILURE;
    }

    dsv4_emit_ready(&engine, "ready");
    run_command_loop(&engine);

    /* Readers may still be finishing a prefetch; engine_free() stops the pool. */
    dsv4_engine_free(&engine);
    return EXIT_SUCCESS;
}

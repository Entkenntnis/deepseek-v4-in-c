/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_main.c - argument parsing, commands and the command loop. */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_cfg.h"
#include "dsv4_serve_int.h"
#include "json.h"
#include "dsv4_threads.h"

static const char *jstr(jval *o, const char *key)
{
    jval *v = json_get(o, key);
    return v && v->t == J_STR ? v->str : NULL;
}

static double jnum(jval *o, const char *key, double dflt)
{
    jval *v = json_get(o, key);
    return v && v->t == J_NUM ? v->num : dflt;
}

static int jbool(jval *o, const char *key, int dflt)
{
    jval *v = json_get(o, key);
    if (!v) return dflt;
    if (v->t == J_BOOL) return v->boolean ? 1 : 0;
    if (v->t == J_NUM)  return v->num != 0.0;
    return dflt;
}

/* json.h deliberately has no free: it was written for parse-once-at-startup
 * files. A command stream parses repeatedly, so this walks the tree and frees
 * every j_dup'd string too. */
static void jfree(jval *v)
{
    if (!v) return;
    if (v->t == J_STR) free(v->str);
    for (int i = 0; i < v->len; i++) {
        if (v->kids) jfree(v->kids[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->kids);
    free(v->keys);
    free(v);
}

static void handle_cancel(int sig)
{
    (void)sig;
    dsv4_cancel = 1;
}

static void handle_command(Engine *e, jval *v)
{
    const char *cmd = jstr(v, "cmd");
    if (!cmd) { dsv4_emit_error("command object has no \"cmd\""); return; }

    if (!strcmp(cmd, "generate")) {
        const char *prompt = jstr(v, "prompt");
        if (!prompt) prompt = "";
        const int ngen = (int)jnum(v, "gen", e->gen_default);
        const int think = jbool(v, "think", e->think_default);
        const int layers = jbool(v, "layers", e->emit_layers_default);
        dsv4_engine_generate(e, prompt, ngen, think, layers);
    } else if (!strcmp(cmd, "reset")) {
        if (dsv4_engine_reset_context(e) != 0)
            dsv4_emit_error("context reset failed");
        else {
            printf("{\"event\":\"reset\",");
            dsv4_write_sys_full(e);
            putchar(',');
            dsv4_write_cache(e);
            putchar(',');
            dsv4_write_trunk(e);
            fputs("}\n", stdout);
            fflush(stdout);
        }
    } else if (!strcmp(cmd, "system")) {
        const char *text = jstr(v, "text");
        free(e->system_prompt);
        e->system_prompt = text ? strdup(text) : NULL;
        printf("{\"event\":\"system\",\"ok\":true,\"chars\":%d}\n",
               text ? (int)strlen(text) : 0);
        fflush(stdout);
    } else if (!strcmp(cmd, "stats")) {
        dsv4_emit_ready(e, "stats");
    } else if (!strcmp(cmd, "ping")) {
        printf("{\"event\":\"pong\"}\n");
        fflush(stdout);
    } else if (!strcmp(cmd, "quit")) {
        /* handled by the caller */
    } else {
        dsv4_emit_error("unknown command \"%s\"", cmd);
    }
}

int main(int argc, char **argv)
{
    const char *model = NULL, *trunkdir = NULL, *tokfile = NULL;
    double budget_gb = 8.0;
    int budget_auto = 0;
    int ctx = 8192, ngen = 128, think = 0, layers = 1;
    const char *pin = "p", *reader_pin = "off";
    int threads = 4, readers = 6;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !model) { model = argv[i]; continue; }
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunkdir = argv[++i];
        else if (!strcmp(argv[i], "--tok") && i + 1 < argc) tokfile = argv[++i];
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "auto")) budget_auto = 1;
            else { budget_auto = 0; budget_gb = atof(v); }
        }
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) pin = argv[++i];
        else if (!strcmp(argv[i], "--readers") && i + 1 < argc)
            readers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reader-pin") && i + 1 < argc)
            reader_pin = argv[++i];
        else if (!strcmp(argv[i], "--stream-miss") && i + 1 < argc)
            dsv4_set_stream_miss(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--think")) think = 1;
        else if (!strcmp(argv[i], "--no-layers")) layers = 0;
        else if (!strcmp(argv[i], "--help")) { dsv4_usage(argv[0]); return 0; }
    }
    if (!model || !trunkdir || !tokfile) { dsv4_usage(argv[0]); return 2; }

    struct sigaction cancel_sa;
    memset(&cancel_sa, 0, sizeof cancel_sa);
    cancel_sa.sa_handler = handle_cancel;
    sigemptyset(&cancel_sa.sa_mask);
    cancel_sa.sa_flags = SA_RESTART;
    if (sigaction(SIGUSR1, &cancel_sa, NULL) != 0)
        fprintf(stderr, "warning: cannot install cancel handler: %s\n",
                strerror(errno));

    /* Thread policy before engine_init(): that is where the expert cache
     * creates its reader pthreads, and they inherit the calling thread's mask. */
    const int pin_id = dsv4_threads_parse_pin(pin);
    const int rpin_id = dsv4_threads_parse_pin(reader_pin);
    if (pin_id < 0 || rpin_id < 0) {
        fprintf(stderr, "invalid --pin/--reader-pin (use off|all|p|e)\n");
        return 2;
    }
    const DSV4ThreadOpts topts = { pin_id, threads, readers, rpin_id };
    dsv4_threads_apply(&topts);

    /* Line buffering is not enough on a pipe: every event must be visible to
     * the TUI the moment it is produced. */
    static char outbuf[1 << 16];
    setvbuf(stdout, outbuf, _IOFBF, sizeof outbuf);

    Engine e;
    if (dsv4_engine_init(&e, model, trunkdir, tokfile, budget_gb, budget_auto,
                         ctx, ngen, think, layers) != 0) {
        dsv4_emit_error("engine init failed (see stderr for the loader's message)");
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

        jval *v = json_parse(line, NULL);
        if (!v || v->t != J_OBJ) {
            dsv4_emit_error("invalid JSON command");
            jfree(v);
            continue;
        }
        const char *cmd = jstr(v, "cmd");
        const int quit = cmd && !strcmp(cmd, "quit");
        handle_command(&e, v);
        jfree(v);
        if (quit) break;
    }
    free(line);

    dsv4_engine_free(&e);
    return 0;
}

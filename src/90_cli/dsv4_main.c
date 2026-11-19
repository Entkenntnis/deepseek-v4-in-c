/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_main.c - argument parsing, commands and the command loop. */
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsv4_serve_int.h"
#include "dsv4_threads.h"

#define MAX_FIELDS 16

typedef struct {
    char *key[MAX_FIELDS];
    char *val[MAX_FIELDS];
    int   n;
} Fields;

static const char *field(const Fields *f, const char *key)
{
    for (int i = 0; i < f->n; i++)
        if (!strcmp(f->key[i], key)) return f->val[i];
    return NULL;
}

static int field_int(const Fields *f, const char *key, int dflt)
{
    const char *v = field(f, key);
    return v ? (int)strtol(v, NULL, 10) : dflt;
}

static int ensure_in_ids(Engine *e, int need)
{
    if (need <= e->in_cap) return 0;
    int cap = e->in_cap ? e->in_cap : 1024;
    while (cap < need) cap *= 2;
    int32_t *p = realloc(e->in_ids, (size_t)cap * sizeof *p);
    if (!p) return -1;
    e->in_ids = p;
    e->in_cap = cap;
    return 0;
}

/* Parse a comma-separated token-id list into e->in_ids. */
static int parse_ids(Engine *e, const char *s, int *n_out)
{
    int n = 0;
    const char *p = s;
    while (*p) {
        char *end = NULL;
        const long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return -1;
        if (ensure_in_ids(e, n + 1) != 0) return -1;
        e->in_ids[n++] = (int32_t)v;
        p = end;
        if (*p == ',') p++;
        else if (*p) return -1;
    }
    *n_out = n;
    return 0;
}

static int decode_path(const char *v, char *out, size_t cap)
{
    if (!v || strncmp(v, "hex:", 4) != 0) return -1;
    if (dsv4_hex_decode(v + 4, out, (int)cap) < 0) return -1;
    return 0;
}

static void handle_cancel(int sig)
{
    (void)sig;
    dsv4_cancel = 1;
}

/* Returns 1 when the command asks the caller to stop. */
static int handle_command(Engine *e, char *line)
{
    char *save = NULL;
    char *cmd = strtok_r(line, " \t\r\n", &save);
    if (!cmd) return 0;

    Fields f = { .n = 0 };
    for (char *tok = strtok_r(NULL, " \t\r\n", &save);
         tok; tok = strtok_r(NULL, " \t\r\n", &save)) {
        char *eq = strchr(tok, '=');
        if (!eq || f.n >= MAX_FIELDS) {
            dsv4_emit_error("malformed field \"%s\"", tok);
            return 0;
        }
        *eq = 0;
        f.key[f.n] = tok;
        f.val[f.n] = eq + 1;
        f.n++;
    }

    if (!strcmp(cmd, "generate")) {
        int nt = 0;
        const char *tv = field(&f, "tokens");
        if (!tv || parse_ids(e, tv, &nt) != 0 || nt == 0) {
            dsv4_emit_error("generate needs a non-empty tokens= list");
            return 0;
        }
        const int eos = field_int(&f, "eos", -1);
        const int ngen = field_int(&f, "gen", e->gen_default);
        const int think = field_int(&f, "think", e->think_default);
        const int layers = field_int(&f, "layers", e->emit_layers_default);
        dsv4_engine_generate(e, e->in_ids, nt, ngen, eos, think, layers);
    } else if (!strcmp(cmd, "reset")) {
        if (dsv4_engine_reset_context(e) != 0)
            dsv4_emit_error("context reset failed");
        else
            dsv4_emit_reset(e, "user");
    } else if (!strcmp(cmd, "kv_build")) {
        char path[4096];
        int nt = 0;
        const char *tv = field(&f, "tokens");
        if (decode_path(field(&f, "path"), path, sizeof path) != 0 ||
            !tv || parse_ids(e, tv, &nt) != 0 || nt == 0) {
            dsv4_emit_kv_error("kv_build needs path=hex:... and tokens=...");
        } else {
            dsv4_engine_kv_build(e, e->in_ids, nt, path);
        }
    } else if (!strcmp(cmd, "kv_load")) {
        char path[4096];
        int nt = 0;
        const char *tv = field(&f, "tokens");
        if (decode_path(field(&f, "path"), path, sizeof path) != 0) {
            dsv4_emit_kv_error("kv_load needs path=hex:...");
        } else if (tv && parse_ids(e, tv, &nt) != 0) {
            dsv4_emit_kv_error("kv_load tokens= is malformed");
        } else {
            dsv4_engine_kv_load(e, e->in_ids, nt, path);
        }
    } else if (!strcmp(cmd, "ping")) {
        fputs("pong\n", stdout);
        fflush(stdout);
    } else if (!strcmp(cmd, "quit")) {
        return 1;
    } else {
        dsv4_emit_error("unknown command \"%s\"", cmd);
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *trunkdir = NULL;
    double budget_gb = 8.0;
    int budget_auto = 0;
    int ctx = 8192, ngen = 128, think = 0, layers = 1;
    /* Readers sit on the E-cores by default: the prefill overlap keeps the
     * P-cores busy with CPU fill while the GPU runs, and active OpenMP
     * workers otherwise starve the reader pool. */
    const char *pin = "p", *reader_pin = "e";
    int threads = 4, readers = 6;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !model) { model = argv[i]; continue; }
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunkdir = argv[++i];
        /* Kept as an ignored option so old launch lines do not mistake the
         * tokenizer path for the model argument. The engine no longer sees
         * text, so it does not load a tokenizer. */
        else if (!strcmp(argv[i], "--tok") && i + 1 < argc) ++i;
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
    if (!model || !trunkdir) { dsv4_usage(argv[0]); return 2; }

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
    if (dsv4_engine_init(&e, model, trunkdir, budget_gb, budget_auto,
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
        if (handle_command(&e, line)) break;
    }
    free(line);

    /* Readers may still be finishing a prefetch; engine_free() stops the pool. */
    dsv4_engine_free(&e);
    return 0;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cmd.c - command-line field parsing and command dispatch. */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_serve_int.h"

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

int dsv4_handle_command(Engine *e, char *line)
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
        const int layers = field_int(&f, "layers", 1);
        dsv4_engine_generate(e, e->in_ids, nt, ngen, eos, layers);
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
            dsv4_emit_error("KV: kv_build needs path=hex:... and tokens=...");
        } else {
            dsv4_engine_kv_build(e, e->in_ids, nt, path);
        }
    } else if (!strcmp(cmd, "kv_load")) {
        char path[4096];
        int nt = 0;
        const char *tv = field(&f, "tokens");
        if (decode_path(field(&f, "path"), path, sizeof path) != 0) {
            dsv4_emit_error("KV: kv_load needs path=hex:...");
        } else if (tv && parse_ids(e, tv, &nt) != 0) {
            dsv4_emit_error("KV: kv_load tokens= is malformed");
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


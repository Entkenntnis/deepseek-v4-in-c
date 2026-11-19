/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_kv.c - build and load a prompt-prefix KV cache.
 *
 * These commands are deliberately small wrappers around the existing prefill
 * and layer-state machinery. Building a cache is just "append the prefix,
 * prefill it, write the state"; loading it is "validate the file, copy the
 * state in". No new arithmetic lives here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "dsv4_serve_int.h"
#include "dsv4_state_file.h"
#include "dsv4_time.h"

static void model_tag(const char *dir, char *out, size_t cap)
{
    if (!out || cap == 0) return;
    out[0] = 0;
    if (!dir) return;

    const char *end = dir + strlen(dir);
    while (end > dir && end[-1] == '/') end--;
    const char *base = dir;
    for (const char *p = dir; p < end; p++)
        if (*p == '/') base = p + 1;
    const size_t n = (size_t)(end - base);
    if (n >= cap) {
        memcpy(out, base, cap - 1);
        out[cap - 1] = 0;
    } else {
        memcpy(out, base, n);
        out[n] = 0;
    }
}

void dsv4_engine_kv_build(Engine *e, const int32_t *tokens, int n_tokens,
                          const char *path)
{
    if (!tokens || n_tokens <= 0 || !path || !path[0]) {
        dsv4_emit_kv_error("kv_build needs both tokens and path");
        return;
    }
    if (e->n_ids != 0 || e->pending_forward) {
        dsv4_emit_kv_error("kv_build needs an empty context");
        return;
    }

    const double t0 = dsv4_now_s();
    dsv4_cancel = 0;

    const int n = dsv4_append_ids(e, tokens, n_tokens);
    if (n < 0) {
        if (n == -2)
            dsv4_emit_kv_error("prefix does not fit in the context");
        else
            dsv4_emit_kv_error("prefix contains an invalid token id");
        return;
    }

    int ok = 0;
    if (e->pf && n > 1 && dsv4_prefill_reserve(e->pf, n) == 0) {
        const int rc = dsv4_prefill_range(e->pf, e->ids, n, 0,
                                          e->cs_d, e->sn_d, e->cs_c, e->sn_c,
                                          e->h);
        dsv4_prefill_release(e->pf);
        ok = rc == 0;
        if (!ok)
            dsv4_emit_kv_error(rc == DSV4_PREFILL_CANCELLED ||
                               rc == DSV4_PREFILL_ABORTED
                               ? "prefix prefill cancelled"
                               : "prefix prefill failed");
    } else {
        /* No block prefill: the per-token path produces the same KV state. */
        ok = 1;
        for (int i = 0; i < n; i++) {
            if (dsv4_forward(e, e->ids[i], i, 0, 0, "prefix",
                             dsv4_now_s() - t0, NULL) != 0) {
                ok = 0;
                dsv4_emit_kv_error("prefix prefill failed");
                break;
            }
        }
    }

    if (!ok) {
        dsv4_engine_reset_context(e);
        dsv4_cancel = 0;
        return;
    }

    char tag[128];
    model_tag(e->model_dir, tag, sizeof tag);
    if (dsv4_state_file_save(path, tag, &e->c, e->lst, e->ids, n) != 0) {
        dsv4_emit_kv_error("could not write the KV cache file");
        dsv4_cancel = 0;
        return;
    }

    struct stat sb;
    long long bytes = stat(path, &sb) == 0 ? (long long)sb.st_size : 0;
    e->kv_prefix = 1;
    dsv4_emit_kv_built(n, dsv4_now_s() - t0, bytes);
    dsv4_cancel = 0;
}

void dsv4_engine_kv_load(Engine *e, const int32_t *tokens, int n_tokens,
                         const char *path)
{
    if (!path || !path[0]) {
        dsv4_emit_kv_error("kv_load needs a path");
        return;
    }
    if (e->n_ids != 0 || e->pending_forward) {
        dsv4_emit_kv_error("kv_load needs an empty context");
        return;
    }

    const int n_expect = tokens ? n_tokens : 0;
    const double t0 = dsv4_now_s();
    char tag[128];
    model_tag(e->model_dir, tag, sizeof tag);
    int loaded = 0;
    if (dsv4_state_file_load(path, tag, &e->c, e->lst, e->maxpos,
                             e->ids, e->cap_ids, &loaded,
                             n_expect > 0 ? tokens : NULL, n_expect) != 0) {
        dsv4_emit_kv_error("cached prefix rejected (see stderr)");
        return;
    }

    e->n_ids = loaded;
    e->pending_forward = 0;
    e->kv_prefix = 1;
    e->prefill_s = 0.0;
    e->prefill_tokens = 0;
    dsv4_emit_kv_loaded(loaded, e->cap_ids, dsv4_now_s() - t0);
}

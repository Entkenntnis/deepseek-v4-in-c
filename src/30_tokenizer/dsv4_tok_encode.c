/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok_encode.c - byte-level BPE merges and added-token matching. */
#include "dsv4_tok.h"
#include "dsv4_tok_int.h"

#define MAXSYM 4096

static int encode_plain(const DSV4Tok *t, const char *text, int n,
                        int32_t *ids, int max_ids)
{
    if (n <= 0) return 0;

    static DSV4Piece a[65536], b[65536], c[65536];
    DSV4Piece whole = { 0, n };
    int na = dsv4_tok_split_isolated(text, &whole, 1, a, 65536, 0);
    int nb = dsv4_tok_split_isolated(text, a, na, b, 65536, 1);
    int nc = dsv4_tok_split_stage2(text, b, nb, c, 65536);

    int nids = 0;
    for (int p = 0; p < nc; p++) {
        /* byte-level: every byte becomes a printable codepoint */
        static char enc[MAXSYM * 4];
        static int  sym_off[MAXSYM], sym_len[MAXSYM];
        static int32_t sym[MAXSYM];
        int ns = 0, w = 0;
        for (int i = 0; i < c[p].len && ns < MAXSYM; i++) {
            const unsigned char by = (unsigned char)text[c[p].off + i];
            sym_off[ns] = w;
            w += dsv4_cp_to_utf8(dsv4_byte_to_cp(by), enc + w);
            sym_len[ns] = w - sym_off[ns];
            sym[ns] = dsv4_tok_find(t, enc + sym_off[ns], sym_len[ns]);
            ns++;
        }

        /* Merge the best-ranked adjacent pair until none applies. Buckets are
         * in ascending rank order, so the first hit in a bucket is that pair's
         * rank and the scan stops there. */
        for (;;) {
            int best = -1, best_at = -1; uint32_t best_out = 0;
            for (int i = 0; i + 1 < ns; i++) {
                if (sym[i] < 0 || sym[i + 1] < 0) continue;
                for (int32_t m = t->pair_head[sym[i]]; m >= 0;
                     m = t->pair_next[m]) {
                    if (t->mb[m] == (uint32_t)sym[i + 1]) {
                        if (best < 0 || m < best) {
                            best = m; best_at = i; best_out = t->mo[m];
                        }
                        break;
                    }
                }
            }
            if (best < 0) break;
            sym[best_at] = (int32_t)best_out;
            for (int i = best_at + 1; i + 1 < ns; i++) sym[i] = sym[i + 1];
            ns--;
        }

        for (int i = 0; i < ns && nids < max_ids; i++)
            if (sym[i] >= 0) ids[nids++] = sym[i];
    }
    return nids;
}

/* Added tokens must be matched BEFORE the pre-tokenizer sees them: they are not
 * BPE material. The first-byte index makes ordinary prose cost one comparison
 * per character rather than 1,283, and longest match wins within a bucket. */
static int32_t  g_add_head[256];
static int32_t *g_add_next;
static const DSV4Tok *g_add_for;

static void build_added_index(const DSV4Tok *t)
{
    if (g_add_for == t) return;
    free(g_add_next);
    g_add_next = (int32_t *)malloc((size_t)t->n_added * sizeof *g_add_next);
    if (!g_add_next) { g_add_for = NULL; return; }
    for (int i = 0; i < 256; i++) g_add_head[i] = -1;

    /* Insert so each bucket ends up sorted by DESCENDING length: walk the ids
     * by increasing length and push each to the front. */
    int32_t *order = (int32_t *)malloc((size_t)t->n_added * sizeof *order);
    if (!order) {
        free(g_add_next); g_add_next = NULL; g_add_for = NULL;
        return;
    }
    /* Only added ids that exist in THIS vocab. A reduced tokenizer can carry
     * added ids from the full vocab that are past its own n_vocab, and
     * indexing vlen[] with those reads off the end. */
    uint32_t nord = 0;
    for (uint32_t i = 0; i < t->n_added; i++)
        if (t->added[i] < t->n_vocab) order[nord++] = (int32_t)i;
    for (uint32_t i = 1; i < nord; i++) {                /* insertion sort */
        const int32_t v = order[i];
        const uint16_t lv = t->vlen[t->added[v]];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && t->vlen[t->added[order[j]]] > lv) {
            order[j + 1] = order[j]; j--;
        }
        order[j + 1] = v;
    }
    for (uint32_t k = 0; k < nord; k++) {
        const int32_t slot = order[k];
        const uint32_t id = t->added[slot];
        if (t->vlen[id] == 0) continue;                  /* holes: unreachable */
        const unsigned char first = (unsigned char)t->blob[t->voff[id]];
        g_add_next[slot] = g_add_head[first];
        g_add_head[first] = slot;
    }
    free(order);
    g_add_for = t;
}

/* The added-token id starting at text[i], or -1. */
static int32_t match_added(const DSV4Tok *t, const char *text, int i, int n)
{
    const unsigned char first = (unsigned char)text[i];
    for (int32_t s = g_add_head[first]; s >= 0; s = g_add_next[s]) {
        const uint32_t id = t->added[s];
        const int len = (int)t->vlen[id];
        if (len <= n - i && memcmp(text + i, t->blob + t->voff[id],
                                   (size_t)len) == 0)
            return (int32_t)id;
    }
    return -1;
}

int dsv4_tok_encode(const DSV4Tok *t, const char *text, int n,
                    int32_t *ids, int max_ids)
{
    if (n <= 0) return 0;
    build_added_index(t);
    if (g_add_for != t) return encode_plain(t, text, n, ids, max_ids);

    int nids = 0, seg = 0;
    for (int i = 0; i < n && nids < max_ids; ) {
        const int32_t id = match_added(t, text, i, n);
        if (id < 0) { i++; continue; }
        if (i > seg)
            nids += encode_plain(t, text + seg, i - seg, ids + nids,
                                 max_ids - nids);
        if (nids < max_ids) ids[nids++] = id;
        i  += (int)t->vlen[id];
        seg = i;
    }
    if (n > seg && nids < max_ids)
        nids += encode_plain(t, text + seg, n - seg, ids + nids,
                             max_ids - nids);
    return nids;
}

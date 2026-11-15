/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok_lookup.c - vocab lookup, added-token test and decoding. */
#include "dsv4_tok.h"

/* ------------------------------------------------------------ lookup ------ */

/* Linear-probed hash over the vocab blob, built lazily on first use. */
static int32_t *g_bucket;
static uint32_t g_nbucket;
static const DSV4Tok *g_owner;

static uint32_t hash_bytes(const char *s, int n)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

static void build_index(const DSV4Tok *t)
{
    g_nbucket = 1;
    while (g_nbucket < t->n_vocab * 2) g_nbucket <<= 1;
    g_bucket = (int32_t *)malloc((size_t)g_nbucket * sizeof(int32_t));
    for (uint32_t i = 0; i < g_nbucket; i++) g_bucket[i] = -1;
    for (uint32_t i = 0; i < t->n_vocab; i++) {
        if (t->vlen[i] == 0) continue;
        uint32_t h = hash_bytes(t->blob + t->voff[i], t->vlen[i])
                   & (g_nbucket - 1);
        while (g_bucket[h] >= 0) h = (h + 1) & (g_nbucket - 1);
        g_bucket[h] = (int32_t)i;
    }
    g_owner = t;
}

int32_t dsv4_tok_find(const DSV4Tok *t, const char *s, int n)
{
    if (g_owner != t) build_index(t);
    uint32_t h = hash_bytes(s, n) & (g_nbucket - 1);
    while (g_bucket[h] >= 0) {
        const int32_t id = g_bucket[h];
        if (t->vlen[id] == n && memcmp(t->blob + t->voff[id], s, (size_t)n) == 0)
            return id;
        h = (h + 1) & (g_nbucket - 1);
    }
    return -1;
}

/* ------------------------------------------------------------ decoding ---- */

int dsv4_tok_is_added(const DSV4Tok *t, int32_t id)
{
    /* t->added is sorted by id -- see build_added_index. */
    int lo = 0, hi = (int)t->n_added - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        const int32_t v = (int32_t)t->added[mid];
        if (v == id) return 1;
        if (v < id) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

/* Inverse of dsv4_byte_to_cp. The alphabet's largest codepoint is 256 + 67, so
 * a 512-entry table covers it. -1 means "not a byte-level codepoint", which can
 * only happen for a literal token not in the added list; those fall through as
 * ordinary UTF-8 rather than being mangled. */
static int16_t g_inv_cp[512];
static int     g_inv_ready;

static void build_inverse_alphabet(void)
{
    if (g_inv_ready) return;
    for (int i = 0; i < 512; i++) g_inv_cp[i] = -1;
    for (int b = 0; b < 256; b++) {
        const uint32_t cp = dsv4_byte_to_cp((unsigned char)b);
        if (cp < 512) g_inv_cp[cp] = (int16_t)b;
    }
    g_inv_ready = 1;
}

int dsv4_tok_decode(const DSV4Tok *t, int32_t id, unsigned char *out, int cap)
{
    if (!t || !out || cap <= 0 || id < 0 || (uint32_t)id >= t->n_vocab) return 0;
    const int len = (int)t->vlen[id];
    if (len <= 0) return 0;
    const char *p = t->blob + t->voff[id];

    /* Special/added tokens are stored as literal UTF-8. */
    if (dsv4_tok_is_added(t, id)) {
        const int n = len < cap ? len : cap;
        memcpy(out, p, (size_t)n);
        return n;
    }

    build_inverse_alphabet();
    int n = 0, i = 0;
    while (i < len) {
        const uint32_t cp = dsv4_utf8_next(p, len, &i);
        const int b = cp < 512 ? g_inv_cp[cp] : -1;
        if (b >= 0) {
            if (n < cap) out[n++] = (unsigned char)b;
        } else {
            /* Literal codepoint outside the byte alphabet: emit its UTF-8 so
             * an unusual vocab entry degrades to visible text, not garbage. */
            char tmp[4];
            const int k = dsv4_cp_to_utf8(cp, tmp);
            for (int j = 0; j < k && n < cap; j++) out[n++] = (unsigned char)tmp[j];
        }
    }
    return n;
}

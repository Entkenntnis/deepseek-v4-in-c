/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok.h - DeepSeek-V4's byte-level BPE, from the packed tokenizer.
 * The pre-tokenizer stages and their traps are documented in README.md. */
#ifndef DSV4_TOK_H
#define DSV4_TOK_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_unicode.h"

#define DSV4_TOK_MAGIC "DSV4TOK"

/* Chat special tokens, spelled exactly as encoding/encoding_dsv4.py does. The
 * bars are U+FF5C and the underscores U+2581, not ASCII -- typing them by eye
 * is how you get a prompt that tokenises into ordinary text and a model that
 * never realises it is in a conversation. */
#define DSV4_BOS         "<｜begin▁of▁sentence｜>"
#define DSV4_EOS         "<｜end▁of▁sentence｜>"
#define DSV4_USER        "<｜User｜>"
#define DSV4_ASSISTANT   "<｜Assistant｜>"
#define DSV4_THINK_OPEN  "<think>"
#define DSV4_THINK_CLOSE "</think>"

typedef struct {
    uint32_t *voff;      /* [n_vocab] offset into blob                 */
    uint16_t *vlen;      /* [n_vocab] byte length                      */
    uint32_t *ma, *mb;   /* [n_merge] the pair, as vocab ids           */
    uint32_t *mo;        /* [n_merge] what it merges into              */
    uint32_t *added;     /* [n_added] special/added ids                */
    char     *blob;
    uint32_t  n_vocab, n_merge, n_added;
    uint64_t  blob_bytes;

    /* Merges bucketed by LEFT id, in ascending rank order. Built once at load:
     * the merge loop then finds a pair's rank without scanning 127,741 entries
     * per step. */
    int32_t  *pair_head;   /* [n_vocab] index into pair_* or -1 */
    int32_t  *pair_next;   /* [n_merge] */

    void     *raw;         /* one allocation backing everything */
} DSV4Tok;

/* Load a packed tokenizer; all accessors below are read-only afterwards. */
int  dsv4_tok_load(DSV4Tok *t, const char *path);
void dsv4_tok_free(DSV4Tok *t);

/* Vocab lookup over the stored byte-level spelling; -1 when absent. */
int32_t dsv4_tok_find(const DSV4Tok *t, const char *s, int n);

/* decode() writes the ORIGINAL BYTES of one token, not UTF-8 text. A token may
 * end in the middle of a multi-byte character, so a caller streaming text must
 * buffer these bytes and decode incrementally. Returns the number of bytes
 * written (never more than cap), or 0 for an unrepresentable id. */
int  dsv4_tok_is_added(const DSV4Tok *t, int32_t id);
int  dsv4_tok_decode(const DSV4Tok *t, int32_t id, unsigned char *out, int cap);

/* Encode `n` bytes of text into at most max_ids ids. Returns the id count. */
int  dsv4_tok_encode(const DSV4Tok *t, const char *text, int n,
                     int32_t *ids, int max_ids);

/* ---- byte-level alphabet -------------------------------------------------
 * GPT-2's map: printable ASCII and most Latin-1 stay put, everything else
 * moves to U+0100.. so that no token ever contains a raw control byte. */
static inline uint32_t dsv4_byte_to_cp(unsigned char b)
{
    if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE))
        return b;
    /* the 68 remaining bytes, in ascending order, map to 256, 257, ... */
    uint32_t n = 0;
    for (unsigned c = 0; c < b; c++)
        if (!((c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE)))
            n++;
    return 256 + n;
}

static inline int dsv4_cp_to_utf8(uint32_t cp, char *out)
{
    if (cp < 0x80)   { out[0] = (char)cp; return 1; }
    if (cp < 0x800)  { out[0] = (char)(0xC0 | (cp >> 6));
                       out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/* ---- UTF-8 decoding of the INPUT ---------------------------------------- */
static inline uint32_t dsv4_utf8_next(const char *s, int n, int *i)
{
    const unsigned char *p = (const unsigned char *)s + *i;
    const unsigned char c = p[0];
    if (c < 0x80)             { *i += 1; return c; }
    if ((c & 0xE0) == 0xC0 && *i + 1 < n)
                              { *i += 2; return ((uint32_t)(c & 0x1F) << 6)
                                              | (p[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0 && *i + 2 < n)
                              { *i += 3; return ((uint32_t)(c & 0x0F) << 12)
                                              | ((uint32_t)(p[1] & 0x3F) << 6)
                                              | (p[2] & 0x3F); }
    if ((c & 0xF8) == 0xF0 && *i + 3 < n)
                              { *i += 4; return ((uint32_t)(c & 0x07) << 18)
                                              | ((uint32_t)(p[1] & 0x3F) << 12)
                                              | ((uint32_t)(p[2] & 0x3F) << 6)
                                              | (p[3] & 0x3F); }
    *i += 1;                                   /* malformed: consume one byte */
    return 0xFFFD;
}

static inline int dsv4_is_ascii_punct(uint32_t c)
{
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@')
        || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}
static inline int dsv4_is_alpha(uint32_t c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static inline int dsv4_is_space(uint32_t c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r'
        || c == '\v' || c == '\f' || c == 0x85 || c == 0xA0
        || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029
        || c == 0x202F || c == 0x205F || c == 0x3000;
}
/* U+4E00..U+9FA5 ideographs, U+3040..U+309F hiragana, U+30A0..U+30FF katakana */
static inline int dsv4_is_cjk(uint32_t c)
{
    return (c >= 0x4E00 && c <= 0x9FA5)
        || (c >= 0x3040 && c <= 0x309F)
        || (c >= 0x30A0 && c <= 0x30FF);
}

#endif /* DSV4_TOK_H */

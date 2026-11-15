/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st_scan.c - the single-pass JSON scanner for safetensors headers. */
#include <string.h>

#include "dsv4_st.h"
#include "dsv4_st_int.h"

DSV4Dtype dsv4_st_dtype_of(const char *s, size_t n)
{
    /* Spellings taken from the released DeepSeek-V4-Flash headers, not from the
     * safetensors spec: accepting a name no kernel implements is how a wrong
     * model loads quietly. */
    if (n == 2 && !memcmp(s, "U8", 2))      return DSV4_DT_U8;
    if (n == 2 && !memcmp(s, "I8", 2))      return DSV4_DT_I8;       /* packed FP4 */
    if (n == 3 && !memcmp(s, "F16", 3))     return DSV4_DT_F16;
    if (n == 3 && !memcmp(s, "F32", 3))     return DSV4_DT_F32;
    if (n == 3 && !memcmp(s, "I64", 3))     return DSV4_DT_I64;      /* gate.tid2eid */
    if (n == 4 && !memcmp(s, "BF16", 4))    return DSV4_DT_BF16;
    if (n == 7 && !memcmp(s, "F8_E4M3", 7)) return DSV4_DT_F8_E4M3;
    if (n == 7 && !memcmp(s, "F8_E8M0", 7)) return DSV4_DT_F8_E8M0;
    return DSV4_DT_UNKNOWN;
}

void dsv4_st_ws(DSV4Scan *s)
{
    while (s->p < s->end && (unsigned char)*s->p <= ' ') s->p++;
}

int dsv4_st_lit(DSV4Scan *s, char c)
{
    dsv4_st_ws(s);
    if (s->p < s->end && *s->p == c) { s->p++; return 1; }
    return 0;
}

/* Read a JSON string into out (NUL terminated). Tensor names are dotted
 * identifiers so escapes never appear in practice, but a parser that silently
 * mangled one would turn a lookup into a spurious "missing weight". */
int dsv4_st_str(DSV4Scan *s, char *out, size_t cap, size_t *len)
{
    dsv4_st_ws(s);
    if (s->p >= s->end || *s->p != '"') return 0;
    s->p++;
    size_t n = 0;
    while (s->p < s->end && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\') {
            if (s->p >= s->end) return 0;
            char e = *s->p++;
            switch (e) {
            case 'n': c = '\n'; break;  case 't': c = '\t'; break;
            case 'r': c = '\r'; break;  case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'u': {                 /* \uXXXX: keep ASCII, drop the rest */
                if (s->end - s->p < 4) return 0;
                unsigned v = 0;
                for (int i = 0; i < 4; i++) {
                    char h = *s->p++;
                    v = v * 16u + (unsigned)(h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                }
                c = (char)(v < 128 ? v : '?');
                break;
            }
            default: c = e;             /* covers \" \\ \/ */
            }
        }
        if (n + 1 < cap) out[n] = c;
        n++;
    }
    if (s->p >= s->end) return 0;
    s->p++;                              /* closing quote */
    if (n + 1 > cap) return 0;
    out[n] = '\0';
    if (len) *len = n;
    return 1;
}

int dsv4_st_i64(DSV4Scan *s, int64_t *v)
{
    dsv4_st_ws(s);
    int neg = 0;
    if (s->p < s->end && (*s->p == '-' || *s->p == '+')) neg = (*s->p++ == '-');
    if (s->p >= s->end || *s->p < '0' || *s->p > '9') return 0;
    int64_t a = 0;
    while (s->p < s->end && *s->p >= '0' && *s->p <= '9')
        a = a * 10 + (*s->p++ - '0');
    *v = neg ? -a : a;
    return 1;
}

/* Skip any value, tracking nesting and staying out of strings. Used for
 * __metadata__, whose shape is arbitrary. */
int dsv4_st_skip_value(DSV4Scan *s)
{
    dsv4_st_ws(s);
    if (s->p >= s->end) return 0;
    if (*s->p == '"') {
        s->p++;
        while (s->p < s->end && *s->p != '"') { if (*s->p == '\\') s->p++; s->p++; }
        return s->p < s->end ? (s->p++, 1) : 0;
    }
    if (*s->p == '{' || *s->p == '[') {
        int depth = 0;
        do {
            if (s->p >= s->end) return 0;
            char c = *s->p++;
            if (c == '"') {
                while (s->p < s->end && *s->p != '"') {
                    if (*s->p == '\\') s->p++;
                    s->p++;
                }
                if (s->p >= s->end) return 0;
                s->p++;
            } else if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') depth--;
        } while (depth > 0);
        return 1;
    }
    while (s->p < s->end && *s->p != ',' && *s->p != '}' && *s->p != ']') s->p++;
    return 1;
}

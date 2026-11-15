/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok_pretok.c - the four-stage pre-tokenizer. See README.md for the
 * ordered alternatives and the traps. */
#include "dsv4_tok.h"
#include "dsv4_tok_int.h"

/* Length in CODEPOINTS consumed at `i`, and the byte length via *blen.
 * Alternatives are tried in order and the first that matches wins. */
static int match_stage2(const char *s, int n, int i, int *blen)
{
    int j = i, cp_i;
    uint32_t c;

    /* a) [ASCII punct][A-Za-z]+ */
    cp_i = j; c = dsv4_utf8_next(s, n, &cp_i);
    if (dsv4_is_ascii_punct(c)) {
        int k = cp_i, cnt = 0, last = cp_i;
        while (k < n) {
            int t2 = k; uint32_t d = dsv4_utf8_next(s, n, &t2);
            if (!dsv4_is_alpha(d)) break;
            cnt++; k = t2; last = t2;
        }
        if (cnt > 0) { *blen = last - i; return 1; }
    }

    /* b) [^\r\n\p{L}\p{P}\p{S}]? [\p{L}\p{M}]+ */
    {
        int k = i;
        int t2 = k; uint32_t d = dsv4_utf8_next(s, n, &t2);
        int opt_end = k;
        if (k < n && d != '\r' && d != '\n' && !dsv4_is_L(d)
            && !dsv4_is_P(d) && !dsv4_is_S(d)) {
            opt_end = t2;                     /* the optional leading char */
        }
        int m = opt_end, cnt = 0, last = opt_end;
        while (m < n) {
            int t3 = m; uint32_t e = dsv4_utf8_next(s, n, &t3);
            if (!dsv4_is_L(e) && !dsv4_is_M(e)) break;
            cnt++; m = t3; last = t3;
        }
        if (cnt > 0) { *blen = last - i; return 1; }
    }

    /* c)   ?[\p{P}\p{S}]+[\r\n]* */
    {
        int k = i;
        int t2 = k; uint32_t d = dsv4_utf8_next(s, n, &t2);
        int after_sp = k;
        if (d == ' ') after_sp = t2;
        int m = after_sp, cnt = 0, last = after_sp;
        while (m < n) {
            int t3 = m; uint32_t e = dsv4_utf8_next(s, n, &t3);
            if (!dsv4_is_P(e) && !dsv4_is_S(e)) break;
            cnt++; m = t3; last = t3;
        }
        if (cnt > 0) {
            while (last < n) {
                int t3 = last;
                uint32_t e = dsv4_utf8_next(s, n, &t3);
                if (e != '\r' && e != '\n') break;
                last = t3;
            }
            *blen = last - i; return 1;
        }
    }

    /* d)  \s*[\r\n]+ */
    {
        int k = i;
        while (k < n) {
            int t2 = k; uint32_t e = dsv4_utf8_next(s, n, &t2);
            if (!dsv4_is_space(e)) break;
            k = t2;
        }
        /* back off to the first \r\n inside that whitespace run */
        int m = i, nl_start = -1;
        while (m < k) {
            int t2 = m;
            uint32_t e = dsv4_utf8_next(s, n, &t2);
            if (e == '\r' || e == '\n') { nl_start = m; break; }
            m = t2;
        }
        if (nl_start >= 0) {
            int e2 = nl_start;
            while (e2 < n) {
                int t2 = e2;
                uint32_t e = dsv4_utf8_next(s, n, &t2);
                if (e != '\r' && e != '\n') break;
                e2 = t2;
            }
            *blen = e2 - i; return 1;
        }
    }

    /* e) \s+(?!\S) and f) \s+ -- both consume the run; the lookahead only
     * distinguishes them when non-space follows, and then e) gives back its
     * last character so f) can take it with what comes next. */
    {
        int k = i, cnt = 0, last = i, prev = i;
        while (k < n) {
            int t2 = k; uint32_t e = dsv4_utf8_next(s, n, &t2);
            if (!dsv4_is_space(e)) break;
            prev = k; cnt++; k = t2; last = t2;
        }
        if (cnt > 0) {
            if (last < n && cnt > 1) { *blen = prev - i; return 1; } /* (?!\S) */
            *blen = last - i; return 1;
        }
    }

    *blen = 0;
    return 0;
}

/* Stage 0: isolate runs of 1..3 digits. Stage 1: isolate CJK runs. */
int dsv4_tok_split_isolated(const char *s, const DSV4Piece *in, int nin,
                            DSV4Piece *out, int max, int mode)
{
    int nout = 0;
    for (int p = 0; p < nin; p++) {
        const int base = in[p].off, end = base + in[p].len;
        int i = base, run_start = base;
        while (i < end) {
            int t2 = i; uint32_t c = dsv4_utf8_next(s, end, &t2);
            const int hit = (mode == 0) ? dsv4_is_N(c) : dsv4_is_cjk(c);
            if (!hit) { i = t2; continue; }

            if (i > run_start && nout < max)
                out[nout++] = (DSV4Piece){ run_start, i - run_start };

            int k = i, taken = 0;
            while (k < end) {
                int t3 = k; uint32_t d = dsv4_utf8_next(s, end, &t3);
                const int h2 = (mode == 0) ? dsv4_is_N(d) : dsv4_is_cjk(d);
                if (!h2) break;
                /* \p{N}{1,3} is BOUNDED at three; the CJK class is not. */
                if (mode == 0 && taken == 3) break;
                taken++; k = t3;
            }
            if (nout < max) out[nout++] = (DSV4Piece){ i, k - i };
            i = k; run_start = k;
        }
        if (end > run_start && nout < max)
            out[nout++] = (DSV4Piece){ run_start, end - run_start };
    }
    return nout;
}

int dsv4_tok_split_stage2(const char *s, const DSV4Piece *in, int nin,
                          DSV4Piece *out, int max)
{
    int nout = 0;
    for (int p = 0; p < nin; p++) {
        const int base = in[p].off, end = base + in[p].len;
        int i = base, gap = base;      /* gap..i is unmatched text so far */
        while (i < end) {
            int blen = 0;
            if (match_stage2(s, end, i, &blen) && blen > 0) {
                /* Flush the unmatched run as ONE piece before the match. A
                 * Split leaves everything else intact; emitting one codepoint
                 * per unmatched character would shatter a digit run before
                 * BPE sees it ("12" as "1" then "2" can never merge). */
                if (i > gap && nout < max)
                    out[nout++] = (DSV4Piece){ gap, i - gap };
                if (nout < max) out[nout++] = (DSV4Piece){ i, blen };
                i += blen;
                gap = i;
            } else {
                int t2 = i;
                dsv4_utf8_next(s, end, &t2);
                i = t2;                 /* extend the unmatched run */
            }
        }
        if (end > gap && nout < max)
            out[nout++] = (DSV4Piece){ gap, end - gap };
    }
    return nout;
}

/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok_load.c - open and close the packed tokenizer. */
#include "dsv4_tok.h"

int dsv4_tok_load(DSV4Tok *t, const char *path)
{
    memset(t, 0, sizeof *t);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *raw = (char *)malloc((size_t)n);
    if (!raw || fread(raw, 1, (size_t)n, f) != (size_t)n) {
        fclose(f); free(raw); return -1;
    }
    fclose(f);
    t->raw = raw;

    if (memcmp(raw, DSV4_TOK_MAGIC, 7) != 0) {
        fprintf(stderr, "%s: not a packed DeepSeek-V4 tokenizer\n", path);
        return -1;
    }
    const char *p = raw + 8;
    uint32_t ver;
    memcpy(&ver, p, 4);          p += 4;
    memcpy(&t->n_vocab, p, 4);   p += 4;
    memcpy(&t->n_merge, p, 4);   p += 4;
    memcpy(&t->n_added, p, 4);   p += 4;
    memcpy(&t->blob_bytes, p, 8); p += 8;
    if (ver != 1) {
        fprintf(stderr, "%s: version %u, expected 1\n", path, ver);
        return -1;
    }

    t->voff  = (uint32_t *)p; p += (size_t)t->n_vocab * 4;
    t->vlen  = (uint16_t *)p; p += (size_t)t->n_vocab * 2;
    t->ma    = (uint32_t *)p; p += (size_t)t->n_merge * 4;
    t->mb    = (uint32_t *)p; p += (size_t)t->n_merge * 4;
    t->mo    = (uint32_t *)p; p += (size_t)t->n_merge * 4;
    t->added = (uint32_t *)p; p += (size_t)t->n_added * 4;
    t->blob  = (char *)p;

    /* Bucket merges by their LEFT id. Without this the merge loop scans all
     * 127,741 pairs for every adjacent pair at every step, which is quadratic
     * in a way that shows up immediately on real text. */
    t->pair_head = (int32_t *)malloc((size_t)t->n_vocab * sizeof(int32_t));
    t->pair_next = (int32_t *)malloc((size_t)t->n_merge * sizeof(int32_t));
    if (!t->pair_head || !t->pair_next) return -1;
    for (uint32_t i = 0; i < t->n_vocab; i++) t->pair_head[i] = -1;
    /* Walk backwards so each bucket is in ascending rank order, and the first
     * match found is therefore the best one. */
    for (int32_t i = (int32_t)t->n_merge - 1; i >= 0; i--) {
        t->pair_next[i] = t->pair_head[t->ma[i]];
        t->pair_head[t->ma[i]] = i;
    }
    return 0;
}

void dsv4_tok_free(DSV4Tok *t)
{
    if (!t) return;
    free(t->pair_head); free(t->pair_next); free(t->raw);
    memset(t, 0, sizeof *t);
}

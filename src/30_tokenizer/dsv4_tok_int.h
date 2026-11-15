/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok_int.h - seams between the tokenizer files. Not public. */
#ifndef DSV4_TOK_INT_H
#define DSV4_TOK_INT_H

typedef struct { int off, len; } DSV4Piece;

/* Stage 0 (mode 0, digits) / stage 1 (mode 1, CJK) isolating split. */
int dsv4_tok_split_isolated(const char *s, const DSV4Piece *in, int nin,
                            DSV4Piece *out, int max, int mode);

/* Stage 2: the ordered alternation. */
int dsv4_tok_split_stage2(const char *s, const DSV4Piece *in, int nin,
                          DSV4Piece *out, int max);

#endif /* DSV4_TOK_INT_H */

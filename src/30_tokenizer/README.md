# src/30_tokenizer — byte-level BPE

## The pre-tokenizer is the hard part, not the merges

The merge loop is mechanical: repeatedly join the adjacent pair with the lowest
rank. Everything that can go subtly wrong lives in how the input is cut up
first, because a piece boundary in the wrong place changes the ids without
producing anything that looks like an error.

`tokenizer.json`'s pre_tokenizer is a Sequence of four stages, applied in order,
each operating on the pieces the previous one produced:

```
0.  Split \p{N}{1,3}                     Isolated
1.  Split [CJK|hiragana|katakana]+       Isolated
2.  Split <the big alternation>          Isolated
3.  ByteLevel(add_prefix_space=false, use_regex=false)
```

"Isolated" means the match becomes its own piece and the text around it stays.
Splits do NOT consume: a stage that finds nothing leaves its input untouched.

## Three things that are easy to get wrong

1. **`\p{N}{1,3}` is bounded.** It isolates runs of one to THREE digits, so
   "1234" becomes "123" + "4", not one piece and not four.
2. **The CJK class is specific.** U+4E00..U+9FA5 plus hiragana plus katakana,
   NOT "anything non-Latin". Cyrillic and Arabic go through stage 2 and are
   handled by the letter-run alternative instead.
3. **Stage 2's alternatives are ordered and the first match wins.**

   ```
   a) [ASCII punct][A-Za-z]+          "#define" is ONE piece
   b) [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+   optional leading non-letter,
                                      then a letter run -- this is what puts
                                      the SPACE on " the"
   c)  ?[\p{P}\p{S}]+[\r\n]*          punctuation runs
   d) \s*[\r\n]+                      newline runs
   e) \s+(?!\S)                      trailing whitespace
   f) \s+                             any other whitespace
   ```

   Reordering b) and c) alone silently moves every leading space.

Splitting also has to flush the UNMATCHED run as one piece before a match.
Emitting one codepoint per unmatched character shatters a digit run before BPE
sees it: "12" arrives as "1" then "2" and can never merge.

## Byte level, not UTF-8

Each input BYTE maps to a printable codepoint before matching against the vocab
(`dsv4_byte_to_cp`), so the token for " the" is spelled "Gthe" with U+0120. The
packed vocab stores exactly those spellings, which is why lookup never decodes
UTF-8.

Decoding undoes that map. `dsv4_tok_decode()` writes the ORIGINAL BYTES of one
token, not UTF-8 text: a token may end in the middle of a multi-byte character,
so a caller streaming text must buffer these bytes and decode incrementally.
Added/special tokens are the exception — their stored spelling is literal text,
so they are copied through unchanged.

## Added tokens

Added tokens must be matched BEFORE the regex pre-tokenizer ever sees them.
They are not BPE material: `<｜User｜>` is one id, atomically, and the merge
table contains nothing that could rebuild it. Feeding it through the normal
path shatters it into ordinary text; measured, the chat prompt came out as 37
ids where the reference gives 17. The model then never sees a turn boundary and
still generates fluent text, just not as an assistant answering a question.

The scan is indexed by first byte, so ordinary prose costs one comparison per
character rather than 1,283. Longest match wins within a bucket. Only added ids
that exist in THIS vocab are indexed: a reduced tokenizer can carry added ids
past its own `n_vocab`, and indexing `vlen[]` with those reads off the end.

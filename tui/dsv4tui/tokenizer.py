# SPDX-License-Identifier: Apache-2.0
"""Python side of the DeepSeek-V4 byte-level BPE.

The C engine no longer sees text at all: it accepts and emits token ids, and
this module is the only place that turns UTF-8 into ids and back. It reads the
same packed DSV4TOK file that ``tools/pack_tokenizer.py`` writes, so the pack
format stays useful and startup costs one binary read rather than parsing
tokenizer.json.

The pre-tokenizer is a direct port of the four-stage splitter that used to live
in ``src/30_tokenizer``. Keeping the port literal matters: prompt token counts
and the e2e/prefill golden token ids depend on it.
"""

from __future__ import annotations

import struct
import unicodedata
from pathlib import Path

from .defaults import (ASSISTANT, BOS, EOS, THINK_CLOSE, THINK_OPEN, USER)

_TOK_MAGIC = b"DSV4TOK\0"
_MAX_PIECES = 65536
_MAX_SYMBOLS = 4096


# ---------------------------------------------------------------- categories


def _is_L(cp: int) -> bool:
    if cp < 0x80:
        return (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A)
    return unicodedata.category(chr(cp)).startswith("L")


def _is_N(cp: int) -> bool:
    if cp < 0x80:
        return 0x30 <= cp <= 0x39
    return unicodedata.category(chr(cp)).startswith("N")


def _is_P(cp: int) -> bool:
    if cp < 0x80:
        return (0x21 <= cp <= 0x2F) or (0x3A <= cp <= 0x40) \
            or (0x5B <= cp <= 0x60) or (0x7B <= cp <= 0x7E)
    return unicodedata.category(chr(cp)).startswith("P")


def _is_S(cp: int) -> bool:
    if cp < 0x80:
        return (0x24 <= cp <= 0x2B) or (0x3C <= cp <= 0x3E) \
            or cp in (0x5E, 0x60, 0x7C, 0x7E)
    return unicodedata.category(chr(cp)).startswith("S")


def _is_M(cp: int) -> bool:
    if cp < 0x80:
        return False
    return unicodedata.category(chr(cp)).startswith("M")


def _is_ascii_punct(cp: int) -> bool:
    return (0x21 <= cp <= 0x2F) or (0x3A <= cp <= 0x40) \
        or (0x5B <= cp <= 0x60) or (0x7B <= cp <= 0x7E)


def _is_alpha(cp: int) -> bool:
    return (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A)


def _is_space(cp: int) -> bool:
    return cp in (0x20, 0x09, 0x0A, 0x0D, 0x0B, 0x0C, 0x85, 0xA0) \
        or (0x2000 <= cp <= 0x200A) or cp in (0x2028, 0x2029, 0x202F,
                                              0x205F, 0x3000)


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FA5) or (0x3040 <= cp <= 0x309F) \
        or (0x30A0 <= cp <= 0x30FF)


def _utf8_next(s: bytes, n: int, i: int) -> tuple[int, int]:
    c = s[i]
    if c < 0x80:
        return c, i + 1
    if (c & 0xE0) == 0xC0 and i + 1 < n:
        return ((c & 0x1F) << 6) | (s[i + 1] & 0x3F), i + 2
    if (c & 0xF0) == 0xE0 and i + 2 < n:
        return (((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6)
                | (s[i + 2] & 0x3F)), i + 3
    if (c & 0xF8) == 0xF0 and i + 3 < n:
        return (((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12)
                | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F)), i + 4
    return 0xFFFD, i + 1


def _byte_to_cp(b: int) -> int:
    if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or b >= 0xAE:
        return b
    n = 0
    for c in range(b):
        if not ((0x21 <= c <= 0x7E) or (0xA1 <= c <= 0xAC) or c >= 0xAE):
            n += 1
    return 256 + n


class Tokenizer:
    """Packed DeepSeek-V4 tokenizer: byte-level BPE plus added tokens."""

    def __init__(self, path: str | Path) -> None:
        raw = Path(path).read_bytes()
        if len(raw) < 32 or raw[:8] != _TOK_MAGIC:
            raise ValueError(f"{path}: not a packed DeepSeek-V4 tokenizer")
        ver, n_vocab, n_merge, n_added = struct.unpack_from("<IIII", raw, 8)
        if ver != 1:
            raise ValueError(f"{path}: version {ver}, expected 1")
        blob_bytes = struct.unpack_from("<Q", raw, 24)[0]

        p = 32
        voff = struct.unpack_from(f"<{n_vocab}I", raw, p)
        p += 4 * n_vocab
        vlen = struct.unpack_from(f"<{n_vocab}H", raw, p)
        p += 2 * n_vocab
        ma = struct.unpack_from(f"<{n_merge}I", raw, p)
        p += 4 * n_merge
        mb = struct.unpack_from(f"<{n_merge}I", raw, p)
        p += 4 * n_merge
        mo = struct.unpack_from(f"<{n_merge}I", raw, p)
        p += 4 * n_merge
        added = struct.unpack_from(f"<{n_added}I", raw, p)
        p += 4 * n_added
        blob = raw[p:p + blob_bytes]

        self.path = str(path)
        self.n_vocab = n_vocab
        self._blob = blob
        self._voff = voff
        self._vlen = vlen
        self._added = set(added)

        # Vocab lookup over the stored byte-level spelling. Duplicate spellings
        # keep the first id, matching the packer's merge index.
        self._vocab: dict[str, int] = {}
        for i in range(n_vocab):
            if vlen[i] == 0:
                continue
            s = blob[voff[i]:voff[i] + vlen[i]].decode("utf-8", "replace")
            if s not in self._vocab:
                self._vocab[s] = i

        # Pair ranks. The first occurrence of a pair is its rank, which is what
        # the C tokenizer's ascending-rank buckets selected.
        self._pairs: dict[tuple[int, int], tuple[int, int]] = {}
        for r in range(n_merge):
            key = (ma[r], mb[r])
            if key not in self._pairs:
                self._pairs[key] = (r, mo[r])

        # Added tokens bucketed by first byte, longest first.
        self._added_bucket: list[list[int]] = [[] for _ in range(256)]
        for i in added:
            if i < n_vocab and vlen[i] > 0:
                first = blob[voff[i]]
                self._added_bucket[first].append(i)
        for bucket in self._added_bucket:
            bucket.sort(key=lambda i: (-vlen[i], i))

        # For decoding: byte -> byte-level codepoint, and the inverse.
        self._byte_cp = [_byte_to_cp(b) for b in range(256)]
        self._byte_char = [chr(cp) for cp in self._byte_cp]
        self._inv_cp = {cp: b for b, cp in enumerate(self._byte_cp)}

        self.bos_id = self._special(BOS)
        self.eos_id = self._special(EOS)
        self.user_id = self._special(USER)
        self.assistant_id = self._special(ASSISTANT)
        self.think_open_id = self._special(THINK_OPEN)
        self.think_close_id = self._special(THINK_CLOSE)

    # ---------------------------------------------------------- special ids

    def _special(self, text: str) -> int:
        if text not in self._vocab:
            raise ValueError(f"{self.path}: required token {text!r} is absent")
        return self._vocab[text]

    # ------------------------------------------------------------- decoding

    def token_bytes(self, token_id: int) -> bytes:
        if token_id < 0 or token_id >= self.n_vocab:
            return b""
        off = self._voff[token_id]
        length = self._vlen[token_id]
        if length <= 0:
            return b""
        raw = self._blob[off:off + length]
        if token_id in self._added:
            return raw

        out = bytearray()
        i = 0
        while i < length:
            cp, i = _utf8_next(raw, length, i)
            b = self._inv_cp.get(cp)
            if b is not None:
                out.append(b)
            else:
                out.extend(chr(cp).encode("utf-8", "replace"))
        return bytes(out)

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self.token_bytes(int(i)) for i in ids)

    # ------------------------------------------------------------- encoding

    def _match_added(self, s: bytes, i: int, n: int) -> int:
        for token_id in self._added_bucket[s[i]]:
            length = self._vlen[token_id]
            if length <= n - i and s.startswith(
                    self._blob[self._voff[token_id]:
                                self._voff[token_id] + length], i):
                return token_id
        return -1

    def _match_stage2(self, s: bytes, n: int, i: int) -> int:
        cp, cp_i = _utf8_next(s, n, i)
        if _is_ascii_punct(cp):
            k, cnt, last = cp_i, 0, cp_i
            while k < n:
                d, t2 = _utf8_next(s, n, k)
                if not _is_alpha(d):
                    break
                cnt += 1
                k = last = t2
            if cnt > 0:
                return last - i

        k = i
        d, t2 = _utf8_next(s, n, k)
        opt_end = k
        if k < n and d not in (0x0D, 0x0A) and not _is_L(d) \
                and not _is_P(d) and not _is_S(d):
            opt_end = t2
        m, cnt, last = opt_end, 0, opt_end
        while m < n:
            e, t3 = _utf8_next(s, n, m)
            if not _is_L(e) and not _is_M(e):
                break
            cnt += 1
            m = last = t3
        if cnt > 0:
            return last - i

        k = i
        d, t2 = _utf8_next(s, n, k)
        after_sp = t2 if d == 0x20 else k
        m, cnt, last = after_sp, 0, after_sp
        while m < n:
            e, t3 = _utf8_next(s, n, m)
            if not _is_P(e) and not _is_S(e):
                break
            cnt += 1
            m = last = t3
        if cnt > 0:
            while last < n:
                e, t3 = _utf8_next(s, n, last)
                if e not in (0x0D, 0x0A):
                    break
                last = t3
            return last - i

        k = i
        while k < n:
            e, t2 = _utf8_next(s, n, k)
            if not _is_space(e):
                break
            k = t2
        m, nl_start = i, -1
        while m < k:
            e, t2 = _utf8_next(s, n, m)
            if e in (0x0D, 0x0A):
                nl_start = m
                break
            m = t2
        if nl_start >= 0:
            e2 = nl_start
            while e2 < n:
                e, t2 = _utf8_next(s, n, e2)
                if e not in (0x0D, 0x0A):
                    break
                e2 = t2
            return e2 - i

        k, cnt, last, prev = i, 0, i, i
        while k < n:
            e, t2 = _utf8_next(s, n, k)
            if not _is_space(e):
                break
            prev, cnt, k, last = k, cnt + 1, t2, t2
        if cnt > 0:
            if last < n and cnt > 1:
                return prev - i
            return last - i
        return 0

    def _split_isolated(self, s: bytes,
                        pieces: list[tuple[int, int]], mode: int
                        ) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for off, length in pieces:
            base, end = off, off + length
            i = run_start = base
            while i < end:
                c, t2 = _utf8_next(s, end, i)
                hit = _is_N(c) if mode == 0 else _is_cjk(c)
                if not hit:
                    i = t2
                    continue
                if i > run_start and len(out) < _MAX_PIECES:
                    out.append((run_start, i - run_start))
                k, taken = i, 0
                while k < end:
                    d, t3 = _utf8_next(s, end, k)
                    h2 = _is_N(d) if mode == 0 else _is_cjk(d)
                    if not h2:
                        break
                    if mode == 0 and taken == 3:
                        break
                    taken += 1
                    k = t3
                if len(out) < _MAX_PIECES:
                    out.append((i, k - i))
                i = run_start = k
            if end > run_start and len(out) < _MAX_PIECES:
                out.append((run_start, end - run_start))
        return out

    def _split_stage2(self, s: bytes,
                      pieces: list[tuple[int, int]]
                      ) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for off, length in pieces:
            base, end = off, off + length
            i = gap = base
            while i < end:
                blen = self._match_stage2(s, end, i)
                if blen > 0:
                    if i > gap and len(out) < _MAX_PIECES:
                        out.append((gap, i - gap))
                    if len(out) < _MAX_PIECES:
                        out.append((i, blen))
                    i += blen
                    gap = i
                else:
                    _, t2 = _utf8_next(s, end, i)
                    i = t2
            if end > gap and len(out) < _MAX_PIECES:
                out.append((gap, end - gap))
        return out

    def _encode_plain(self, s: bytes, ids: list[int]) -> None:
        whole = [(0, len(s))]
        a = self._split_isolated(s, whole, 0)
        b = self._split_isolated(s, a, 1)
        pieces = self._split_stage2(s, b)

        for off, length in pieces:
            symbols: list[int] = []
            for i in range(off, off + length):
                if len(symbols) >= _MAX_SYMBOLS:
                    break
                symbols.append(self._vocab.get(self._byte_char[s[i]], -1))

            while True:
                best_rank = None
                best_at = -1
                best_out = 0
                for i in range(len(symbols) - 1):
                    left, right = symbols[i], symbols[i + 1]
                    if left < 0 or right < 0:
                        continue
                    pair = self._pairs.get((left, right))
                    if pair is not None and (best_rank is None
                                             or pair[0] < best_rank):
                        best_rank, best_at, best_out = pair[0], i, pair[1]
                if best_rank is None:
                    break
                symbols[best_at] = best_out
                del symbols[best_at + 1]

            for token_id in symbols:
                if token_id >= 0:
                    ids.append(token_id)

    def encode(self, text: str) -> list[int]:
        s = text.encode("utf-8")
        n = len(s)
        if n == 0:
            return []

        ids: list[int] = []
        i = seg = 0
        while i < n:
            token_id = self._match_added(s, i, n)
            if token_id < 0:
                i += 1
                continue
            if i > seg:
                self._encode_plain(s[seg:i], ids)
            ids.append(token_id)
            i += self._vlen[token_id]
            seg = i
        if seg < n:
            self._encode_plain(s[seg:], ids)
        return ids

    # ------------------------------------------------------------ turn framing

    def turn_tokens(self, prompt: str, system: str | None = None,
                    think: bool = False, first: bool = False) -> list[int]:
        """The exact chat framing the C engine used to build itself."""
        parts: list[str] = []
        if first:
            parts.append(BOS)
        if system is not None:
            parts.append(system)
            parts.append("\n")
        parts.append(USER)
        parts.append(prompt)
        parts.append(ASSISTANT)
        parts.append(THINK_OPEN if think else THINK_CLOSE)
        return self.encode("".join(parts))

    def prefix_tokens(self, system: str) -> list[int]:
        """BOS + system + newline, used by the /tools KV prefix cache."""
        return self.encode(BOS + system + "\n")

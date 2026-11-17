# SPDX-License-Identifier: Apache-2.0
"""Curses colour pairs, attrs and the clipped write helper."""

from __future__ import annotations

import curses
from typing import Any

from .markdown import Span
from .xterm import _nearest_basic, _rgb_to_xterm, _xterm_rgb


class CursesMixin:
    def attr(self, name: str) -> int:
        return self.pairs.get(name, 0)

    def _color_index(self, spec: Any) -> int:
        """RGB/xterm-256 -> an index this terminal can actually init_pair()."""
        if spec is None:
            return -1
        if isinstance(spec, tuple):
            rgb = (int(spec[0]), int(spec[1]), int(spec[2]))
        else:
            idx = int(spec)
            if curses.COLORS >= 256 and 0 <= idx < curses.COLORS:
                return idx
            rgb = _xterm_rgb(idx)
        key = (rgb, curses.COLORS)
        if key in self._color_cache:
            return self._color_cache[key]
        if curses.COLORS >= 256:
            out = _rgb_to_xterm(rgb)
        elif curses.COLORS >= 16:
            out = _nearest_basic(rgb, 16)
        elif curses.COLORS >= 8:
            out = _nearest_basic(rgb, 8)
        else:
            out = -1
        self._color_cache[key] = out
        return out

    def pair_for(self, fg: Any, bg: Any) -> int:
        """Lazily allocate a curses colour pair for one Markdown span."""
        if not curses.has_colors():
            return 0
        f = self._color_index(fg)
        b = self._color_index(bg)
        if f < 0 and b < 0:
            return 0
        key = (f, b)
        if key in self._pair_cache:
            return self._pair_cache[key]
        try:
            limit = curses.COLOR_PAIRS
        except Exception:
            limit = 64
        if self._next_pair >= limit:
            return 0
        pid = self._next_pair
        self._next_pair += 1
        try:
            curses.init_pair(pid, f, b)
        except curses.error:
            return 0
        attr = curses.color_pair(pid)
        self._pair_cache[key] = attr
        return attr

    def span_attr(self, span: Span) -> int:
        if span.named:
            return self.attr(span.named)
        attr = self.pair_for(span.fg, span.bg)
        if span.bold:
            attr |= curses.A_BOLD
        if span.italic:
            attr |= getattr(curses, "A_ITALIC", 0)
        if span.underline:
            attr |= curses.A_UNDERLINE
        if span.dim:
            attr |= curses.A_DIM
        if span.reverse:
            attr |= curses.A_REVERSE
        return attr

    def _add(self, scr: "curses.window", row: int, col: int, text: str,
             width: int, attr: int = 0) -> None:
        if not text or width <= 0 or row < 0 or col < 0:
            return
        try:
            scr.addnstr(row, col, text, max(1, width), attr)
        except curses.error:
            pass

    def init_colors(self) -> None:
        self._pair_cache = {}
        self._color_cache = {}
        self._next_pair = 1
        self.pairs: dict[str, int] = {
            "default": 0, "value": 0, "dim": curses.A_DIM,
            "title": curses.A_BOLD, "status": 0, "active": curses.A_BOLD,
            "user": 0, "assistant": 0, "system": 0, "error": curses.A_BOLD,
            "tool": 0, "section": curses.A_BOLD,
            "input": 0, "prefill": 0, "decode": 0,
            "muted": curses.A_DIM, "cancel": curses.A_BOLD,
        }
        if not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
            specs = {
                "user": (curses.COLOR_CYAN, -1),
                "assistant": (curses.COLOR_GREEN, -1),
                "tool": (curses.COLOR_BLUE, -1),
                "system": (curses.COLOR_YELLOW, -1),
                "error": (curses.COLOR_RED, -1),
                "section": (curses.COLOR_BLUE, -1),
                "active": (curses.COLOR_MAGENTA, -1),
                "status": (curses.COLOR_GREEN, -1),
                "title": (curses.COLOR_WHITE, curses.COLOR_BLUE),
                "value": (-1, -1),
                "input": (curses.COLOR_WHITE, -1),
                "prefill": (curses.COLOR_CYAN, -1),
                "decode": (curses.COLOR_GREEN, -1),
                "cancel": (curses.COLOR_RED, -1),
            }
            if curses.COLORS >= 256:
                specs["muted"] = (240, -1)   # dark grey, quieter than A_DIM
                specs["cancel"] = (88, -1)   # dark red
                specs["tool"] = (141, -1)    # soft purple
            for i, (name, (fg, bg)) in enumerate(specs.items(), start=1):
                curses.init_pair(i, fg, bg)
                extra = 0
                if name in ("title", "active", "error", "section"):
                    extra = curses.A_BOLD
                self.pairs[name] = curses.color_pair(i) | extra
            self._next_pair = len(specs) + 1
        except curses.error:
            pass

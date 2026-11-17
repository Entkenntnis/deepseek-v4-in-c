# SPDX-License-Identifier: Apache-2.0
"""Paint the stats pane, footer and input line."""

from __future__ import annotations

import curses
import time


class DrawMixin:
    def _draw_stats(self, scr: "curses.window", top: int, bottom: int,
                    col: int, width: int) -> None:
        if width <= 4:
            return
        # The decode graph, MEMORY and CPU are a 1 Hz instrument: repainting
        # them on every layer event only makes them jitter. The PREFILL
        # section is the exception: it is prepended fresh on every render, so
        # its bar, percentage and layer/token counters follow the prefill
        # events while the rate figures stay on the stats cadence.
        height = bottom - top + 1
        now = time.monotonic()
        if (self.stats_force or not self.stats_cache
                or width != self.stats_cache_width
                or height != self.stats_cache_height
                or now - self.stats_cache_ts >= self.stats_interval):
            self.update_cpu()
            self.mem_available_gb = self._read_mem_available()
            self.stats_cache = self.stats_lines(width, height)
            self.stats_cache_ts = now
            self.stats_cache_width = width
            self.stats_cache_height = height
            self.stats_force = False
        lines = self.stats_cache
        if self.prefill_active or self.prefill_tokens > 0:
            lines = self.prefill_lines(width) + lines
        for i, segments in enumerate(lines):
            row = top + i
            if row > bottom:
                break
            x = 0
            for text, color in segments:
                if x >= width:
                    break
                text = text[:width - x]
                if text:
                    self._add(scr, row, col + x, text, width - x,
                              self.attr(color))
                    x += len(text)

    def _draw_footer(self, scr: "curses.window", h: int, w: int) -> None:
        self.cancel_rect = None
        if self.generating:
            hint = "Mausrad/PgUp/PgDn scroll  ·  Ctrl-C quit  ·  "
            label = "[cancelling]" if self.cancel_pending else "[cancel]"
            self._add(scr, h - 2, 0, hint, w - 1, self.attr("dim"))
            self._add(scr, h - 2, len(hint), label, w - len(hint) - 1,
                      self.attr("cancel"))
            self.cancel_rect = (h - 2, len(hint), len(label))
        else:
            hint = "Enter send  ·  /help commands  ·  Mausrad/PgUp/PgDn scroll  ·  Ctrl-C quit"
            self._add(scr, h - 2, 0, hint, w - 1, self.attr("dim"))

    def _draw_input(self, scr: "curses.window", h: int, w: int) -> None:
        prompt = "› "
        room = max(1, w - len(prompt) - 1)
        start = 0
        if self.input_cursor > room:
            start = self.input_cursor - room
        shown = self.input[start:start + room]
        color = "input" if not self.generating else "dim"
        self._add(scr, h - 1, 0, prompt + shown, w - 1, self.attr(color))
        cx = len(prompt) + (self.input_cursor - start)
        try:
            # The input text never reaches the last column (see `room`), so the
            # caret may sit there: clamping to w-2 would leave it on top of the
            # last visible character instead of after it.
            scr.move(h - 1, min(w - 1, cx))
            curses.curs_set(1)
        except curses.error:
            pass

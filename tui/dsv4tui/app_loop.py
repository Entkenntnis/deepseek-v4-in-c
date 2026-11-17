# SPDX-License-Identifier: Apache-2.0
"""The curses main loop."""

from __future__ import annotations

import curses
import time


class LoopMixin:
    def run(self, stdscr: "curses.window") -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        stdscr.timeout(35)
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
            curses.mouseinterval(0)
        except curses.error:
            pass
        self.init_colors()
        self.dirty = True
        self.render(stdscr)

        while self.running:
            self.drain()
            now = time.monotonic()
            active = self.status in ("starting", "waiting") or self.generating
            if (self.dirty or (active and now - self.last_render > 0.12)
                    or now - self.last_render > 0.8):
                self.render(stdscr)

            try:
                ch = stdscr.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:
                self.interrupt()
                continue
            if ch == curses.KEY_RESIZE:
                stdscr.erase()
                self.dirty = True
                continue
            if ch == curses.KEY_MOUSE:
                self.handle_mouse()
                continue
            self.key(ch)
            if self.running:
                self.render(stdscr)

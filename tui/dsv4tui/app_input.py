# SPDX-License-Identifier: Apache-2.0
"""Keyboard and mouse routing for the input line and transcript."""

from __future__ import annotations

import curses
from typing import Any


class InputMixin:
    def key(self, ch: Any) -> None:
        if isinstance(ch, int):
            self.key_code(ch)
            return
        if ch == "\x03":                        # Ctrl-C
            self.running = False
        elif ch == "\x04":                      # Ctrl-D
            if not self.input:
                self.running = False
        elif ch == "\x15":                      # Ctrl-U
            self.input = ""
            self.input_cursor = 0
        elif ch in ("\n", "\r"):
            self.submit()
        elif ch == "\x7f" or ch == "\b":
            if self.input_cursor > 0:
                self.input = (self.input[:self.input_cursor - 1]
                              + self.input[self.input_cursor:])
                self.input_cursor -= 1
        elif ch == "\x1b":                      # ESC
            self.input = ""
            self.input_cursor = 0
        elif ch >= " ":
            if self.input_cursor < len(self.input):
                self.input = (self.input[:self.input_cursor] + ch
                              + self.input[self.input_cursor:])
            else:
                self.input += ch
            self.input_cursor += 1
        self.dirty = True

    def handle_mouse(self) -> None:
        try:
            _, mx, my, _, bstate = curses.getmouse()
        except curses.error:
            return

        # Wheel: BUTTON4 is up, BUTTON5 is down. Some curses builds only define
        # the PRESSED bits, so look them up defensively. RELEASED is ignored on
        # purpose: terminals that report it would otherwise scroll twice.
        up = bstate & (getattr(curses, "BUTTON4_PRESSED", 0)
                       | getattr(curses, "BUTTON4_CLICKED", 0))
        down = bstate & (getattr(curses, "BUTTON5_PRESSED", 0)
                         | getattr(curses, "BUTTON5_CLICKED", 0))
        rect = self.chat_rect
        over_chat = (rect is not None and rect[0] <= my <= rect[1]
                     and 0 <= mx < rect[2])
        if (up or down) and over_chat:
            if up:
                self.scroll = min(self.max_scroll, self.scroll + 3)
            else:
                self.scroll = max(0, self.scroll - 3)
            self.dirty = True
            return

        buttons = (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED
                   | curses.BUTTON1_RELEASED)
        if not (bstate & buttons):
            return
        cancel = self.cancel_rect
        if cancel and my == cancel[0] and cancel[1] <= mx < cancel[1] + cancel[2]:
            self.cancel()
            self.dirty = True

    def key_code(self, code: int) -> None:
        if code in (curses.KEY_ENTER, 10, 13):
            self.submit()
        elif code in (curses.KEY_BACKSPACE, 127, 8):
            if self.input_cursor > 0:
                self.input = (self.input[:self.input_cursor - 1]
                              + self.input[self.input_cursor:])
                self.input_cursor -= 1
        elif code == curses.KEY_DC:
            if self.input_cursor < len(self.input):
                self.input = (self.input[:self.input_cursor]
                              + self.input[self.input_cursor + 1:])
        elif code == curses.KEY_LEFT:
            self.input_cursor = max(0, self.input_cursor - 1)
        elif code == curses.KEY_RIGHT:
            self.input_cursor = min(len(self.input), self.input_cursor + 1)
        elif code == curses.KEY_HOME:
            self.input_cursor = 0
        elif code == curses.KEY_END:
            self.input_cursor = len(self.input)
        elif code == curses.KEY_PPAGE:
            self.scroll += 12
        elif code == curses.KEY_NPAGE:
            self.scroll = max(0, self.scroll - 12)
        elif code == curses.KEY_UP:
            self.scroll += 1
        elif code == curses.KEY_DOWN:
            self.scroll = max(0, self.scroll - 1)
        elif code == curses.KEY_RESIZE:
            pass
        self.dirty = True

# SPDX-License-Identifier: Apache-2.0
"""Top-level render pass: header, status line and pane geometry."""

from __future__ import annotations

import time
from pathlib import Path

from .defaults import DEFAULT_MODEL
from .dsml import WEB_SEARCH_NAME


class ViewMixin:
    def render(self, stdscr: "curses.window") -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 14 or w < 62:
            self._add(stdscr, 0, 0, "terminal too small; need at least 62x14",
                      w - 1, self.attr("error"))
            stdscr.refresh()
            return

        self.spinner = (self.spinner + 1) % 4
        panel_w = min(52, max(36, int(w * 0.37)))
        sep_x = w - panel_w - 1
        chat_w = sep_x - 1
        body_top, body_bottom = 2, h - 3
        self.chat_rect = (body_top, body_bottom, chat_w)

        self._draw_header(stdscr, w)
        self._draw_status(stdscr, w, body_top - 1)
        self._draw_chat(stdscr, body_top, body_bottom, chat_w)
        self._draw_separator(stdscr, body_top, body_bottom, sep_x)
        self._draw_stats(stdscr, body_top, body_bottom, sep_x + 2, panel_w - 1)
        self._draw_footer(stdscr, h, w)
        self._draw_input(stdscr, h, w)

        stdscr.refresh()
        self.last_render = time.monotonic()
        self.dirty = False

    def _draw_header(self, scr: "curses.window", w: int) -> None:
        model = Path(DEFAULT_MODEL).name if not self.demo else "demo"
        brand = f" DSV4  {model}"
        status = self.status
        if self.demo:
            brand += "  [demo]"
        progress = ""
        if self.tool_running:
            spin = "|/-\\"[self.spinner]
            tool = ("web search"
                    if self.pending_tool_name == WEB_SEARCH_NAME else "python")
            progress = (f"  {spin} {tool}  round {self.chain_round}"
                        f"  ctx {self.context_used}/{self.context_cap}"
                        f"  Ctrl-C cancel")
        elif status == "generating":
            n_layers = int((self.ready.get("config") or {}).get("layers") or 0) or 43
            if self.phase == "prompt":
                phase = f"prefill {self.current_layer}/{n_layers}"
            elif self.phase == "resume":
                phase = "closing previous turn"
            else:
                phase = f"layer {self.current_layer}/{n_layers}"
            progress = (f"  generating {self.generation}/{self.gen_max} tok"
                        f"  {phase}"
                        f"  ctx {self.context_used}/{self.context_cap}")
        self._add(scr, 0, 0, brand, w - 1, self.attr("title"))
        if progress:
            self._add(scr, 0, len(brand), progress, max(1, w - len(brand) - 1),
                      self.attr("muted"))
        right = "" if status in ("generating", "ready") else f"{status} "
        if right:
            self._add(scr, 0, max(0, w - len(right) - 1), right, len(right),
                      self.attr("value"))

    def _draw_status(self, scr: "curses.window", w: int, row: int) -> None:
        if self.status == "generating":
            # Progress and context live in the title bar while a turn runs.
            return
        elif self.status == "waiting":
            text = f"... waiting for engine  (prompt {self.prompt_tokens} tokens)"
            color = "active"
        elif self.status == "starting":
            spin = "|/-\\"[self.spinner]
            if self.demo:
                text = f"{spin} starting demo backend"
            else:
                text = f"{spin} loading checkpoint (embed/head read)"
            color = "active"
        elif self.status == "failed":
            text = "engine failed; see the error in the transcript"
            color = "error"
        else:
            text = (f"context {self.context_used}/{self.context_cap}  "
                    f"gen limit {self.gen_limit}  "
                    f"tools {'on' if self.tool_enabled else 'off'}")
            color = "status"
        self._add(scr, row, 0, text, w - 1, self.attr(color))

# SPDX-License-Identifier: Apache-2.0
"""Transcript wrapping, Markdown layout and chat-pane drawing."""

from __future__ import annotations

from .markdown import RICH_AVAILABLE, Span
from .markdown_render import render_markdown


class ChatMixin:
    def _wrap(self, text: str, width: int) -> list[str]:
        width = max(8, width)
        out: list[str] = []
        for raw in text.replace("\t", "    ").split("\n"):
            raw = "".join(c if c.isprintable() else "?" for c in raw)
            if not raw:
                out.append("")
                continue
            while len(raw) > width:
                cut = raw.rfind(" ", 0, width + 1)
                if cut <= 0:
                    cut = width
                out.append(raw[:cut].rstrip())
                raw = raw[cut:].lstrip()
            out.append(raw)
        return out

    def chat_lines(self, width: int) -> list[list[Span]]:
        """Build the transcript as lines of styled spans.

        Each line is a list because Markdown can put several styles on one line
        (e.g. a paragraph containing *italic* and **bold**). The speaker prefix
        is a named span so it keeps the role colour even when the body is
        Markdown.
        """
        lines: list[list[Span]] = []
        for msg in self.messages:
            if msg.role == "user":
                label, color = "you", "user"
            elif msg.role == "assistant":
                label, color = "v4", "assistant"
            elif msg.role == "error":
                label, color = "!!", "error"
            elif msg.role == "tool":
                label, color = "py", "tool"
            else:
                label, color = "··", "system"

            body = msg.text
            label_prefix = f"{label} › "
            indent = " " * len(label_prefix)
            # The first line carries the speaker prefix, so the body has to wrap
            # to the remaining columns. Wrapping to width-4 made the first line
            # one column too wide ("v4 › ") and the renderer clipped its tail.
            body_width = max(8, width - len(label_prefix))

            use_md = msg.role in ("assistant", "tool") and RICH_AVAILABLE
            if use_md:
                md_width = max(10, body_width)
                # The body only grows while streaming, so (length, width) is a
                # sufficient cache key.
                key = (len(body), md_width)
                if msg.md_cache_key != key:
                    rendered = render_markdown(body, md_width) if body else None
                    if rendered is None:
                        rendered = [[Span(ln)] for ln in self._wrap(body, md_width)]
                    msg.md_cache_lines = rendered
                    msg.md_cache_key = key
                md_lines = msg.md_cache_lines or [[]]
                for i, spans in enumerate(md_lines):
                    prefix = Span(label_prefix if i == 0 else indent,
                                  named=color)
                    lines.append([prefix] + list(spans))
            else:
                for i, ln in enumerate(self._wrap(body, body_width)):
                    prefix = label_prefix if i == 0 else indent
                    lines.append([Span(prefix + ln, named=color)])

            if msg.meta:
                m = msg.meta
                meta = (f"{m.get('generated', 0)} tok · "
                        f"{float(m.get('seconds') or 0):.1f}s · "
                        f"{float(m.get('tps') or 0):.2f} tok/s · "
                        f"ttft {float(m.get('ttft') or 0):.1f}s · "
                        f"{m.get('stop', '')}")
                for ln in self._wrap(meta, body_width):
                    lines.append([Span(indent + ln, named="dim")])
            lines.append([])
        return lines

    def _draw_chat(self, scr: "curses.window", top: int, bottom: int,
                   width: int) -> None:
        view_h = bottom - top + 1
        lines = self.chat_lines(max(10, width))
        max_scroll = max(0, len(lines) - view_h)
        self.max_scroll = max_scroll
        self.scroll = min(self.scroll, max_scroll)
        end = max(0, len(lines) - self.scroll)
        start = max(0, end - view_h)
        visible = lines[start:end]

        # Right-align the transcript to the bottom while it fits, so a new chat
        # starts at the bottom edge rather than floating at the top.
        pad = max(0, view_h - len(visible))
        for i, spans in enumerate(visible):
            row = top + pad + i
            if row > bottom:
                break
            x = 0
            for span in spans:
                if x >= width:
                    break
                text = span.text.replace("\t", "    ")
                text = "".join(c if c.isprintable() else "?" for c in text)
                if not text:
                    continue
                self._add(scr, row, x, text, width - x, self.span_attr(span))
                x += len(text)

    def _draw_separator(self, scr: "curses.window", top: int, bottom: int,
                        col: int) -> None:
        for r in range(top, bottom + 1):
            self._add(scr, r, col, "|", 1, self.attr("dim"))

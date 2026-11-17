# SPDX-License-Identifier: Apache-2.0
"""Rich Markdown -> lines of styled spans."""

from __future__ import annotations

from typing import Optional

from .markdown import (RICH_AVAILABLE, Console, Markdown, Span, _THEME, _append,
                      _is_blank, _style_span, _trim_line)


def render_markdown(text: str, width: int) -> Optional[list[list[Span]]]:
    """Render Markdown to lines of styled spans, or None if Rich is unavailable.

    `width` is the number of columns available for the Markdown output itself
    (the caller adds the speaker prefix). Rich wraps to this width, including
    list indentation and code blocks.
    """
    if not RICH_AVAILABLE or not text or width < 8:
        return None
    try:
        console = Console(
            width=max(8, width),
            color_system="256",
            theme=_THEME,
            force_terminal=False,
            legacy_windows=False,
        )
        lines: list[list[Span]] = [[]]
        for segment in console.render(Markdown(text, code_theme="monokai")):
            if getattr(segment, "control", None):
                continue
            chunk = segment.text
            start = 0
            for i, ch in enumerate(chunk):
                if ch != "\n":
                    continue
                if i > start:
                    _append(lines[-1], _style_span(chunk[start:i], segment.style))
                lines.append([])
                start = i + 1
            if start < len(chunk):
                _append(lines[-1], _style_span(chunk[start:], segment.style))

        for line in lines:
            _trim_line(line)
        while lines and _is_blank(lines[0]):
            lines.pop(0)
        while lines and _is_blank(lines[-1]):
            lines.pop()
        return lines
    except Exception:
        # A malformed/incomplete fenced block while streaming is normal. The
        # caller falls back to plain text for this render rather than killing
        # the TUI over a cosmetic failure.
        return None

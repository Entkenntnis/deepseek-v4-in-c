# SPDX-License-Identifier: Apache-2.0
"""Markdown -> styled spans for the curses TUI.

WHY THIS IS A SEPARATE MODULE
    The chat pane is a curses surface, not a terminal stream: it cannot accept
    ANSI escape codes. Rich already knows how to parse CommonMark, lay out
    headings/lists/tables/quotes and syntax-highlight fenced code, so the useful
    division is:

        Markdown text -> Rich segments -> Span objects -> curses attrs/pairs

    This module owns the first two arrows and has no curses dependency. The TUI
    owns the last one (see App.span_attr / App.pair_for).

WHY RICH IS OPTIONAL
    The TUI worked with no third-party packages, and that is a property worth
    keeping for a prototype. If Rich is absent, render_markdown() returns None
    and the caller keeps its plain wrapped text. Install it with:

        python3 -m pip install -r tui/requirements.txt

COLOUR CHOICES
    The theme is deliberately close to the request and to opencode's palette:
      * emphasis / italic  -> green italic
      * strong / bold      -> orange bold
      * headings           -> orange (h1/h2), yellow (h3+)
      * inline code        -> cyan bold
      * bullets            -> orange; ordered-list numbers -> cyan
      * quotes             -> magenta
      * links              -> blue underline
      * fenced code        -> monokai syntax colours on a dark background

    Colours are reported as xterm-256 indexes when Rich can express them that
    way, or as RGB triplets for syntax themes. The caller downgrades those to
    the terminal's actual palette in App.pair_for.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union

try:
    from rich.color import ColorSystem
    from rich.console import Console
    from rich.markdown import Heading, Markdown
    from rich.theme import Theme

    RICH_AVAILABLE = True
    RICH_ERROR = ""
except Exception as exc:                      # ImportError, broken install, ...
    RICH_AVAILABLE = False
    RICH_ERROR = str(exc)
    ColorSystem = None                        # type: ignore[assignment]
    Console = None                            # type: ignore[assignment]
    Heading = None                            # type: ignore[assignment]
    Markdown = None                           # type: ignore[assignment]
    Theme = None                              # type: ignore[assignment]


ColorSpec = Union[int, tuple[int, int, int], None]


@dataclass(frozen=True)
class Span:
    """A run of text with one visual style.

    `named` is for the TUI's own role colours (user/assistant/error/...). When it
    is set, fg/bg and the boolean attributes are ignored. Otherwise fg/bg are
    xterm-256 indexes or RGB triplets, and the caller maps them to curses pairs.
    """
    text: str
    fg: ColorSpec = None
    bg: ColorSpec = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    dim: bool = False
    reverse: bool = False
    named: Optional[str] = None


_THEME = None
if RICH_AVAILABLE:
    # Rich's Markdown renderer uses markdown.<tag> keys. In particular note
    # "markdown.em", not "markdown.emph": the inline token tag is `em`.
    _THEME = Theme({
        "markdown.h1": "bold orange3",
        "markdown.h2": "bold orange3",
        "markdown.h3": "bold yellow",
        "markdown.h4": "bold yellow",
        "markdown.h5": "bold cyan",
        "markdown.h6": "bold cyan",
        "markdown.em": "italic green",
        "markdown.strong": "bold orange3",
        "markdown.code": "bold cyan",
        "markdown.paragraph": "none",
        "markdown.item.bullet": "bold orange3",
        "markdown.item.number": "cyan",
        "markdown.link": "underline blue",
        "markdown.link_url": "underline blue",
        "markdown.block_quote": "magenta",
        "markdown.table.header": "bold orange3",
        "markdown.table.border": "cyan",
        "markdown.hr": "dim",
    })
    # Rich centres H1 by default, which looks wrong as the first line of a chat
    # bubble. Left-align every heading.
    try:
        Heading.LEVEL_ALIGN["h1"] = "left"
    except Exception:
        pass


def _color_spec(color) -> ColorSpec:
    """Rich Color -> xterm-256 index, RGB triplet, or None (terminal default)."""
    if color is None or getattr(color, "is_default", False):
        return None
    if getattr(color, "number", None) is not None:
        return int(color.number)
    triplet = getattr(color, "triplet", None)
    if triplet is not None:
        return (int(triplet.red), int(triplet.green), int(triplet.blue))
    # Named standard colour that Rich has not resolved to a number.
    try:
        downgraded = color.downgrade(ColorSystem.EIGHT_BIT)
        if downgraded.number is not None:
            return int(downgraded.number)
    except Exception:
        pass
    return None


def _style_span(text: str, style) -> Span:
    if not text:
        return Span("")
    if style is None:
        return Span(text)
    return Span(
        text,
        fg=_color_spec(getattr(style, "color", None)),
        bg=_color_spec(getattr(style, "bgcolor", None)),
        bold=bool(getattr(style, "bold", False)),
        italic=bool(getattr(style, "italic", False)),
        underline=bool(getattr(style, "underline", False)),
        dim=bool(getattr(style, "dim", False)),
        reverse=bool(getattr(style, "reverse", False)),
    )


def _same_style(a: Span, b: Span) -> bool:
    return (a.fg == b.fg and a.bg == b.bg and a.bold == b.bold
            and a.italic == b.italic and a.underline == b.underline
            and a.dim == b.dim and a.reverse == b.reverse
            and a.named == b.named)


def _append(line: list[Span], span: Span) -> None:
    if not span.text:
        return
    if line and _same_style(line[-1], span):
        line[-1] = replace(line[-1], text=line[-1].text + span.text)
    else:
        line.append(span)


def _trim_line(line: list[Span]) -> list[Span]:
    """Drop trailing whitespace, but keep code indentation at the front."""
    while line:
        last = line[-1]
        stripped = last.text.rstrip()
        if stripped:
            if stripped != last.text:
                line[-1] = replace(last, text=stripped)
            break
        line.pop()
    return line


def _is_blank(line: list[Span]) -> bool:
    return not any(s.text.strip() for s in line)


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

# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim; the implementation lives in dsv4tui/markdown_render.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsv4tui.markdown import RICH_AVAILABLE, RICH_ERROR, Span  # noqa: E402
from dsv4tui.markdown_render import render_markdown  # noqa: E402

__all__ = ["RICH_AVAILABLE", "RICH_ERROR", "Span", "render_markdown"]

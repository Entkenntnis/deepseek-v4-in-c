# SPDX-License-Identifier: Apache-2.0
"""Argument parsing and process entry point."""

from __future__ import annotations

import argparse
import curses
import locale
import os
import sys
from pathlib import Path
from typing import Optional

from .app import App
from .backend import EventSink
from .defaults import DEFAULT_BINARY, DEFAULT_MODEL, DEFAULT_TOK, DEFAULT_TRUNK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live TUI for deepseek-v4-in-c (streams tokens, timings, "
                    "memory, cache and disk stats).")
    p.add_argument("--demo", action="store_true",
                   help="synthetic backend; no checkpoint required")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.demo:
        if not Path(DEFAULT_BINARY).exists():
            print(f"error: {DEFAULT_BINARY} does not exist.\n"
                  f"       run `make serve` first, or use --demo.",
                  file=sys.stderr)
            return 2
        for what, path in (("model", DEFAULT_MODEL), ("trunk", DEFAULT_TRUNK),
                           ("tokenizer", DEFAULT_TOK)):
            if not Path(path).exists():
                print(f"error: {what} path does not exist: {path}", file=sys.stderr)
                return 2

    locale.setlocale(locale.LC_ALL, "")
    os.environ.setdefault("ESCDELAY", "25")
    sink = EventSink()
    app = App(args, sink)
    app.start_backend()

    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
    return 0

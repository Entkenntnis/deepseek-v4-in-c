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

def parse_budget(value: str) -> float | str:
    """Accept a GB number or the literal 'auto' for the engine's own sizing."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        gb = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number of GB or 'auto', got {value!r}") from None
    if gb <= 0:
        raise argparse.ArgumentTypeError("budget must be greater than 0 GB")
    return gb

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live TUI for deepseek-v4-in-c (streams tokens, timings, "
                    "memory, cache and disk stats).")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help="checkpoint directory (default: %(default)s)")
    p.add_argument("--trunk", default=str(DEFAULT_TRUNK),
                   help="packed trunk directory (default: %(default)s)")
    p.add_argument("--tok", default=str(DEFAULT_TOK),
                   help="packed tokenizer file (default: %(default)s)")
    p.add_argument("--binary", default=str(DEFAULT_BINARY),
                   help="dsv4-serve binary (default: %(default)s)")
    p.add_argument("--budget", type=parse_budget, default=10.0,
                   metavar="GB|auto",
                   help="RAM budget in GB handed to trunk+cache, or 'auto' to "
                        "size it from available memory (default: 10)")
    p.add_argument("--ctx", type=int, default=4096,
                   help="context capacity in tokens (default: 4096)")
    p.add_argument("--gen", type=int, default=256,
                   help="max tokens per turn (default: 256)")
    p.add_argument("--threads", type=int, default=None,
                   help="OpenMP compute threads; default: engine policy")
    p.add_argument("--pin", choices=["off", "all", "p", "e"], default=None,
                   help="compute affinity: off, all, P-cores (p) or E-cores (e); "
                        "default: engine policy")
    p.add_argument("--readers", type=int, default=None,
                   help="expert-reader threads; default: engine policy (6)")
    p.add_argument("--reader-pin", choices=["off", "all", "p", "e"],
                   default=None,
                   help="affinity for the expert-reader threads; default: engine")
    p.add_argument("--system", default="",
                   help="system prompt for a new context")
    p.add_argument("--think", action="store_true",
                   help="start in thinking mode")
    p.add_argument("--no-layers", action="store_true",
                   help="do not stream per-layer events (less chatter)")
    p.add_argument("--no-markdown", action="store_true",
                   help="render assistant output as plain text, not Markdown")
    p.add_argument("--demo", action="store_true",
                   help="synthetic backend; no checkpoint required")
    return p

def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.demo:
        if not Path(args.binary).exists():
            print(f"error: {args.binary} does not exist.\n"
                  f"       run `make serve` first, or use --demo.",
                  file=sys.stderr)
            return 2
        for what, path in (("model", args.model), ("trunk", args.trunk),
                           ("tokenizer", args.tok)):
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

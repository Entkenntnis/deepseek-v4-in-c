# SPDX-License-Identifier: Apache-2.0
"""Paths, checkpoint tokens and the small numeric formatters."""

from __future__ import annotations

import math
from pathlib import Path

HOME = Path.home()
DEFAULT_MODEL = HOME / "colibri-models" / "deepseek-v4-flash"
DEFAULT_TRUNK = HOME / "colibri-models" / "dsv4-trunk-flash"
DEFAULT_TOK = HOME / "colibri-models" / "dsv4-tok-flash.bin"
# tui/dsv4tui/defaults.py -> repo root
REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_BINARY = REPO / "bin" / "dsv4-serve"

# These are spelled as they are in the checkpoint's own encoding/README.md.
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
SPECIAL_TOKENS = (BOS, EOS, USER, ASSISTANT)


def strip_special_tokens(s: str) -> str:
    for tok in SPECIAL_TOKENS:
        s = s.replace(tok, "")
    return s


def fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, sec = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {sec:04.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m:02d}m"


def fmt_bytes_gb(gb: float) -> str:
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{gb * 1024:.0f} MB"


def fmt_gbps(value: float) -> str:
    if value >= 100.0:
        return f"{value:.0f} GB/s"
    if value >= 10.0:
        return f"{value:.1f} GB/s"
    return f"{value:.2f} GB/s"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def nice_step(value: float, target: int = 5) -> float:
    """A round axis interval, so labels are 0, 0.25, 0.5, ... not max/2."""
    if value <= 0 or target < 1:
        return 1.0
    raw = value / target
    base = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mult * base:
            return mult * base
    return 10.0 * base


def axis_label(value: float, step: float) -> str:
    """Keep a decimal on whole ticks when the interval itself is fractional."""
    if step < 1.0 and abs(value - round(value)) < 1e-9:
        return "0" if abs(value) < 1e-9 else f"{value:.1f}"
    return f"{value:g}"

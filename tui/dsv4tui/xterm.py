# SPDX-License-Identifier: Apache-2.0
"""Map arbitrary RGB to an index the current terminal can init_pair()."""

from __future__ import annotations

# xterm's first 16 colours. Rich downgrades named/true colours to an xterm-256
# index, but a terminal with only 8 or 16 colours needs one more step down.
_XTERM_STD = (
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
)


def _xterm_rgb(n: int) -> tuple[int, int, int]:
    n = int(n)
    if 0 <= n < 16:
        return _XTERM_STD[n]
    if 16 <= n < 232:
        n -= 16
        levels = (0, 95, 135, 175, 215, 255)
        return levels[(n // 36) % 6], levels[(n // 6) % 6], levels[n % 6]
    if 232 <= n < 256:
        v = 8 + (n - 232) * 10
        return v, v, v
    return 0, 0, 0


def _dist2(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2)


def _nearest_basic(rgb: tuple[int, int, int], n: int) -> int:
    return min(range(max(1, min(n, 16))), key=lambda i: _dist2(rgb, _XTERM_STD[i]))


def _rgb_to_xterm(rgb: tuple[int, int, int]) -> int:
    return min(range(256), key=lambda i: _dist2(rgb, _xterm_rgb(i)))

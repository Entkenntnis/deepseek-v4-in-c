#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""dsv4_tui.py - a live terminal front end for deepseek-v4-in-c.

The implementation lives in the dsv4tui package next to this file, split into
modules of at most 200 lines.

Usage:
    python3 tui/dsv4_tui.py          # real model, built-in defaults
    python3 tui/dsv4_tui.py --demo   # no model needed

Keys:
    Enter send, /help for commands.
    Ctrl-C cancels a running turn, clears the input, or quits.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts tui/ on sys.path; -m tui.dsv4_tui does not.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsv4tui.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

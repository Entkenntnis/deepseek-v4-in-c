#!/usr/bin/env python3
"""Check the two structural rules of src/.

1. Every .c/.h file is at most MAX_LINES lines.
2. An include of another src header may only point at the same folder or a
   LOWER-numbered one.

Run from the repository root: python3 tools/check_src.py
"""
import re
import sys
from pathlib import Path

MAX_LINES = 200
SRC = Path("src")
INC_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def layer_of(path: Path) -> int:
    m = re.match(r"(\d+)_", path.parent.name)
    if not m:
        sys.exit(f"check_src: {path.parent} has no numeric prefix")
    return int(m.group(1))


def main() -> int:
    headers: dict[str, int] = {}
    for h in SRC.glob("*/*.h"):
        headers[h.name] = layer_of(h)

    errors: list[str] = []
    for f in sorted(SRC.glob("*/*.[ch]")):
        own = layer_of(f)
        text = f.read_text(encoding="utf-8", errors="replace")
        lines = text.count("\n") + (0 if text.endswith("\n") else 1)
        if lines > MAX_LINES:
            errors.append(f"{f}: {lines} lines (limit {MAX_LINES})")
        for inc in INC_RE.findall(text):
            name = Path(inc).name
            target = headers.get(name)
            if target is not None and target > own:
                errors.append(
                    f"{f}: includes {inc} from layer {target:02d} "
                    f"(own layer {own:02d}) -- dependencies point downward")

    if errors:
        print("check_src: FAILED", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    print(f"check_src: ok -- {len(headers)} headers, no upward includes, "
          f"all files <= {MAX_LINES} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())

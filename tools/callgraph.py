#!/usr/bin/env python3
"""Generate a file-to-file C call graph for the sources under src/.

The graph only contains .c files.  An edge A -> B means: a function defined in
A calls a function that is defined in B (headers are deliberately ignored).
The edge label is the number of call sites between the two files; the SVG
tooltip on an edge lists the called function names.

Usage:
    python3 tools/callgraph.py [repo-root]

Outputs src-callgraph.dot and src-callgraph.svg in the repo root.
"""
import html
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
SRC = ROOT / "src"
OUT_DOT = ROOT / "src-callgraph.dot"
OUT_SVG = ROOT / "src-callgraph.svg"

KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "do", "else",
    "defined", "__attribute__", "__asm__", "asm", "_Pragma", "alignof",
    "_Alignof", "_Static_assert", "typeof", "__typeof__", "case", "goto",
}

# (cluster fill, node fill, cluster border / edge colour)
GROUP_COLORS = {
    "00_base":    ("#eef4ff", "#dbe8ff", "#4c6fa8"),
    "10_compute": ("#eafaf8", "#d2f2ed", "#2c7a7b"),
    "20_runtime": ("#fff8e8", "#ffedc7", "#a8781a"),
    "40_io":      ("#f5efff", "#e8daff", "#6b46c1"),
    "50_loader":  ("#fff1e8", "#ffdfc7", "#b45309"),
    "60_cache":   ("#edf9ef", "#d5f2da", "#2f855a"),
    "70_decode":  ("#fff0f7", "#ffd9ea", "#b83280"),
    "80_prefill": ("#eef7ff", "#d6ebff", "#2b6cb0"),
    "90_cli":     ("#f4f5f7", "#e2e5e9", "#4a5568"),
}
DEFAULT_COLOR = ("#f7fafc", "#edf2f7", "#718096")


def strip_comments_and_directives(text):
    """Blank out comments, strings and preprocessor lines, keep newlines.

    Offsets are not preserved exactly (strings become spaces of equal length,
    comments too, so actually they are), which keeps the brace/paren scanner
    simple and correct for ordinary C.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c == "/" and nxt == "/":
            j = text.find("\n", i + 2)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c in ('"', "'"):
            quote = c
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == quote:
                    j += 1
                    break
                j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    text = "".join(out)

    lines = text.split("\n")
    result = []
    continued = False
    for line in lines:
        if continued or line.lstrip().startswith("#"):
            continued = line.rstrip().endswith("\\")
            result.append("")
        else:
            result.append(line)
    return "\n".join(result)


def match_brace(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_functions(text):
    """Yield (name, is_static, body_start, body_end) for top-level definitions."""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            if depth == 0:
                k = i - 1
                while k >= 0 and text[k].isspace():
                    k -= 1
                # A top-level "{ directly after ")" is a function definition.
                if k >= 0 and text[k] == ")":
                    pdepth = 0
                    p = k
                    while p >= 0:
                        if text[p] == ")":
                            pdepth += 1
                        elif text[p] == "(":
                            pdepth -= 1
                            if pdepth == 0:
                                break
                        p -= 1
                    if p > 0:
                        m = re.search(r"([A-Za-z_]\w*)\s*$", text[:p])
                        if m and m.group(1) not in KEYWORDS:
                            name = m.group(1)
                            start = max(
                                text.rfind(";", 0, p),
                                text.rfind("}", 0, p),
                                text.rfind("{", 0, p),
                            )
                            header = text[start + 1:p]
                            is_static = bool(re.search(r"\bstatic\b", header))
                            end = match_brace(text, i)
                            if end > 0:
                                yield name, is_static, i, end
                                i = end + 1
                                continue
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
        i += 1


CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def calls_in(body):
    for m in CALL_RE.finditer(body):
        name = m.group(1)
        if name not in KEYWORDS:
            yield name


def nid(rel):
    return "f_" + re.sub(r"\W", "_", rel)


def main():
    files = sorted(SRC.rglob("*.c"))
    definitions = []                       # (file, name, is_static)
    bodies = defaultdict(list)             # file -> [(name, body)]
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        text = strip_comments_and_directives(f.read_text(errors="replace"))
        for name, is_static, a, b in extract_functions(text):
            definitions.append((rel, name, is_static))
            bodies[rel].append((name, text[a:b + 1]))

    by_name = defaultdict(list)
    for rel, name, is_static in definitions:
        by_name[name].append((rel, is_static))

    def resolve(name, caller_file):
        cands = by_name.get(name, [])
        if not cands:
            return None
        for rel, _st in cands:             # same file wins (also for statics)
            if rel == caller_file:
                return rel
        nonstatic = [rel for rel, st in cands if not st]
        if len(set(nonstatic)) == 1:       # unique global definition
            return nonstatic[0]
        return None                        # ambiguous / static elsewhere

    edges = defaultdict(int)
    call_details = defaultdict(lambda: defaultdict(int))
    for rel, funcs in bodies.items():
        for _fn, body in funcs:
            for callee in calls_in(body):
                dst = resolve(callee, rel)
                if dst is None or dst == rel:
                    continue
                edges[(rel, dst)] += 1
                call_details[(rel, dst)][callee] += 1

    groups = defaultdict(list)
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        parts = Path(rel).parts
        groups[parts[1] if len(parts) > 2 else "src"].append(rel)

    def group_of(rel):
        parts = Path(rel).parts
        return parts[1] if len(parts) > 2 else "src"

    L = []
    L.append("digraph srccallgraph {")
    L.append('  graph [rankdir=LR, splines=spline, nodesep=0.40, ranksep=1.35,')
    L.append('         bgcolor="#ffffff", fontname="Helvetica", labelloc=t, labeljust=c,')
    L.append('         outputorder=edgesfirst,')
    L.append('         label="C-Aufrufgraph der .c-Dateien unter src/\\n'
             'Kante A → B: eine in A definierte Funktion ruft eine in B definierte Funktion auf\\n'
             'Namen an der Kante = aufgerufene Funktionen, ×N = Anzahl ihrer Aufrufstellen '
             '(Header/Inline-Helfer ignoriert)",')
    L.append('         fontsize=17];')
    L.append('  node [shape=box, style="rounded,filled", fontname="Helvetica",')
    L.append('        fontsize=9.5, color="#4a5568", fillcolor="#edf2f7", margin="0.09,0.04"];')
    L.append('  edge [fontname="Helvetica", fontsize=7.5, fontcolor="#1a202c", arrowsize=0.65];')

    for group in sorted(groups):
        fill, nodefill, color = GROUP_COLORS.get(group, DEFAULT_COLOR)
        L.append(f'  subgraph cluster_{re.sub(r"[^A-Za-z0-9_]", "_", group)} {{')
        L.append(f'    label="{html.escape(group)}";')
        L.append(f'    style="rounded,dashed,filled"; fillcolor="{fill}"; '
                 f'color="{color}"; fontsize=13; fontname="Helvetica-Bold"; fontcolor="{color}";')
        for rel in sorted(groups[group]):
            label = Path(rel).name
            L.append(f'    {nid(rel)} [label="{label}", fillcolor="{nodefill}", '
                     f'color="{color}", tooltip="{html.escape(rel)}"];')
        L.append("  }")

    max_count = max(edges.values()) if edges else 1
    for (src, dst), count in sorted(edges.items(),
                                    key=lambda kv: (-kv[1], kv[0])):
        penwidth = 0.6 + 2.6 * (count / max_count) ** 0.5
        details = sorted(call_details[(src, dst)].items(),
                         key=lambda kv: (-kv[1], kv[0]))
        label = "\\n".join(
            f"{name} ×{n}" if n > 1 else name for name, n in details)
        tip = (f"{src} -> {dst}\\n{count} Aufrufstellen\\n" +
               "\\n".join(f"{name}: {n}" for name, n in details))
        color = GROUP_COLORS.get(group_of(src), DEFAULT_COLOR)[2]
        L.append(f'  {nid(src)} -> {nid(dst)} [penwidth={penwidth:.2f}, '
                 f'color="{color}", label="{label}", '
                 f'tooltip="{html.escape(tip)}"];')

    L.append("}")
    OUT_DOT.write_text("\n".join(L) + "\n")

    subprocess.run(["dot", "-Tsvg", str(OUT_DOT), "-o", str(OUT_SVG)], check=True)

    print(f"{len(files)} .c files, {len(definitions)} function definitions, "
          f"{len(edges)} file-to-file edges")
    print(f"wrote {OUT_SVG.relative_to(ROOT)} and {OUT_DOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

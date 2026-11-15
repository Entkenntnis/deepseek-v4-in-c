#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end benchmark for the resident DeepSeek-V4 engine.

One serve process, three generations of the same prompt. The first run is the
warm-up (disk, expert cache and allocator); runs 2 and 3 are the measured ones.
`reset` between runs clears the KV/compressor state but deliberately keeps the
expert cache and the pinned trunk, which is what makes the measured runs
comparable.

The golden output is the generated token-id sequence. It is compared exactly:
the engine's reduction order is designed to be independent of cache hits and
read completion order, so a different sequence is a functional regression, not
noise. Use --update-golden only after an intentional model/kernel change.

Examples:
    python3 bench/e2e.py                  # table + minimal progress
    python3 bench/e2e.py --detailed       # serve diagnostics + per-run detail
    python3 bench/e2e.py --runs 3 --update-golden
    python3 bench/e2e.py --json
"""
import argparse
import codecs
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = HOME / "colibri-models" / "deepseek-v4-flash"
DEFAULT_TRUNK = HOME / "colibri-models" / "dsv4-trunk-flash"
DEFAULT_TOK = HOME / "colibri-models" / "dsv4-tok-flash.bin"
DEFAULT_BIN = REPO / "bin" / "dsv4-serve"
DEFAULT_GOLDEN = REPO / "bench" / "golden" / "e2e_flash_b16.json"

PROMPT = "\n".join(["Vielleicht müssen wir uns in Zukunft doch ..."] * 5)


def env_path(name, default):
    return Path(os.path.expanduser(os.environ.get(name, str(default))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=env_path("DSV4_MODEL", DEFAULT_MODEL))
    p.add_argument("--trunk", default=env_path("DSV4_TRUNK", DEFAULT_TRUNK))
    p.add_argument("--tok", default=env_path("DSV4_TOK", DEFAULT_TOK))
    p.add_argument("--binary", default=env_path("DSV4_SERVE_BIN", DEFAULT_BIN))
    p.add_argument("--budget", type=float, default=16.0)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--gen", type=int, default=30)
    p.add_argument("--runs", type=int, default=3,
                   help="total generations in one process; run 1 is warm-up")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pin", default="p", choices=("off", "all", "p", "e"))
    p.add_argument("--readers", type=int, default=6)
    p.add_argument("--reader-pin", default="off", choices=("off", "all", "p", "e"))
    p.add_argument("--stream-miss", type=int, default=6)
    p.add_argument("--golden", default=DEFAULT_GOLDEN)
    p.add_argument("--update-golden", action="store_true")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="print a JSON summary instead of the table")
    p.add_argument("--detailed", action="store_true",
                   help="show serve stderr and per-run counters")
    return p.parse_args()


def build_command(a):
    return [
        str(a.binary),
        "--model", str(a.model),
        "--trunk", str(a.trunk),
        "--tok", str(a.tok),
        "--budget", str(a.budget),
        "--ctx", str(a.ctx),
        "--threads", str(a.threads),
        "--pin", a.pin,
        "--readers", str(a.readers),
        "--reader-pin", a.reader_pin,
        "--stream-miss", str(a.stream_miss),
    ]


def send(proc, obj):
    proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def read_event(proc):
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            print(f"e2e: non-JSON output: {line}", file=sys.stderr)
            return None


def wait_for(proc, event):
    while True:
        ev = read_event(proc)
        if ev is None:
            return None
        if ev.get("event") == "error":
            raise RuntimeError(ev.get("message", "engine error"))
        if ev.get("event") == event:
            return ev


class Progress:
    """Minimal stderr progress, kept off the table's stdout."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.tty = sys.stderr.isatty()
        self.last = None

    def update(self, text, force=False):
        if not self.enabled:
            return
        if self.tty:
            if force or text != self.last:
                sys.stderr.write("\r" + text + "\x1b[K")
                sys.stderr.flush()
                self.last = text
        elif force and text != self.last:
            print(text, file=sys.stderr, flush=True)
            self.last = text

    def finish(self, text):
        if not self.enabled:
            return
        if self.tty:
            sys.stderr.write("\r" + text + "\x1b[K\n")
        else:
            print(text, file=sys.stderr, flush=True)
        sys.stderr.flush()
        self.last = None


def dump_stderr(errfile, lines=40):
    if errfile is None:
        return
    try:
        errfile.flush()
        errfile.seek(0)
        text = errfile.read()
    except (OSError, ValueError):
        return
    tail = text.strip().splitlines()[-lines:]
    if tail:
        print("--- serve stderr (tail) ---", file=sys.stderr)
        print("\n".join(tail), file=sys.stderr)


def generate(proc, prompt, gen, label, progress):
    send(proc, {"cmd": "generate", "prompt": prompt, "gen": gen})
    start = None
    tokens = []
    while True:
        ev = read_event(proc)
        if ev is None:
            raise RuntimeError("serve exited during generation")
        kind = ev.get("event")
        if kind == "start":
            start = ev
            progress.update(f"{label}  prefill", force=True)
        elif kind == "prefill":
            pct = 100.0 * float(ev.get("progress", 0.0))
            stage = ev.get("stage")
            if progress.tty:
                progress.update(f"{label}  prefill {pct:3.0f}%")
            elif stage != "layer":
                progress.update(f"{label}  prefill {pct:3.0f}%", force=True)
        elif kind == "token":
            tokens.append(ev)
            if progress.tty:
                progress.update(f"{label}  decode {len(tokens)}/{gen}")
            elif len(tokens) == 1:
                progress.update(f"{label}  decode", force=True)
        elif kind == "done":
            if start is None:
                raise RuntimeError("done without start")
            progress.finish(f"{label}  done {len(tokens)} tokens")
            return start, tokens, ev
        elif kind == "error":
            raise RuntimeError(ev.get("message", "engine error"))


def decode_text(tokens):
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    out = []
    for t in tokens:
        out.append(dec.decode(bytes.fromhex(t.get("hex", ""))))
    out.append(dec.decode(b"", final=True))
    return "".join(out)


def token_hash(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def metrics(start, tokens, done):
    prefill_tokens = int(start.get("prefill_tokens") or done.get("prompt_tokens") or 0)
    prefill_s = float(tokens[0].get("prefill_s", 0.0)) if tokens else 0.0
    prefill_tps = prefill_tokens / prefill_s if prefill_s > 0 else 0.0

    n = len(tokens)
    if n > 1:
        dt = float(tokens[-1].get("t", 0.0)) - float(tokens[0].get("t", 0.0))
        decode_tps = (n - 1) / dt if dt > 0 else 0.0
    else:
        decode_tps = 0.0

    c0 = start.get("cache", {})
    c1 = done.get("cache", {})
    hits = int(c1.get("hit", 0)) - int(c0.get("hit", 0))
    misses = int(c1.get("miss", 0)) - int(c0.get("miss", 0))
    reads = float(c1.get("read_gb", 0.0)) - float(c0.get("read_gb", 0.0))
    hit_rate = hits / (hits + misses) if hits + misses else 0.0

    return {
        "prefill_tokens": prefill_tokens,
        "prefill_s": prefill_s,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "generated": n,
        "stop": done.get("stop", "?"),
        "cache_hit_rate": hit_rate,
        "cache_read_gb": reads,
        "sys": done.get("sys", {}),
        "prof": done.get("prof", {}),
    }


def print_table(rows):
    print(f"{'run':<8} {'prefill tok/s':>14} {'decode tok/s':>13} "
          f"{'hit%':>7} {'read GB':>9} {'generated':>10} {'stop':>6}")
    for name, m in rows:
        print(f"{name:<8} {m['prefill_tps']:14.2f} {m['decode_tps']:13.2f} "
              f"{100.0 * m['cache_hit_rate']:7.1f} {m['cache_read_gb']:9.2f} "
              f"{m['generated']:10d} {m['stop']:>6}")


def main():
    a = parse_args()
    if a.runs < 2:
        print("e2e: --runs must be at least 2 (one warm-up plus a measured run)",
              file=sys.stderr)
        return 2
    if not Path(a.binary).is_file():
        print(f"e2e: {a.binary} not found; run `make serve` first", file=sys.stderr)
        return 2

    progress = Progress(enabled=not a.detailed and not a.as_json)
    errfile = None
    if not a.detailed or a.as_json:
        errfile = tempfile.TemporaryFile(mode="w+", encoding="utf-8")

    if a.detailed and not a.as_json:
        print(f"e2e: model   {a.model}", file=sys.stderr)
        print(f"e2e: trunk   {a.trunk}", file=sys.stderr)
        print(f"e2e: binary  {a.binary}", file=sys.stderr)
        print(f"e2e: gen={a.gen} budget={a.budget:g} GB ctx={a.ctx} "
              f"runs={a.runs} (run 1 warm-up)", file=sys.stderr)
        print(f"e2e: prompt  {len(PROMPT)} chars, repeated 5x", file=sys.stderr)

    progress.update("loading model", force=True)
    proc = subprocess.Popen(build_command(a), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=errfile,
                            text=True, encoding="utf-8", bufsize=1)
    try:
        ready = wait_for(proc, "ready")
        if ready is None:
            progress.finish("failed before ready")
            print("e2e: serve exited before ready", file=sys.stderr)
            dump_stderr(errfile)
            return 2
        progress.update("model ready", force=True)

        results = []
        for i in range(a.runs):
            if i:
                send(proc, {"cmd": "reset"})
                if wait_for(proc, "reset") is None:
                    progress.finish("failed during reset")
                    print("e2e: serve exited during reset", file=sys.stderr)
                    dump_stderr(errfile)
                    return 2
            label = f"run {i + 1}/{a.runs}" + (" (warmup)" if i == 0 else "")
            start, tokens, done = generate(proc, PROMPT, a.gen, label, progress)
            ids = [int(t["id"]) for t in tokens]
            m = metrics(start, tokens, done)
            m["token_ids"] = ids
            m["text"] = decode_text(tokens)
            results.append(m)

        send(proc, {"cmd": "quit"})
    except (RuntimeError, BrokenPipeError) as exc:
        progress.finish("failed")
        print(f"e2e: {exc}", file=sys.stderr)
        dump_stderr(errfile)
        return 2
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
        if errfile is not None:
            errfile.close()

    # Functional check first: every run must agree, warm-up included.
    first = results[0]["token_ids"]
    for i, r in enumerate(results[1:], start=2):
        ids = r["token_ids"]
        if ids != first:
            at = next((j for j, (x, y) in enumerate(zip(first, ids)) if x != y),
                      min(len(first), len(ids)))
            print(f"e2e: run {i} diverges from run 1 at token {at}: "
                  f"{first[at:at+5]} vs {ids[at:at+5]}", file=sys.stderr)
            return 1

    ids = first
    digest = token_hash(ids)
    golden_path = Path(os.path.expanduser(str(a.golden)))
    golden = {
        "model": str(a.model),
        "prompt": PROMPT,
        "gen": a.gen,
        "budget_gb": a.budget,
        "ctx": a.ctx,
        "generated": len(ids),
        "stop": results[0]["stop"],
        "token_ids": ids,
        "text": results[0]["text"],
        "sha256": digest,
    }

    golden_path.parent.mkdir(parents=True, exist_ok=True)
    existed = golden_path.is_file()
    status = "PASS"
    if existed and not a.update_golden:
        old = json.loads(golden_path.read_text(encoding="utf-8"))
        if old.get("prompt") != PROMPT or old.get("gen") != a.gen:
            print("e2e: golden was recorded for a different prompt/gen; "
                  "use --update-golden if that change was intentional",
                  file=sys.stderr)
            return 1
        if old.get("token_ids") != ids:
            old_ids = old.get("token_ids", [])
            at = next((j for j, (x, y) in enumerate(zip(old_ids, ids)) if x != y),
                      min(len(old_ids), len(ids)))
            print(f"e2e: GOLDEN MISMATCH at token {at}: "
                  f"golden {old_ids[at:at+5]} vs now {ids[at:at+5]}",
                  file=sys.stderr)
            return 1
        if old.get("stop") != results[0]["stop"]:
            print(f"e2e: GOLDEN MISMATCH stop: golden {old.get('stop')!r} "
                  f"vs now {results[0]['stop']!r}", file=sys.stderr)
            return 1
        status = "PASS (golden)"
    else:
        golden_path.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        status = "GOLDEN UPDATED" if existed else "GOLDEN CREATED"

    measured = results[1:]
    rows = [("warmup", results[0])]
    for i, r in enumerate(measured, start=1):
        rows.append((f"run {i}", r))
    median = {
        "prefill_tps": statistics.median(r["prefill_tps"] for r in measured),
        "decode_tps": statistics.median(r["decode_tps"] for r in measured),
        "cache_hit_rate": statistics.median(r["cache_hit_rate"] for r in measured),
        "cache_read_gb": statistics.median(r["cache_read_gb"] for r in measured),
        "generated": measured[0]["generated"],
        "stop": measured[0]["stop"],
    }
    rows.append(("median", median))

    if a.as_json:
        print(json.dumps({
            "args": {
                "gen": a.gen, "budget_gb": a.budget, "ctx": a.ctx,
                "runs": a.runs, "prompt": PROMPT,
            },
            "runs": [{k: v for k, v in r.items()
                      if k not in ("token_ids", "text")} for r in results],
            "median": median,
            "golden": str(golden_path),
            "sha256": digest,
            "status": status,
        }, ensure_ascii=False, indent=2))
    else:
        print_table(rows)
        if a.detailed:
            print("\ndetails:")
            for idx, m in enumerate(results):
                label = "warmup" if idx == 0 else f"run {idx}"
                print(f"  {label}: prefill {m['prefill_tokens']} tok in "
                      f"{m['prefill_s']:.3f} s ({m['prefill_tps']:.2f} tok/s), "
                      f"decode {m['generated']} tok ({m['decode_tps']:.2f} tok/s), "
                      f"cache {100.0 * m['cache_hit_rate']:.1f}%, "
                      f"read {m['cache_read_gb']:.2f} GB")
                p = m.get("prof", {})
                if p:
                    keys = ("hc", "attn", "gate", "expert_io", "expert_mm",
                            "shared", "head", "bind")
                    print("    prof: " + " ".join(
                        f"{k}={p.get(k, 0.0):.3f}s" for k in keys))
                s = m.get("sys", {})
                if s:
                    print(f"    rss={s.get('rss_mb', 0.0) / 1024.0:.2f} GB "
                          f"peak={s.get('peak_mb', 0.0) / 1024.0:.2f} GB "
                          f"read={s.get('io_read_gb', 0.0):.2f} GB "
                          f"cpu={s.get('cpu_s', 0.0):.1f}s")
            print(f"\ngolden: {golden_path}")
            print(f"sha256: {digest}")
            print(status)
        else:
            print(f"golden: {status}", file=sys.stderr)

    # Warn about the thing the user wants to trust: run-to-run stability.
    if len(measured) >= 2:
        for key in ("prefill_tps", "decode_tps"):
            vals = [r[key] for r in measured]
            lo, hi = min(vals), max(vals)
            if lo > 0 and (hi - lo) / lo > 0.05:
                print(f"warning: {key} varies {100.0 * (hi - lo) / lo:.1f}% "
                      f"across measured runs ({lo:.2f} .. {hi:.2f})",
                      file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

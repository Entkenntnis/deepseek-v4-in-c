#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Targeted prefill benchmark for the resident DeepSeek-V4 engine.

The workload is the first turn of a real TUI tool session: the compact tools
preamble from `tui/dsv4tui/dsml.py` (currently 176 tokens) plus a short user
prompt. The engine wraps both in its chat framing, so the block is exactly 200
tokens. That is the length a tool preamble plus a little user text actually
produces, not a synthetic token array.

One serve process runs the same turn several times. Run 1 is the warm-up (disk,
expert cache and allocator); the later runs are measured after `reset`, which
clears KV/compressor state but keeps the expert cache and pinned trunk. That is
the same warm-cache contract as bench/e2e.py and makes kernel A/B runs
comparable. The expert cache is reported rather than assumed: a 200-token block
routes to thousands of distinct experts, so a 16 GB budget can still miss every
request even after the warm-up. The table shows the real hit rate and bytes
read; the detailed view splits expert and trunk reads.

`gen=1` is deliberate: the first sampled token is the functional reference and
the token event reports `prefill_s`, which stops before the output head. The
script reports both `prefill_s` (pure block prefill) and `ttft` (tokenization,
buffer allocation, prefill, head and sampling), so improvements in either place
are visible.

The golden output is the first token id. A fixed build is deterministic for a
fixed prompt, so a different id is a real change -- or a different prefill
backend. Vulkan and the CPU fallback use different reduction orders, so a
backend change can move the golden. Use --update-golden only intentionally.

Examples:
    make prefill
    python3 bench/prefill.py --detailed
    python3 bench/prefill.py --runs 5 --json
    python3 bench/prefill.py --instrument --instrument-file /tmp/prefill.txt
    python3 bench/prefill.py --trace /tmp/prefill.trace --trace-view
    python3 bench/prefill.py --update-golden
    python3 bench/prefill.py --no-golden
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

sys.path.insert(0, str(REPO / "tui"))
from dsv4tui import proto  # noqa: E402
from dsv4tui.dsml import system_prompt as tui_system_prompt  # noqa: E402
from dsv4tui.tokenizer import Tokenizer  # noqa: E402
DEFAULT_MODEL = HOME / "colibri-models" / "deepseek-v4-flash"
DEFAULT_TRUNK = HOME / "colibri-models" / "dsv4-trunk-flash"
DEFAULT_TOK = HOME / "colibri-models" / "dsv4-tok-flash.bin"
DEFAULT_BIN = REPO / "bin" / "dsv4-serve"
DEFAULT_GOLDEN = REPO / "bench" / "golden" / "prefill_flash_b16.json"

# The first turn of a tool session. With the current compact preamble this is
# exactly 200 tokens: 176 preamble + 24 for the user turn and chat framing.
TARGET_TOKENS = 200
USER_PROMPT = ("Erkläre kurz und sachlich, warum Paris die Hauptstadt "
               "von Frankreich ist.")


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
    p.add_argument("--runs", type=int, default=3,
                   help="total turns in one process; run 1 is warm-up")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pin", default="p", choices=("off", "all", "p", "e"))
    p.add_argument("--readers", type=int, default=6)
    p.add_argument("--reader-pin", default="e", choices=("off", "all", "p", "e"))
    p.add_argument("--stream-miss", type=int, default=6)
    p.add_argument("--prompt", default=USER_PROMPT,
                   help="user text after the tool preamble")
    p.add_argument("--preamble-file", default=None, metavar="FILE",
                   help="use this system prompt instead of the TUI's tools "
                        "preamble")
    p.add_argument("--golden", default=DEFAULT_GOLDEN)
    p.add_argument("--update-golden", action="store_true")
    p.add_argument("--no-golden", action="store_true",
                   help="measure without comparing the first token id")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="print a JSON summary instead of the table")
    p.add_argument("--detailed", action="store_true",
                   help="show serve stderr and per-run counters")
    p.add_argument("--instrument", action="store_true",
                   help="enable deep prefill instrumentation in serve "
                        "(summary + per-layer lines)")
    p.add_argument("--instrument-file", default=None, metavar="FILE",
                   help="write instrumentation lines to FILE instead of stderr")
    p.add_argument("--trace", default=None, metavar="FILE",
                   help="write a per-layer prefill activity trace to FILE")
    p.add_argument("--trace-run", type=int, default=2,
                   help="1-based run to trace (default 2, the first measured)")
    p.add_argument("--trace-layers", default=None, metavar="A-B",
                   help="only record these layers")
    p.add_argument("--trace-view", action="store_true",
                   help="run tools/prefill_trace_view.py --summary after the "
                        "turn")
    return p.parse_args()


def build_env(a):
    env = os.environ.copy()
    if a.instrument or a.instrument_file:
        env["DSV4_PREFILL_INSTRUMENT"] = "2"
    if a.instrument_file:
        env["DSV4_PREFILL_INSTRUMENT_FILE"] = str(
            Path(os.path.expanduser(a.instrument_file)))
    return env


def build_command(a):
    cmd = [
        str(a.binary),
        "--model", str(a.model),
        "--trunk", str(a.trunk),
        "--budget", str(a.budget),
        "--ctx", str(a.ctx),
        "--threads", str(a.threads),
        "--pin", a.pin,
        "--readers", str(a.readers),
        "--reader-pin", a.reader_pin,
        "--stream-miss", str(a.stream_miss),
    ]
    if a.trace:
        # One prefill call per run; select the requested 1-based run.
        cmd += ["--trace", str(a.trace),
                "--trace-phase", "prefill",
                "--trace-from", str(a.trace_run - 1),
                "--trace-tokens", "1"]
        if a.trace_layers:
            cmd += ["--trace-layers", a.trace_layers]
    return cmd


def load_preamble(path):
    if path:
        return Path(os.path.expanduser(str(path))).read_text(encoding="utf-8")

    # The preamble must match the real TUI. Import it from the package so its
    # relative imports keep working.
    return tui_system_prompt()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def send(proc, obj):
    proc.stdin.write(proto.encode_command(obj) + "\n")
    proc.stdin.flush()


def read_event(proc):
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        ev = proto.parse_event(line)
        if ev is None:
            print(f"prefill: bad event: {line}", file=sys.stderr)
            return None
        return ev


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


def generate(proc, tok, prompt, preamble, label, progress):
    tokens = tok.turn_tokens(prompt, system=preamble, first=True)
    send(proc, {"cmd": "generate", "tokens": tokens, "gen": 1,
                "eos": tok.eos_id})
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
            frac = float(ev.get("progress", 0.0))
            if progress.tty:
                progress.update(f"{label}  prefill {100.0 * frac:3.0f}%")
        elif kind == "token":
            tokens.append(ev)
        elif kind == "done":
            if start is None:
                raise RuntimeError("done without start")
            progress.finish(f"{label}  done {len(tokens)} token(s)")
            return start, tokens, ev
        elif kind == "error":
            raise RuntimeError(ev.get("message", "engine error"))


def decode_text(tok, tokens):
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    out = []
    for t in tokens:
        out.append(dec.decode(tok.token_bytes(int(t.get("id") or 0))))
    out.append(dec.decode(b"", final=True))
    return "".join(out)


def token_hash(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def metrics(tok, start, tokens, done):
    if len(tokens) != 1:
        raise RuntimeError(f"expected one token, got {len(tokens)}")

    prompt_tokens = int(start.get("prompt_tokens") or done.get("prompt_tokens") or 0)
    prefill_tokens = int(start.get("prefill_tokens") or 0)
    prefill_s = float(tokens[0].get("prefill_s", 0.0))
    ttft_s = float(tokens[0].get("ttft", tokens[0].get("t", 0.0)))
    prof = done.get("prof", {}) or {}
    head_s = float(prof.get("head", 0.0))
    setup_s = max(ttft_s - prefill_s - head_s, 0.0)
    prefill_tps = prompt_tokens / prefill_s if prefill_s > 0 else 0.0
    prefill_ms_tok = 1000.0 * prefill_s / prompt_tokens if prompt_tokens else 0.0

    c0 = start.get("cache", {}) or {}
    c1 = done.get("cache", {}) or {}
    hits = int(c1.get("hit", 0)) - int(c0.get("hit", 0))
    misses = int(c1.get("miss", 0)) - int(c0.get("miss", 0))
    read_gb = float(c1.get("read_gb", 0.0)) - float(c0.get("read_gb", 0.0))
    hit_rate = hits / (hits + misses) if hits + misses else 0.0

    t0 = start.get("trunk", {}) or {}
    t1 = done.get("trunk", {}) or {}
    trunk_read_gb = float(t1.get("read_gb", 0.0)) - float(t0.get("read_gb", 0.0))

    return {
        "prompt_tokens": prompt_tokens,
        "prefill_tokens": prefill_tokens,
        "prefill_s": prefill_s,
        "prefill_ms_tok": prefill_ms_tok,
        "prefill_tps": prefill_tps,
        "ttft_s": ttft_s,
        "head_s": head_s,
        "setup_s": setup_s,
        "cache_hit_rate": hit_rate,
        "cache_read_gb": read_gb,
        "trunk_read_gb": trunk_read_gb,
        "generated": int(done.get("generated", 0)),
        "stop": done.get("stop", "?"),
        "token_ids": [int(t["id"]) for t in tokens],
        "token_id": int(tokens[0]["id"]),
        "text": decode_text(tok, tokens),
        "sys": done.get("sys", {}) or {},
        "prof": prof,
    }


def print_table(rows):
    print(f"{'run':<8} {'tokens':>7} {'prefill s':>10} {'ms/tok':>8} "
          f"{'prefill tok/s':>14} {'ttft s':>8} {'head s':>7} "
          f"{'hit%':>7} {'read GB':>9} {'token':>7}")
    for name, m in rows:
        print(f"{name:<8} {m['prompt_tokens']:7d} {m['prefill_s']:10.3f} "
              f"{m['prefill_ms_tok']:8.2f} {m['prefill_tps']:14.2f} "
              f"{m['ttft_s']:8.3f} {m['head_s']:7.3f} "
              f"{100.0 * m['cache_hit_rate']:7.1f} {m['cache_read_gb']:9.2f} "
              f"{m['token_id']:7d}")


def main():
    a = parse_args()
    if a.runs < 2:
        print("prefill: --runs must be at least 2 (one warm-up plus a measured run)",
              file=sys.stderr)
        return 2
    if a.no_golden and a.update_golden:
        print("prefill: --no-golden and --update-golden are mutually exclusive",
              file=sys.stderr)
        return 2
    if a.trace and not 1 <= a.trace_run <= a.runs:
        print("prefill: --trace-run must be within 1..--runs", file=sys.stderr)
        return 2
    if not Path(a.binary).is_file():
        print(f"prefill: {a.binary} not found; run `make serve` first",
              file=sys.stderr)
        return 2

    if a.instrument_file:
        a.instrument_file = str(Path(os.path.expanduser(a.instrument_file)))
        Path(a.instrument_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            Path(a.instrument_file).unlink()
        except FileNotFoundError:
            pass
    elif a.instrument:
        print("prefill: instrumentation goes to serve stderr; use "
              "--instrument-file or --detailed to keep it", file=sys.stderr)

    try:
        preamble = load_preamble(a.preamble_file)
    except (OSError, RuntimeError) as exc:
        print(f"prefill: {exc}", file=sys.stderr)
        return 2
    try:
        tok = Tokenizer(a.tok)
    except (OSError, ValueError) as exc:
        print(f"prefill: cannot load tokenizer {a.tok}: {exc}", file=sys.stderr)
        return 2
    preamble_sha = sha256_text(preamble)

    progress = Progress(enabled=not a.detailed and not a.as_json)
    errfile = None
    if not a.detailed or a.as_json:
        errfile = tempfile.TemporaryFile(mode="w+", encoding="utf-8")

    if a.detailed and not a.as_json:
        print(f"prefill: model   {a.model}", file=sys.stderr)
        print(f"prefill: trunk   {a.trunk}", file=sys.stderr)
        print(f"prefill: binary  {a.binary}", file=sys.stderr)
        print(f"prefill: budget={a.budget:g} GB ctx={a.ctx} runs={a.runs} "
              f"(run 1 warm-up)", file=sys.stderr)
        print(f"prefill: preamble {len(preamble)} chars, "
              f"user prompt {len(a.prompt)} chars", file=sys.stderr)
        if a.instrument_file:
            print(f"prefill: instrument {a.instrument_file}", file=sys.stderr)

    progress.update("loading model", force=True)
    proc = subprocess.Popen(build_command(a), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=errfile,
                            text=True, encoding="utf-8", bufsize=1,
                            env=build_env(a))
    try:
        ready = wait_for(proc, "ready")
        if ready is None:
            progress.finish("failed before ready")
            print("prefill: serve exited before ready", file=sys.stderr)
            dump_stderr(errfile)
            return 2
        progress.update("model ready", force=True)

        results = []
        for i in range(a.runs):
            if i:
                send(proc, {"cmd": "reset"})
                if wait_for(proc, "reset") is None:
                    progress.finish("failed during reset")
                    print("prefill: serve exited during reset", file=sys.stderr)
                    dump_stderr(errfile)
                    return 2
            label = f"run {i + 1}/{a.runs}" + (" (warmup)" if i == 0 else "")
            start, tokens, done = generate(proc, tok, a.prompt, preamble,
                                           label, progress)
            results.append(metrics(tok, start, tokens, done))

        send(proc, {"cmd": "quit"})
    except (RuntimeError, BrokenPipeError) as exc:
        progress.finish("failed")
        print(f"prefill: {exc}", file=sys.stderr)
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

    # A per-token fallback is a different code path; its timing is not the
    # block-prefill number this benchmark exists to measure.
    for i, r in enumerate(results):
        if r["prefill_tokens"] != r["prompt_tokens"]:
            print(f"prefill: run {i + 1} used the per-token fallback "
                  f"(prefill_tokens={r['prefill_tokens']}, "
                  f"prompt_tokens={r['prompt_tokens']}); timing is not comparable",
                  file=sys.stderr)
            return 2

    first = results[0]["token_ids"]
    for i, r in enumerate(results[1:], start=2):
        ids = r["token_ids"]
        if ids != first:
            at = next((j for j, (x, y) in enumerate(zip(first, ids)) if x != y),
                      min(len(first), len(ids)))
            print(f"prefill: run {i} diverges from run 1 at token {at}: "
                  f"{first[at:at+5]} vs {ids[at:at+5]}", file=sys.stderr)
            return 1

    prompt_tokens = results[0]["prompt_tokens"]
    if (prompt_tokens != TARGET_TOKENS and a.prompt == USER_PROMPT
            and not a.preamble_file):
        print(f"prefill: warning: workload is {prompt_tokens} tokens, expected "
              f"{TARGET_TOKENS}; the TUI tools preamble changed",
              file=sys.stderr)

    ids = first
    digest = token_hash(ids)
    golden_path = Path(os.path.expanduser(str(a.golden)))
    golden = {
        "model": str(a.model),
        "system_prompt_sha256": preamble_sha,
        "prompt": a.prompt,
        "prompt_tokens": prompt_tokens,
        "target_tokens": TARGET_TOKENS,
        "gen": 1,
        "budget_gb": a.budget,
        "ctx": a.ctx,
        "token_ids": ids,
        "text": results[0]["text"],
        "sha256": digest,
    }

    status = "SKIPPED"
    if not a.no_golden:
        existed = golden_path.is_file()
        if existed and not a.update_golden:
            old = json.loads(golden_path.read_text(encoding="utf-8"))
            if (old.get("prompt") != a.prompt
                    or old.get("system_prompt_sha256") != preamble_sha):
                print("prefill: golden was recorded for a different prompt or "
                      "preamble; use --update-golden if that change was "
                      "intentional", file=sys.stderr)
                return 1
            if old.get("prompt_tokens") != prompt_tokens:
                print(f"prefill: GOLDEN MISMATCH prompt_tokens: "
                      f"golden {old.get('prompt_tokens')} vs now {prompt_tokens}",
                      file=sys.stderr)
                return 1
            if old.get("token_ids") != ids:
                old_ids = old.get("token_ids", [])
                at = next((j for j, (x, y) in enumerate(zip(old_ids, ids))
                           if x != y), min(len(old_ids), len(ids)))
                print(f"prefill: GOLDEN MISMATCH at token {at}: "
                      f"golden {old_ids[at:at+5]} vs now {ids[at:at+5]}",
                      file=sys.stderr)
                return 1
            status = "PASS (golden)"
        else:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(
                json.dumps(golden, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            status = "GOLDEN UPDATED" if existed else "GOLDEN CREATED"

    measured = results[1:]
    rows = [("warmup", results[0])]
    for i, r in enumerate(measured, start=1):
        rows.append((f"run {i}", r))
    median = {
        "prompt_tokens": measured[0]["prompt_tokens"],
        "prefill_s": statistics.median(r["prefill_s"] for r in measured),
        "prefill_ms_tok": statistics.median(r["prefill_ms_tok"] for r in measured),
        "prefill_tps": statistics.median(r["prefill_tps"] for r in measured),
        "ttft_s": statistics.median(r["ttft_s"] for r in measured),
        "head_s": statistics.median(r["head_s"] for r in measured),
        "setup_s": statistics.median(r["setup_s"] for r in measured),
        "cache_hit_rate": statistics.median(r["cache_hit_rate"] for r in measured),
        "cache_read_gb": statistics.median(r["cache_read_gb"] for r in measured),
        "trunk_read_gb": statistics.median(r["trunk_read_gb"] for r in measured),
        "token_id": measured[0]["token_id"],
    }
    rows.append(("median", median))

    if a.as_json:
        print(json.dumps({
            "args": {
                "budget_gb": a.budget, "ctx": a.ctx, "runs": a.runs,
                "prompt": a.prompt, "target_tokens": TARGET_TOKENS,
                "preamble_sha256": preamble_sha,
            },
            "runs": [{k: v for k, v in r.items() if k != "text"}
                     for r in results],
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
                print(f"  {label}: prefill {m['prompt_tokens']} tok in "
                      f"{m['prefill_s']:.3f} s ({m['prefill_tps']:.2f} tok/s, "
                      f"{m['prefill_ms_tok']:.2f} ms/tok), "
                      f"ttft {m['ttft_s']:.3f} s, head {m['head_s']:.3f} s, "
                      f"setup {m['setup_s']:.3f} s, "
                      f"token {m['token_id']} ({m['stop']})")
                p = m.get("prof", {})
                if p:
                    keys = ("hc", "attn", "idxer", "gate", "expert_io",
                            "expert_mm", "shared", "head", "bind", "trunk_io")
                    print("    prof: " + " ".join(
                        f"{k}={p.get(k, 0.0):.3f}s" for k in keys))
                print(f"    cache: {100.0 * m['cache_hit_rate']:.1f}% hit, "
                      f"{m['cache_read_gb']:.2f} GB read; "
                      f"trunk {m['trunk_read_gb']:.2f} GB")
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

    if a.trace:
        print(f"trace: {a.trace} (run {a.trace_run})", file=sys.stderr)
        if a.trace_view:
            view = REPO / "tools" / "prefill_trace_view.py"
            subprocess.run([sys.executable, str(view), str(a.trace),
                            "--summary"], check=False)

    if len(measured) >= 2:
        for key in ("prefill_tps", "ttft_s"):
            vals = [r[key] for r in measured]
            lo, hi = min(vals), max(vals)
            if lo > 0 and (hi - lo) / lo > 0.05:
                print(f"warning: {key} varies {100.0 * (hi - lo) / lo:.1f}% "
                      f"across measured runs ({lo:.3f} .. {hi:.3f})",
                      file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

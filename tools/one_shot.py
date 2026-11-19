#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one prompt through bin/dsv4-serve and print the full diagnostics.

Serve is the resident engine: it loads the model once, speaks the line protocol
on stdin/stdout, and keeps the KV state alive. This script is the one-shot
client: it tokenizes the prompt, starts serve, sends one generate command,
streams the text, prints every counter serve reports, then quits.

Examples:
    python3 tools/one_shot.py "The capital of France is" --gen 16
    python3 tools/one_shot.py --prompt "Write a quicksort." --think --gen 128
    python3 tools/one_shot.py "hello" --top --ids --layers
    python3 tools/one_shot.py "hello" --json > events.jsonl
"""
import argparse
import codecs
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO / "tui"))
from dsv4tui import proto  # noqa: E402
from dsv4tui.tokenizer import Tokenizer  # noqa: E402

DEFAULT_MODEL = HOME / "colibri-models" / "deepseek-v4-flash"
DEFAULT_TRUNK = HOME / "colibri-models" / "dsv4-trunk-flash"
DEFAULT_TOK = HOME / "colibri-models" / "dsv4-tok-flash.bin"
DEFAULT_BINARY = REPO / "bin" / "dsv4-serve"


def env_path(name, default):
    return Path(os.path.expanduser(os.environ.get(name, str(default))))


def parse_args():
    p = argparse.ArgumentParser(
        description="One-shot prompt runner and diagnostics printer for dsv4-serve.")
    p.add_argument("prompt", nargs="?", help="prompt text (or --prompt, or stdin)")
    p.add_argument("--prompt", dest="prompt_opt", help="prompt text")
    p.add_argument("--model", default=env_path("DSV4_MODEL", DEFAULT_MODEL))
    p.add_argument("--trunk", default=env_path("DSV4_TRUNK", DEFAULT_TRUNK))
    p.add_argument("--tok", default=env_path("DSV4_TOK", DEFAULT_TOK))
    p.add_argument("--binary", default=env_path("DSV4_SERVE_BIN", DEFAULT_BINARY))
    p.add_argument("--budget", default="auto", help="GB or auto (default: auto)")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--gen", type=int, default=32)
    p.add_argument("--system", help="system prompt for this turn")
    p.add_argument("--think", action="store_true", help="open a think block")
    p.add_argument("--layers", action="store_true", help="show per-layer events")
    p.add_argument("--top", action="store_true", help="show top-5 alternatives")
    p.add_argument("--ids", action="store_true", help="show generated token ids")
    p.add_argument("--json", action="store_true", dest="raw_json",
                   help="print parsed serve events as JSON instead of the "
                        "human report")
    p.add_argument("--quiet", action="store_true", help="suppress the human report")
    p.add_argument("--threads", type=int)
    p.add_argument("--pin", choices=("off", "all", "p", "e"))
    p.add_argument("--readers", type=int)
    p.add_argument("--reader-pin", choices=("off", "all", "p", "e"))
    p.add_argument("--stream-miss", type=int)
    args = p.parse_args()

    if args.prompt and args.prompt_opt:
        p.error("give the prompt either positionally or with --prompt, not both")
    prompt = args.prompt if args.prompt is not None else args.prompt_opt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read()
        else:
            p.error("no prompt given")
    args.prompt = prompt
    return args


def build_command(a):
    cmd = [str(a.binary), "--model", str(a.model),
           "--trunk", str(a.trunk),
           "--budget", str(a.budget), "--ctx", str(a.ctx)]
    for flag, value in (("--threads", a.threads), ("--pin", a.pin),
                        ("--readers", a.readers), ("--reader-pin", a.reader_pin),
                        ("--stream-miss", a.stream_miss)):
        if value is not None:
            cmd += [flag, str(value)]
    return cmd


def pct(x):
    return f"{100.0 * x:5.1f}%"


def num(d, key, default=0.0):
    v = d.get(key, default)
    return v if isinstance(v, (int, float)) else default


def row(label, value):
    print(f"  {label:<14} {value}", file=sys.stderr)


def announce(a, text):
    if not a.quiet:
        print(f"[one-shot] {text}", file=sys.stderr)


class Client:
    def __init__(self, args):
        self.a = args
        self.proc = None
        self.tok: Tokenizer | None = None
        self.ready = None
        self.start = None
        self.done = None
        self.error = None
        self.interrupted = False
        self.tokens = []
        self.last_layer = None
        self.last_prefill = -1.0
        self.last_stage = None

    def send(self, obj):
        try:
            self.proc.stdin.write(proto.encode_command(obj) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None, None
            line = line.strip()
            if not line:
                continue
            ev = proto.parse_event(line)
            if ev is None:
                announce(self.a, f"bad event from serve: {line}")
                continue
            return ev, line

    def on_sigint(self, _sig, _frame):
        self.interrupted = True
        if self.proc is None or self.proc.poll() is not None:
            return
        if self.ready is None:
            self.proc.terminate()
        else:
            self.proc.send_signal(signal.SIGUSR1)

    def ready_event(self, ev):
        self.ready = ev
        if self.a.raw_json or self.a.quiet:
            return
        cfg = ev.get("config", {})
        ctx = ev.get("context", {})
        cache = ev.get("cache", {})
        trunk = ev.get("trunk", {})
        announce(self.a, f"model:   {ev.get('model')}")
        announce(self.a, f"threads: {ev.get('threads')}")
        announce(self.a, f"config:  {cfg.get('layers')} layers, hidden "
                         f"{cfg.get('hidden')}, vocab {cfg.get('vocab')}, "
                         f"experts {cfg.get('experts')}, top-{cfg.get('topk')}")
        announce(self.a, f"context: {ctx.get('cap')} tokens")
        announce(self.a, f"cache:   {cache.get('slots')} slots x "
                         f"{num(cache, 'slot_mb'):.2f} MB = "
                         f"{num(cache, 'resident_gb'):.2f} GB")
        announce(self.a, f"trunk:   {trunk.get('pinned')}/"
                         f"{trunk.get('layers')} pinned, "
                         f"{num(trunk, 'resident_gb'):.2f} GB")

    def start_event(self, ev):
        self.start = ev
        if self.a.raw_json or self.a.quiet:
            return
        announce(self.a, f"prompt:  {ev.get('prompt_tokens')} tokens, "
                         f"prefill {ev.get('prefill_tokens')} batched, "
                         f"mode {ev.get('mode')}")

    def token_event(self, ev, decoder):
        self.tokens.append(ev)
        if self.a.raw_json:
            return
        raw = self.tok.token_bytes(int(ev.get("id") or 0)) if self.tok else b""
        text = decoder.decode(raw)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
        if self.a.ids:
            print(f"  token {ev.get('index'):4d}  id={ev.get('id'):6d}  "
                  f"pos={ev.get('pos'):4d}  t={num(ev, 't'):.3f}s",
                  file=sys.stderr)
        if self.a.top:
            top = ev.get("top") or []
            parts = [f"{t[0]}:{t[1]:.3f}" for t in top if len(t) >= 2]
            if parts:
                print("       top " + "  ".join(parts), file=sys.stderr)

    def layer_event(self, ev):
        self.last_layer = ev
        if not self.a.layers or self.a.raw_json or self.a.quiet:
            return
        cache = ev.get("cache", {})
        sysd = ev.get("sys", {})
        print(f"  layer {ev.get('layer') + 1:2d}/{ev.get('layers')} "
              f"pos={ev.get('pos'):4d} {ev.get('phase'):6s} "
              f"t={num(ev, 't'):8.3f}s  hit={pct(num(cache, 'hit_rate'))}  "
              f"disk={num(sysd, 'io_read_gb'):6.2f}GB", file=sys.stderr)

    def report(self):
        if self.a.raw_json or self.a.quiet or not self.done:
            return
        d = self.done
        cfg = (self.ready or {}).get("config", {})
        ctx = (self.ready or {}).get("context", {})
        sysd = d.get("sys", {})
        cache = d.get("cache", {})
        trunk = d.get("trunk", {})
        prof = d.get("prof", {})
        secs = num(d, "seconds")
        gen = int(num(d, "generated"))
        ttft = num(d, "ttft")
        pre = num(self.tokens[-1], "prefill_s") if self.tokens else 0.0

        print("\n\n--- diagnostics ---", file=sys.stderr)
        row("model", (self.ready or {}).get("model", ""))
        row("threads", (self.ready or {}).get("threads", ""))
        row("config", f"{cfg.get('layers')} layers, hidden {cfg.get('hidden')}, "
                      f"vocab {cfg.get('vocab')}, experts {cfg.get('experts')}, "
                      f"top-{cfg.get('topk')}, mHC x{cfg.get('hc_mult')}")
        row("context", f"{d.get('context_used')}/{ctx.get('cap')} tokens used")

        print("\n  -- turn --", file=sys.stderr)
        row("prompt", f"{d.get('prompt_tokens')} tokens "
                      f"({self.start.get('prefill_tokens') if self.start else 0} batched)")
        row("generated", f"{gen} tokens, stop={d.get('stop')}")
        row("wall", f"{secs:.3f} s")
        row("ttft", f"{ttft:.3f} s")
        row("prefill", f"{pre:.3f} s")
        row("overall", f"{num(d, 's_per_token'):.3f} s/token, "
                       f"{num(d, 'tps'):.3f} tok/s")
        if gen > 1 and len(self.tokens) > 1:
            dec = num(self.tokens[-1], "t") - num(self.tokens[0], "t")
            row("decode", f"{dec / (gen - 1):.3f} s/token, "
                          f"{gen / dec if dec > 0 else 0:.3f} tok/s")

        print("\n  -- process --", file=sys.stderr)
        row("rss", f"{num(sysd, 'rss_mb') / 1024:.2f} GB "
                   f"(peak {num(sysd, 'peak_mb') / 1024:.2f} GB)")
        row("disk read", f"{num(sysd, 'io_read_gb'):.2f} GB")
        row("cpu/wall", f"{num(sysd, 'cpu_s'):.1f} s / "
                        f"{num(sysd, 'wall_s'):.1f} s")

        print("\n  -- cache --", file=sys.stderr)
        row("slots", f"{cache.get('slots')} x {num(cache, 'slot_mb'):.2f} MB "
                     f"= {num(cache, 'resident_gb'):.2f} GB")
        row("hits/misses", f"{cache.get('hit')} / {cache.get('miss')} "
                           f"({pct(num(cache, 'hit_rate'))}), "
                           f"evictions {cache.get('evict')}")
        row("disk read", f"{num(cache, 'read_gb'):.2f} GB")
        if cache.get("pf"):
            row("prefetch", f"hit {cache.get('pf_hit')}, load {cache.get('pf_load')}, "
                            f"{num(cache, 'pf_read_gb'):.2f} GB, "
                            f"wait {num(cache, 'pf_wait_ms'):.1f} ms")

        print("\n  -- trunk --", file=sys.stderr)
        row("layers", f"{trunk.get('pinned')}/{trunk.get('layers')} pinned, "
                      f"{trunk.get('slots')} slots")
        row("hits/misses", f"{trunk.get('hit')} / {trunk.get('miss')} "
                           f"({pct(num(trunk, 'hit_rate'))})")
        row("disk read", f"{num(trunk, 'read_gb'):.2f} GB, "
                         f"load {num(trunk, 'load_s'):.3f} s")

        print("\n  -- profile --", file=sys.stderr)
        for key in ("hc", "attn", "gate", "expert_io", "expert_mm",
                    "shared", "head", "bind"):
            v = num(prof, key)
            row(key, f"{v:7.3f} s  {pct(v / secs if secs else 0)}")
        acc = num(prof, "accounted")
        row("accounted", f"{acc:7.3f} s  {pct(acc / secs if secs else 0)}")
        row("unaccounted", f"{max(secs - acc, 0):7.3f} s  "
                           f"{pct(max(secs - acc, 0) / secs if secs else 0)}")

    def run(self):
        a = self.a
        if not Path(a.binary).is_file():
            print(f"error: {a.binary} not found; run `make serve` first",
                  file=sys.stderr)
            return 2

        try:
            self.tok = Tokenizer(a.tok)
        except (OSError, ValueError) as exc:
            print(f"error: cannot load tokenizer {a.tok}: {exc}", file=sys.stderr)
            return 2

        cmd = build_command(a)
        announce(a, f"loading {a.model}")
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding="utf-8", bufsize=1)
        except OSError as e:
            print(f"error: cannot start {a.binary}: {e}", file=sys.stderr)
            return 1

        signal.signal(signal.SIGINT, self.on_sigint)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        for ev, line in iter(self.read, (None, None)):
            if a.raw_json:
                print(json.dumps(ev, ensure_ascii=False), flush=True)
            if ev.get("event") == "ready":
                break
            if ev.get("event") == "error":
                self.error = ev
                self.fail(ev.get("message", "engine error"))
                self.stop()
                return 130 if self.interrupted else 1
        else:
            self.fail("serve exited before ready")
            self.stop()
            return 130 if self.interrupted else 1
        self.ready_event(ev)

        tokens = self.tok.turn_tokens(
            a.prompt, system=a.system, think=bool(a.think), first=True)
        self.send({"cmd": "generate", "tokens": tokens, "gen": a.gen,
                   "eos": self.tok.eos_id,
                   "think": bool(a.think), "layers": bool(a.layers)})

        failed = False
        while True:
            ev, line = self.read()
            if ev is None:
                self.fail("serve exited during generation")
                failed = True
                break
            if a.raw_json:
                print(json.dumps(ev, ensure_ascii=False), flush=True)
            kind = ev.get("event")
            if kind == "start":
                self.start_event(ev)
            elif kind == "token":
                self.token_event(ev, decoder)
            elif kind == "layer":
                self.layer_event(ev)
            elif kind == "done":
                self.done = ev
                break
            elif kind == "error":
                self.error = ev
                failed = True
                break
            elif kind == "prefill":
                prog = num(ev, "progress")
                stage = ev.get("stage")
                if (a.layers and not a.raw_json and not a.quiet
                        and (stage != self.last_stage
                             or prog - self.last_prefill >= 0.05
                             or prog >= 1.0)):
                    print(f"  prefill {stage} "
                          f"{ev.get('tokens_done')}/{ev.get('tokens_total')} "
                          f"{pct(prog)}", file=sys.stderr)
                    self.last_stage = stage
                    self.last_prefill = prog
            # ready/stats/reset/system are informational

        tail = decoder.decode(b"", final=True)
        if tail and not a.raw_json:
            sys.stdout.write(tail)
        if not a.raw_json:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if self.error:
            self.fail(self.error.get("message", "engine error"))
        self.report()

        code = 1 if failed or self.error else 0
        if self.interrupted:
            code = 130
        self.stop()
        return code

    def fail(self, message):
        print(f"[one-shot] error: {message}", file=sys.stderr)

    def stop(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.send({"cmd": "quit"})
                self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def main():
    args = parse_args()
    return Client(args).run()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""dsv4_tui.py - a live terminal front end for deepseek-v4-in-c.

The engine is 1.6-2.2 s/token on the reference machine, so the interesting part
of a chat turn is not only the text: it is how fast prefill hands over to
decode. This script keeps bin/dsv4-serve resident, streams its JSON events, and
paints the conversation next to a decode tok/s timeline and the live memory,
expert-cache and disk rates while the model runs. The panel is still being
reworked: the engine's profiler split and top-k counters are not drawn yet.

Two backends, deliberately:
  * default: spawn bin/dsv4-serve and talk the real protocol.
  * --demo: a synthetic producer with the same events, so the UI can be developed
    on a laptop that is not sitting on a 160 GB checkpoint.

No third-party packages. curses is in the Python standard library.

Usage:
    python3 tui/dsv4_tui.py                    # real model, defaults below
    python3 tui/dsv4_tui.py --demo             # no model needed
    python3 tui/dsv4_tui.py --model DIR --trunk DIR --tok FILE --budget 10
    python3 tui/dsv4_tui.py --budget auto      # size the budget from free RAM

Keys:
    Enter send, /help for commands, PgUp/PgDn scroll, Ctrl-C or /exit quits.
"""
from __future__ import annotations

import argparse
import codecs
import curses
import json
import locale
import math
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from markdown_render import RICH_AVAILABLE, RICH_ERROR, Span, render_markdown
except ImportError:  # running as `python3 -m tui.dsv4_tui`
    from .markdown_render import (RICH_AVAILABLE, RICH_ERROR, Span,
                                  render_markdown)

# ----------------------------------------------------------------- defaults --

HOME = Path.home()
DEFAULT_MODEL = HOME / "colibri-models" / "deepseek-v4-flash"
DEFAULT_TRUNK = HOME / "colibri-models" / "dsv4-trunk-flash"
DEFAULT_TOK = HOME / "colibri-models" / "dsv4-tok-flash.bin"
REPO = Path(__file__).resolve().parent.parent
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


# ------------------------------------------------------------------ backend --


class EventSink:
    """A queue plus a tiny logger for stderr lines from the child."""

    def __init__(self) -> None:
        self.q: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    def event(self, obj: dict) -> None:
        self.q.put(("event", obj))

    def log(self, line: str) -> None:
        self.q.put(("stderr", line))

    def eof(self, rc: Optional[int]) -> None:
        self.q.put(("eof", rc))

    def get_all(self) -> list[tuple[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out


class RealBackend:
    def __init__(self, argv: list[str], sink: EventSink) -> None:
        self.argv = argv
        self.sink = sink
        self.proc: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    self.sink.log(f"bad JSON from engine: {line[:300]}")
                    continue
                self.sink.event(obj)
        except Exception as exc:  # pipe closed, decoding issue, ...
            self.sink.log(f"stdout reader stopped: {exc}")
        finally:
            rc = self.proc.wait() if self.proc else None
            self.sink.eof(rc)

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        try:
            for line in self.proc.stderr:
                self.sink.log(line.rstrip())
        except Exception:
            pass

    def send(self, obj: dict) -> bool:
        if not self.proc or not self.proc.stdin or self.proc.poll() is not None:
            return False
        try:
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def cancel(self) -> bool:
        """Ask the engine to stop at the next token (SIGUSR1 in dsv4_serve)."""
        if not self.proc or self.proc.poll() is not None:
            return False
        try:
            self.proc.send_signal(signal.SIGUSR1)
            return True
        except OSError:
            return False

    def close(self) -> None:
        if not self.proc:
            return
        try:
            self.send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class DemoBackend:
    """Synthetic engine with the real event shape.

    This exists so the TUI is testable without a checkpoint, and so a screenshot
    can be taken on a machine that has never loaded the model. It is NOT a
    fidelity claim about the engine's timings: the numbers are made up the way a
    dashboard's numbers are made up, and labelled as demo in the header.
    """

    def __init__(self, sink: EventSink) -> None:
        self.sink = sink
        self.commands: "queue.Queue[dict]" = queue.Queue()
        self.stop = threading.Event()
        self.cancel_requested = False
        self.rng = random.Random(4)
        self.cache_hits = 0
        self.cache_miss = 0
        self.cache_evict = 0
        self.trunk_hit = 0
        self.trunk_miss = 0
        self.read_gb = 0.0
        self.t0 = time.monotonic()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, obj: dict) -> bool:
        self.commands.put(obj)
        return True

    def cancel(self) -> bool:
        self.cancel_requested = True
        return True

    def close(self) -> None:
        self.stop.set()
        self.commands.put({"cmd": "quit"})

    # -- fake stats --------------------------------------------------------
    def _sys(self, rss_mb: float) -> dict:
        now = time.monotonic()
        return {
            "rss_mb": rss_mb,
            "peak_mb": rss_mb + 512.0,
            "io_read_gb": self.read_gb,
            "cpu_s": (now - self.t0) * 0.82,
            "wall_s": now - self.t0,
        }

    def _cache(self) -> dict:
        n = self.cache_hits + self.cache_miss
        return {
            "slots": 292,
            "hit": self.cache_hits,
            "miss": self.cache_miss,
            "evict": self.cache_evict,
            "hit_rate": (self.cache_hits / n) if n else 0.0,
            "read_gb": self.read_gb * 0.78,
            "resident_gb": 3.55,
            "slot_mb": 12.77,
        }

    def _trunk(self) -> dict:
        n = self.trunk_hit + self.trunk_miss
        return {
            "layers": 43,
            "pinned": 43,
            "slots": 1,
            "slot_mb": 165.48,
            "hit": self.trunk_hit,
            "miss": self.trunk_miss,
            "hit_rate": (self.trunk_hit / n) if n else 0.0,
            "read_gb": 6.27,
            "load_s": 0.0,
            "resident_gb": 6.43,
        }

    def _prof(self, elapsed: float) -> dict:
        # Proportions roughly matching BENCHMARK.md: disk is the biggest single
        # component, then attention, then expert matmul.
        p = {
            "hc": elapsed * 0.04,
            "attn": elapsed * 0.21,
            "gate": elapsed * 0.01,
            "expert_io": elapsed * 0.38,
            "expert_mm": elapsed * 0.20,
            "shared": elapsed * 0.05,
        }
        p["accounted"] = sum(p.values())
        return p

    def _base(self, elapsed: float = 0.0, rss: float = 2060.0) -> dict:
        return {
            "sys": self._sys(rss),
            "cache": self._cache(),
            "trunk": self._trunk(),
            "prof": self._prof(elapsed),
        }

    def _event(self, name: str, elapsed: float = 0.0, **fields: Any) -> None:
        obj = {"event": name}
        obj.update(fields)
        obj.update(self._base(elapsed))
        self.sink.event(obj)

    # -- command loop ------------------------------------------------------
    def _run(self) -> None:
        time.sleep(0.35)
        self.sink.event({
            "event": "ready",
            "model": "demo-model/deepseek-v4-flash",
            "config": {
                "hidden": 4096, "layers": 43, "vocab": 129280,
                "heads": 64, "head_dim": 512, "experts": 256, "topk": 6,
                "moe_inter": 2048, "sliding_window": 128, "index_topk": 512,
                "hc_mult": 4, "max_pos": 1048576,
                "compress_ratios": [0, 0, 4, 128] * 10 + [0, 0, 0],
            },
            "context": {"used": 0, "cap": 4096},
            "gen_default": 256, "think_default": 0,
            **self._base(),
        })

        while not self.stop.is_set():
            try:
                cmd = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue
            name = cmd.get("cmd")
            if name == "quit":
                break
            if name == "system":
                self.sink.event({"event": "system", "ok": True,
                                 "chars": len(cmd.get("text", ""))})
            elif name == "reset":
                self._event("reset")
            elif name == "stats":
                self._event("ready")
            elif name == "ping":
                self.sink.event({"event": "pong"})
            elif name == "generate":
                self._generate(cmd)
        self.sink.eof(0)

    # -- generation --------------------------------------------------------
    def _response(self, prompt: str, think: bool) -> str:
        low = prompt.lower()
        if "code" in low or "c function" in low or "program" in low:
            return (
                "Here is a small C function that adds two integers, with the "
                "kind of checks this repository cares about:\n\n"
                "```c\n"
                "#include <limits.h>\n\n"
                "int add_checked(int a, int b, int *out)\n"
                "{\n"
                "    if ((b > 0 && a > INT_MAX - b) ||\n"
                "        (b < 0 && a < INT_MIN - b)) return 0;\n"
                "    *out = a + b;\n"
                "    return 1;\n"
                "}\n"
                "```\n\n"
                "The overflow check is the part that matters: signed overflow is "
                "undefined behaviour in C, not a wrap.\n"
            )
        if "hello" in low or "hi" in low[:8]:
            return (
                "Hello. This is the demo backend, so no NVMe was harmed. The "
                "numbers in the right-hand panel are synthetic but have the same "
                "shape as the real engine's events: per-layer progress, expert "
                "cache accounting, trunk reads, RSS, disk bytes and a profile "
                "split into I/O and matmul.\n"
            )
        return (
            f'You said: "{prompt.strip()}"\n\n'
            "This is the synthetic backend. Start without --demo and point the "
            "TUI at a checkpoint to stream real tokens from DeepSeek-V4-Flash. "
            "The panel on the right will then show the actual expert-cache hit "
            "rate and the disk traffic, which is where most of the wall clock "
            "goes at this model size.\n"
        )

    def _generate(self, cmd: dict) -> None:
        prompt = cmd.get("prompt") or ""
        ngen = int(cmd.get("gen") or 128)
        think = bool(cmd.get("think"))
        layers = bool(cmd.get("layers", True))
        mode = "thinking" if think else "chat"
        prompt_tokens = 9
        self.cancel_requested = False
        stop = "max"

        self.sink.event({
            "event": "start", "pos": 0, "prompt_tokens": prompt_tokens,
            "gen_max": ngen, "mode": mode, "prefill_batch": 0,
            "context_cap": 4096, **self._base(),
        })

        text = self._response(prompt, think)
        # Split into plausible token-sized chunks, keeping whitespace attached.
        chunks = re.findall(r"\s+|[^\s]+", text)
        t0 = time.monotonic()
        ttft = 0.0
        produced = 0
        context = prompt_tokens

        def forward(pos: int, phase: str, t_start: float) -> None:
            """Simulate one forward pass, including its per-layer events."""
            time.sleep(0.012)
            for L in range(43):
                if L % 8 == 0:
                    time.sleep(0.002)
                if self.rng.random() < 0.55:
                    self.cache_hits += 1
                else:
                    self.cache_miss += 1
                    if self.rng.random() < 0.8:
                        self.cache_evict += 1
                self.trunk_hit += 1
                self.read_gb += 0.00018
                if layers:
                    # The engine stamps every layer of a forward with the time
                    # the forward *started*, so the TUI can difference prompt
                    # positions into a prefill timeline.
                    self._event("layer", t_start, pos=pos, phase=phase,
                                layer=L, layers=43, t=t_start)

        # Prefill: one forward per prompt token, then the first token event.
        for p in range(prompt_tokens):
            if self.stop.is_set():
                return
            forward(p, "prompt", time.monotonic() - t0)
        ttft = time.monotonic() - t0

        for i, chunk in enumerate(chunks[:ngen]):
            if self.stop.is_set():
                return
            # A token takes ~2.1 s on the reference machine. The demo runs at
            # 1/12 speed so a conversation is watchable.
            if i > 0:
                forward(context - 1, "gen", time.monotonic() - t0)
            if self.cancel_requested:
                stop = "cancel"
                break

            elapsed = time.monotonic() - t0
            self.read_gb += 0.00055
            p = max(0.02, 0.97 - i * 0.004)
            top = [
                [101 + i, p, 31.0 - i * 0.05],
                [102 + i, (1 - p) * 0.5, 29.0],
                [103 + i, (1 - p) * 0.3, 28.0],
                [104 + i, (1 - p) * 0.2, 27.0],
            ]
            self.sink.event({
                "event": "token", "id": 1000 + i,
                "hex": chunk.encode("utf-8").hex(), "index": i,
                "pos": context, "t": elapsed, "ttft": ttft,
                "prefill_s": ttft, "prefill_tokens": prompt_tokens,
                "top": top, **self._base(elapsed),
            })
            context += 1
            produced += 1

        elapsed = time.monotonic() - t0
        self.sink.event({
            "event": "done", "prompt_tokens": prompt_tokens,
            "generated": produced, "seconds": elapsed,
            "tps": produced / elapsed if elapsed else 0.0,
            "s_per_token": elapsed / produced if produced else 0.0,
            "ttft": ttft, "stop": stop, "context_used": context,
            **self._base(elapsed),
        })


# ------------------------------------------------------------------ display --


@dataclass
class Message:
    role: str                     # user | assistant | system | error
    text: str = ""
    think: bool = False
    meta: dict = field(default_factory=dict)
    md_cache_key: tuple = ()
    md_cache_lines: list[list[Span]] = field(default_factory=list)


class App:
    def __init__(self, args: argparse.Namespace, sink: EventSink) -> None:
        self.args = args
        self.sink = sink
        self.backend: Any = None

        self.messages: list[Message] = []
        self.current_msg: Optional[Message] = None
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

        self.ready: dict = {}
        self.last: dict = {}
        self.prev_snap: Optional[dict] = None
        self.prev_snap_t = 0.0
        self.last_stats_ingest = 0.0
        self.stats_interval = 1.0
        self.stats_cache: list[list[tuple[str, str]]] = []
        self.stats_cache_ts = 0.0
        self.stats_cache_width = -1
        self.stats_cache_height = -1
        self.stats_force = True
        self.io_rate = 0.0
        self.cache_rate = 0.0
        self.trunk_rate = 0.0
        self.mem_available_gb = 0.0
        self.cpu_pct = 0.0
        self.cpu_prev: Optional[dict[int, tuple[int, int]]] = None
        self.cpu_prev_t = 0.0
        self.core_pct: list[Optional[float]] = [None, None, None, None]
        self.cpu_temp_c: Optional[float] = None
        self.cpu_temp_path: Optional[str] = None
        self.cpu_temp_checked = False

        self.status = "starting"
        self.status_detail = ""
        self.generating = False
        self.cancel_pending = False
        self.cancel_rect: Optional[tuple[int, int, int]] = None
        self.backend_alive = True
        self.generation = 0
        self.gen_max = args.gen
        self.prompt_tokens = 0
        self.context_used = 0
        self.context_cap = args.ctx
        self.turn_start = 0.0
        self.last_token_t = 0.0
        # One duration per decoded token, in turns.
        self.decode_spt: deque[float] = deque(maxlen=4096)
        self.prefill_batch = int(getattr(args, "prefill_batch_size", 0) or 0)
        self.prefill_progress: dict = {}
        self.prefill_active = False
        self.prefill_s = 0.0
        self.prefill_tokens = 0
        self.ttft = 0.0
        self.current_layer = 0
        self.phase = ""
        self.top: list = []
        self.last_done: dict = {}
        self.mode = "chat"

        self.system_prompt = args.system or ""
        self.think = bool(args.think)
        self.gen_limit = int(args.gen)
        self.show_layers = not args.no_layers

        self.input = ""
        self.input_cursor = 0
        self.scroll = 0
        self.spinner = 0
        self.stderr_lines: deque[str] = deque(maxlen=8)
        self.running = True
        self.dirty = True
        self.last_render = 0.0
        self.demo = bool(args.demo)
        self.pairs: dict[str, int] = {}
        self.markdown_enabled = not args.no_markdown
        self.md_warned = False
        self._pair_cache: dict[tuple[int, int], int] = {}
        self._color_cache: dict[Any, int] = {}
        self._next_pair = 1

    # -- backend lifecycle -------------------------------------------------
    def start_backend(self) -> None:
        if self.demo:
            self.backend = DemoBackend(self.sink)
        else:
            budget = (self.args.budget if isinstance(self.args.budget, str)
                      else f"{self.args.budget:.2f}")
            argv = [
                str(self.args.binary),
                str(self.args.model),
                "--trunk", str(self.args.trunk),
                "--tok", str(self.args.tok),
                "--budget", budget,
                "--ctx", str(self.args.ctx),
                "--gen", str(self.args.gen),
            ]
            # None means "let the engine use its default", so the TUI follows
            # the same thread policy as bin/dsv4 without duplicating it here.
            if self.args.threads is not None:
                argv += ["--threads", str(self.args.threads)]
            if self.args.pin is not None:
                argv += ["--pin", self.args.pin]
            if self.args.readers is not None:
                argv += ["--readers", str(self.args.readers)]
            if self.args.reader_pin is not None:
                argv += ["--reader-pin", self.args.reader_pin]
            # The TUI defaults to a small dense prefill batch. 0 falls back to
            # the per-token decode path; the serve flag is the same one the
            # one-shot CLI uses.
            pb = getattr(self.args, "prefill_batch_size", 0)
            if pb is not None:
                argv += ["--prefill-batch-size", str(pb)]
            if self.args.think:
                argv.append("--think")
            if self.args.no_layers:
                argv.append("--no-layers")
            self.backend = RealBackend(argv, self.sink)
        self.backend.start()
        if self.system_prompt:
            self.backend.send({"cmd": "system", "text": self.system_prompt})

    def close(self) -> None:
        if self.backend:
            self.backend.close()

    # -- event handling ----------------------------------------------------
    def drain(self) -> None:
        for kind, payload in self.sink.get_all():
            if kind == "event":
                self.handle_event(payload)
            elif kind == "stderr":
                self.stderr_lines.append(str(payload))
                self.dirty = True
            elif kind == "log":
                self.stderr_lines.append(str(payload))
                self.dirty = True
            elif kind == "eof":
                self.backend_alive = False
                self.generating = False
                if self.running:
                    self.status = "failed"
                    tail = ""
                    if self.stderr_lines:
                        tail = "  " + " | ".join(list(self.stderr_lines)[-3:])
                    self.messages.append(Message(
                        "error", f"engine exited (code {payload}).{tail}"))
                    self.current_msg = None
                self.dirty = True

    def ingest_stats(self, ev: dict) -> None:
        sysd = ev.get("sys")
        if not sysd:
            return
        now = time.monotonic()
        cache = ev.get("cache") or {}
        trunk = ev.get("trunk") or {}
        if self.prev_snap is not None:
            dt = now - self.prev_snap_t
            # A forced update (start/done) can land milliseconds after the last
            # sample. Rates over that window are noise, not information; keep
            # the previous ones until a real sampling interval has passed. Do
            # not move the baseline meanwhile, or a stream of sub-0.2 s prefill
            # events would reset the window forever and the disk rate would
            # never be recomputed.
            if dt >= 0.2:
                ps = self.prev_snap.get("sys") or {}
                pc = self.prev_snap.get("cache") or {}
                pt = self.prev_snap.get("trunk") or {}
                self.io_rate = max(0.0, (sysd.get("io_read_gb", 0.0)
                                         - ps.get("io_read_gb", 0.0)) / dt)
                self.cache_rate = max(0.0, (cache.get("read_gb", 0.0)
                                            - pc.get("read_gb", 0.0)) / dt)
                self.trunk_rate = max(0.0, (trunk.get("read_gb", 0.0)
                                            - pt.get("read_gb", 0.0)) / dt)
                self.cpu_pct = clamp(
                    (sysd.get("cpu_s", 0.0) - ps.get("cpu_s", 0.0)) / dt * 100.0,
                    0.0, 999.0)
                self.prev_snap = {
                    "sys": dict(sysd),
                    "cache": dict(cache),
                    "trunk": dict(trunk),
                }
                self.prev_snap_t = now
        else:
            self.prev_snap = {
                "sys": dict(sysd),
                "cache": dict(cache),
                "trunk": dict(trunk),
            }
            self.prev_snap_t = now
        self.last = ev

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return ""

    def _read_mem_available(self) -> float:
        for line in self._read_text("/proc/meminfo").splitlines():
            if line.startswith("MemAvailable:"):
                try:
                    return int(line.split()[1]) / 1048576.0  # kB -> GiB
                except (IndexError, ValueError):
                    return 0.0
        return 0.0

    def _discover_cpu_temp(self) -> None:
        """Find the most likely CPU package sensor in sysfs, once."""
        self.cpu_temp_checked = True
        best_score = 0
        best_path: Optional[str] = None

        for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            name = self._read_text(str(hwmon / "name")).lower()
            for sensor in sorted(hwmon.glob("temp*_input")):
                label_path = sensor.with_name(
                    sensor.name.replace("_input", "_label"))
                label = self._read_text(str(label_path)).lower()
                text = f"{name} {label}"
                if "package" in text:
                    score = 100
                elif "tctl" in text or "tdie" in text:
                    score = 90
                elif "x86_pkg_temp" in text or "pkg" in text:
                    score = 80
                elif name in ("coretemp", "k10temp", "zenpower"):
                    score = 70
                elif "core" in label or "cpu" in text or "tcpu" in text:
                    score = 60
                elif "acpitz" in text:
                    score = 10
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_path = str(sensor)

        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            ztype = self._read_text(str(zone / "type")).lower()
            if "x86_pkg_temp" in ztype or "pkg" in ztype:
                score = 80
            elif "tcpu" in ztype:
                score = 75
            elif "cpu" in ztype:
                score = 70
            elif "acpitz" in ztype:
                score = 10
            else:
                score = 0
            if score > best_score:
                best_score = score
                best_path = str(zone / "temp")

        self.cpu_temp_path = best_path

    def _read_cpu_temp(self) -> Optional[float]:
        if not self.cpu_temp_checked:
            self._discover_cpu_temp()
        if not self.cpu_temp_path:
            return None
        text = self._read_text(self.cpu_temp_path)
        try:
            value = int(text)
        except ValueError:
            try:
                value = int(float(text))
            except ValueError:
                self.cpu_temp_path = None
                return None
        if value <= 0:
            return None
        # hwmon and thermal_zone temperatures are millidegrees Celsius.
        return value / 1000.0 if value > 1000 else float(value)

    @staticmethod
    def _read_cpu_times() -> dict[int, tuple[int, int]]:
        """Busy/total jiffies for cores 0-3 from /proc/stat."""
        out: dict[int, tuple[int, int]] = {}
        try:
            lines = Path("/proc/stat").read_text().splitlines()
        except OSError:
            return out
        for line in lines:
            parts = line.split()
            if len(parts) < 5 or not parts[0].startswith("cpu"):
                continue
            if parts[0] == "cpu":
                continue
            try:
                core = int(parts[0][3:])
                values = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            if core > 3 or len(values) < 4:
                continue
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            out[core] = (sum(values) - idle, sum(values))
        return out

    def update_cpu(self) -> None:
        """Sample per-core utilization and the package temperature."""
        now = time.monotonic()
        current = self._read_cpu_times()
        if self.cpu_prev is not None and now - self.cpu_prev_t >= 0.2:
            for core in range(4):
                before = self.cpu_prev.get(core)
                after = current.get(core)
                if not before or not after:
                    self.core_pct[core] = None
                    continue
                busy = after[0] - before[0]
                total = after[1] - before[1]
                self.core_pct[core] = (
                    clamp(busy / total * 100.0, 0.0, 100.0)
                    if total > 0 else self.core_pct[core])
        self.cpu_prev = current
        self.cpu_prev_t = now
        self.cpu_temp_c = self._read_cpu_temp()

    def handle_event(self, ev: dict) -> None:
        name = ev.get("event")
        # Layer events arrive ~27/s on a 1.6 s/token run. The panel is sampled
        # at 1 Hz -- rates over a one-second window are more meaningful than
        # rates over 40 ms, and a number that changes on every layer is noise.
        # Token text and layer progress below are NOT throttled.
        now = time.monotonic()
        force_stats = name in ("ready", "stats", "done")
        # Prefill emits far more progress events than the panel needs. Treat
        # them like any other event: the stats panel is a 1 Hz instrument, and
        # the PREFILL bar in the status line already updates live from every
        # event. Ingesting here too would repaint the MEMORY block many times a
        # second during the prompt.
        if force_stats or now - self.last_stats_ingest >= self.stats_interval:
            self.ingest_stats(ev)
            self.last_stats_ingest = now
            if force_stats:
                self.stats_force = True
        if name in ("ready", "stats"):
            first = not self.ready
            self.ready = ev
            self.context_cap = (ev.get("context") or {}).get("cap", self.args.ctx)
            self.context_used = (ev.get("context") or {}).get("used", 0)
            if first:
                cfg = ev.get("config") or {}
                cache = ev.get("cache") or {}
                trunk = ev.get("trunk") or {}
                self.messages.append(Message(
                    "system",
                    f"loaded {cfg.get('layers', '?')} layers, "
                    f"{cfg.get('hidden', '?')}d, "
                    f"{cfg.get('experts', '?')} experts top{cfg.get('topk', '?')}; "
                    f"cache {cache.get('slots', '?')} slots "
                    f"({cache.get('resident_gb', 0):+.2f} GB), "
                    f"trunk {trunk.get('pinned', '?')} pinned. "
                    f"Context {self.context_cap} tokens."))
                if self.markdown_enabled and not RICH_AVAILABLE and not self.md_warned:
                    self.md_warned = True
                    self.add_system(
                        "Markdown rendering is off because rich is not installed. "
                        "Install it with: python3 -m pip install -r "
                        "tui/requirements.txt  (or run /markdown off to hide this)")
            if self.status in ("starting", "waiting"):
                self.status = "ready"
            self.dirty = True

        elif name == "start":
            self.generating = True
            self.status = "generating"
            self.status_detail = ""
            self.generation = 0
            self.gen_max = int(ev.get("gen_max") or self.gen_limit)
            self.prompt_tokens = int(ev.get("prompt_tokens") or 0)
            self.context_cap = int(ev.get("context_cap") or self.context_cap)
            self.context_used = int(ev.get("pos") or 0) + self.prompt_tokens
            self.turn_start = time.monotonic()
            self.last_token_t = 0.0
            # decode_spt is cleared on the first decoded token, so the graph
            # only ever covers the current turn's decode.
            self.prefill_batch = int(ev.get("prefill_batch") or 0)
            self.prefill_progress = {}
            self.prefill_active = self.prefill_batch > 1
            self.prefill_s = 0.0
            self.prefill_tokens = 0
            self.ttft = 0.0
            self.current_layer = 0
            self.phase = "prompt"
            self.top = []
            self.mode = ev.get("mode", "chat")
            self.decoder.reset()
            if not self.current_msg or self.current_msg.role != "assistant":
                self.current_msg = Message("assistant", "", think=self.think)
                self.messages.append(self.current_msg)
            self.current_msg.think = self.mode == "thinking"
            self.cancel_pending = False
            self.stats_force = True
            self.dirty = True

        elif name == "layer":
            self.current_layer = int(ev.get("layer") or 0) + 1
            self.phase = ev.get("phase", "")
            self.dirty = True

        elif name == "prefill":
            self.prefill_progress = ev
            self.prefill_active = True
            self.prefill_batch = int(ev.get("batch") or self.prefill_batch or 0)
            self.current_layer = int(ev.get("layer") or self.current_layer)
            self.phase = "prompt"
            self.dirty = True

        elif name == "token":
            raw = bytes.fromhex(ev.get("hex") or "")
            text = strip_special_tokens(self.decoder.decode(raw, final=False))
            if text and self.current_msg:
                self.current_msg.text += text
            index = int(ev.get("index") or 0)
            self.generation = index + 1
            self.context_used += 1
            t = float(ev.get("t") or 0.0)
            ttft = float(ev.get("ttft") or self.ttft)
            if index == 0:
                # The first decoded token switches the panel from PREFILL to
                # DECODE; start the decode average fresh.
                self.phase = "gen"
                self.decode_spt.clear()
                ps = float(ev.get("prefill_s") or 0.0)
                pt = int(ev.get("prefill_tokens") or self.prompt_tokens or 0)
                if ps > 0.0 and pt > 0:
                    self.prefill_s = ps
                    self.prefill_tokens = pt
                self.prefill_active = False
            elif self.last_token_t > 0:
                dt = t - self.last_token_t
                if dt > 1e-6:
                    self.decode_spt.append(dt)
            self.last_token_t = t
            self.ttft = ttft
            self.top = ev.get("top") or []
            self.stats_force = True
            self.dirty = True

        elif name == "done":
            self.generating = False
            self.cancel_pending = False
            self.prefill_active = False
            self.status = "ready" if ev.get("stop") != "error" else "failed"
            self.last_done = ev
            self.context_used = int(ev.get("context_used") or self.context_used)
            if self.current_msg is not None:
                self.current_msg.meta = {
                    "generated": ev.get("generated"),
                    "seconds": ev.get("seconds"),
                    "s_per_token": ev.get("s_per_token"),
                    "ttft": ev.get("ttft"),
                    "stop": ev.get("stop"),
                }
                if not self.current_msg.text.strip():
                    self.current_msg.text = "(no tokens)"
            self.current_msg = None
            self.dirty = True

        elif name == "reset":
            self.generating = False
            self.cancel_pending = False
            self.status = "ready"
            self.context_used = 0
            self.current_msg = None
            self.decode_spt.clear()
            self.prefill_active = False
            self.prefill_progress = {}
            self.prefill_s = 0.0
            self.prefill_tokens = 0
            self.stats_force = True
            self.dirty = True

        elif name == "system":
            self.dirty = True

        elif name == "error":
            msg = str(ev.get("message") or "unknown engine error")
            # If an assistant bubble was opened for a turn that never started,
            # do not leave an empty one above the error.
            if self.current_msg is not None and not self.current_msg.text:
                try:
                    self.messages.remove(self.current_msg)
                except ValueError:
                    pass
            self.current_msg = None
            self.generating = False
            self.cancel_pending = False
            self.prefill_active = False
            if self.status not in ("starting",):
                self.status = "ready" if self.backend_alive else "failed"
            self.messages.append(Message("error", msg))
            self.dirty = True

    # -- commands ----------------------------------------------------------
    def add_system(self, text: str) -> None:
        self.messages.append(Message("system", text))
        self.dirty = True

    def submit(self) -> None:
        text = self.input.strip()
        if not text:
            return
        self.input = ""
        self.input_cursor = 0

        if text.startswith("/"):
            self.command(text)
            return

        if not self.backend_alive:
            self.add_system("engine is not running; restart the TUI")
            return
        if self.status in ("starting", "waiting"):
            self.add_system("model is still loading; wait for the ready event")
            return
        if self.generating:
            self.add_system("a turn is already generating")
            return

        self.messages.append(Message("user", text))
        self.current_msg = Message("assistant", "", think=self.think)
        self.messages.append(self.current_msg)
        self.decoder.reset()
        self.scroll = 0
        self.generating = True
        self.status = "waiting"
        sent = self.backend.send({
            "cmd": "generate",
            "prompt": text,
            "gen": self.gen_limit,
            "think": self.think,
            "layers": self.show_layers,
        })
        if not sent:
            self.generating = False
            self.status = "failed"
            self.add_system("could not write to the engine")
        self.dirty = True

    def cancel(self) -> None:
        """Stop at the next token; the turn stays in the model context."""
        if self.status != "generating" or self.cancel_pending:
            return
        if self.backend and self.backend.cancel():
            self.cancel_pending = True
            self.status_detail = "cancelling"
        else:
            self.add_system("could not request cancellation")
        self.dirty = True

    def command(self, text: str) -> None:
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            self.running = False
        elif cmd == "/help":
            self.add_system(
                "commands:\n"
                "  /help                 this list\n"
                "  /reset                clear model context (keeps the disk cache warm)\n"
                "  /clear                clear the chat display only\n"
                "  /system TEXT          system prompt for the NEXT reset\n"
                "  /think [on|off]       toggle thinking mode for the next turn\n"
                "  /gen N                max tokens per turn\n"
                "  /layers [on|off]      stream per-layer progress events\n"
                "  /markdown [on|off]    Markdown rendering (rich, optional)\n"
                "  /stats                ask the engine for a fresh snapshot\n"
                "  /exit                 quit")
        elif cmd == "/reset":
            if self.generating:
                self.add_system("finish the current turn before resetting")
                return
            if self.backend_alive:
                self.backend.send({"cmd": "reset"})
            self.messages.clear()
            self.current_msg = None
            self.generating = False
            self.status = "ready"
            self.context_used = 0
            self.decode_spt.clear()
            self.stats_force = True
            self.add_system("context reset; expert cache stays warm")
        elif cmd == "/clear":
            self.messages.clear()
            self.add_system("display cleared")
        elif cmd == "/system":
            if not arg:
                self.add_system(f"system prompt: {self.system_prompt or '(empty)'}")
            else:
                self.system_prompt = arg
                if self.context_used > 0:
                    self.add_system("system prompt stored; /reset for it to take effect")
                if self.backend_alive:
                    self.backend.send({"cmd": "system", "text": arg})
                else:
                    self.add_system("engine is not running")
        elif cmd == "/think":
            if arg.lower() in ("on", "1", "true", "yes"):
                self.think = True
            elif arg.lower() in ("off", "0", "false", "no"):
                self.think = False
            else:
                self.think = not self.think
            self.add_system(f"thinking mode {'on' if self.think else 'off'} "
                            f"for the next turn")
        elif cmd == "/gen":
            try:
                n = int(arg)
                if n < 1:
                    raise ValueError
                self.gen_limit = n
                self.add_system(f"max {n} tokens per turn")
            except ValueError:
                self.add_system("usage: /gen N   (N >= 1)")
        elif cmd == "/layers":
            if arg.lower() in ("on", "1", "true", "yes"):
                self.show_layers = True
            elif arg.lower() in ("off", "0", "false", "no"):
                self.show_layers = False
            else:
                self.show_layers = not self.show_layers
            self.add_system(f"layer events {'on' if self.show_layers else 'off'}")
        elif cmd == "/markdown":
            if arg.lower() in ("on", "1", "true", "yes"):
                self.markdown_enabled = True
            elif arg.lower() in ("off", "0", "false", "no"):
                self.markdown_enabled = False
            else:
                self.markdown_enabled = not self.markdown_enabled
            if self.markdown_enabled and not RICH_AVAILABLE:
                self.add_system("Markdown needs rich: python3 -m pip install -r "
                                "tui/requirements.txt")
            else:
                self.add_system(f"Markdown rendering "
                                f"{'on' if self.markdown_enabled else 'off'}")
        elif cmd == "/stats":
            if self.backend_alive:
                self.backend.send({"cmd": "stats"})
            self.add_system("requested a fresh snapshot")
        else:
            self.add_system(f"unknown command {cmd}; /help for the list")
        self.dirty = True

    # -- input -------------------------------------------------------------
    def key(self, ch: Any) -> None:
        if isinstance(ch, int):
            self.key_code(ch)
            return
        if ch == "\x03":                        # Ctrl-C
            self.running = False
        elif ch == "\x04":                      # Ctrl-D
            if not self.input:
                self.running = False
        elif ch == "\x15":                      # Ctrl-U
            self.input = ""
            self.input_cursor = 0
        elif ch in ("\n", "\r"):
            self.submit()
        elif ch == "\x7f" or ch == "\b":
            if self.input_cursor > 0:
                self.input = (self.input[:self.input_cursor - 1]
                              + self.input[self.input_cursor:])
                self.input_cursor -= 1
        elif ch == "\x1b":                      # ESC
            self.input = ""
            self.input_cursor = 0
        elif ch >= " ":
            if self.input_cursor < len(self.input):
                self.input = (self.input[:self.input_cursor] + ch
                              + self.input[self.input_cursor:])
            else:
                self.input += ch
            self.input_cursor += 1
        self.dirty = True

    def handle_mouse(self) -> None:
        try:
            _, mx, my, _, bstate = curses.getmouse()
        except curses.error:
            return
        buttons = (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED
                   | curses.BUTTON1_RELEASED)
        if not (bstate & buttons):
            return
        rect = self.cancel_rect
        if rect and my == rect[0] and rect[1] <= mx < rect[1] + rect[2]:
            self.cancel()
            self.dirty = True

    def key_code(self, code: int) -> None:
        if code in (curses.KEY_ENTER, 10, 13):
            self.submit()
        elif code in (curses.KEY_BACKSPACE, 127, 8):
            if self.input_cursor > 0:
                self.input = (self.input[:self.input_cursor - 1]
                              + self.input[self.input_cursor:])
                self.input_cursor -= 1
        elif code == curses.KEY_DC:
            if self.input_cursor < len(self.input):
                self.input = (self.input[:self.input_cursor]
                              + self.input[self.input_cursor + 1:])
        elif code == curses.KEY_LEFT:
            self.input_cursor = max(0, self.input_cursor - 1)
        elif code == curses.KEY_RIGHT:
            self.input_cursor = min(len(self.input), self.input_cursor + 1)
        elif code == curses.KEY_HOME:
            self.input_cursor = 0
        elif code == curses.KEY_END:
            self.input_cursor = len(self.input)
        elif code == curses.KEY_PPAGE:
            self.scroll += 12
        elif code == curses.KEY_NPAGE:
            self.scroll = max(0, self.scroll - 12)
        elif code == curses.KEY_UP:
            self.scroll += 1
        elif code == curses.KEY_DOWN:
            self.scroll = max(0, self.scroll - 1)
        elif code == curses.KEY_RESIZE:
            pass
        self.dirty = True

    # -- derived display values -------------------------------------------
    def values(self) -> dict:
        cfg = self.ready.get("config") or {}
        sysd = self.last.get("sys") or self.ready.get("sys") or {}
        cache = self.last.get("cache") or self.ready.get("cache") or {}
        trunk = self.last.get("trunk") or self.ready.get("trunk") or {}
        prof = self.last.get("prof") or {}

        if self.decode_spt:
            recent = list(self.decode_spt)[-16:]
            spt = sum(recent) / len(recent)
        elif self.last_done.get("s_per_token"):
            spt = float(self.last_done["s_per_token"])
        else:
            spt = 0.0
        tps = 1.0 / spt if spt > 0 else 0.0

        elapsed = (time.monotonic() - self.turn_start) if self.generating else \
            float(self.last_done.get("seconds") or 0.0)

        return {
            "cfg": cfg, "sys": sysd, "cache": cache, "trunk": trunk,
            "prof": prof, "spt": spt, "tps": tps, "elapsed": elapsed,
        }

    # -- rendering ---------------------------------------------------------
    def render(self, stdscr: "curses.window") -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 14 or w < 62:
            self._add(stdscr, 0, 0, "terminal too small; need at least 62x14",
                      w - 1, self.attr("error"))
            stdscr.refresh()
            return

        self.spinner = (self.spinner + 1) % 4
        panel_w = min(52, max(36, int(w * 0.37)))
        sep_x = w - panel_w - 1
        chat_w = sep_x - 1
        body_top, body_bottom = 2, h - 3

        self._draw_header(stdscr, w)
        self._draw_status(stdscr, w, body_top - 1)
        self._draw_chat(stdscr, body_top, body_bottom, chat_w)
        self._draw_separator(stdscr, body_top, body_bottom, sep_x)
        self._draw_stats(stdscr, body_top, body_bottom, sep_x + 2, panel_w - 1)
        self._draw_footer(stdscr, h, w)
        self._draw_input(stdscr, h, w)

        stdscr.refresh()
        self.last_render = time.monotonic()
        self.dirty = False

    def _draw_header(self, scr: "curses.window", w: int) -> None:
        model = Path(self.args.model).name if not self.demo else "demo"
        brand = f" DSV4  {model}"
        status = self.status
        if self.demo:
            brand += "  [demo]"
        progress = ""
        if status == "generating":
            n_layers = int((self.ready.get("config") or {}).get("layers") or 0) or 43
            if self.phase == "prompt":
                if self.prefill_batch > 1:
                    phase = f"prefill batch {self.prefill_batch}"
                else:
                    phase = f"prefill {self.current_layer}/{n_layers}"
            elif self.phase == "resume":
                phase = "closing previous turn"
            else:
                phase = f"layer {self.current_layer}/{n_layers}"
            progress = (f"  generating {self.generation}/{self.gen_max} tok"
                        f"  {phase}"
                        f"  ctx {self.context_used}/{self.context_cap}")
        self._add(scr, 0, 0, brand, w - 1, self.attr("title"))
        if progress:
            self._add(scr, 0, len(brand), progress, max(1, w - len(brand) - 1),
                      self.attr("muted"))
        right = "" if status in ("generating", "ready") else f"{status} "
        if right:
            self._add(scr, 0, max(0, w - len(right) - 1), right, len(right),
                      self.attr("value"))

    def _draw_status(self, scr: "curses.window", w: int, row: int) -> None:
        if self.status == "generating":
            # Progress and context live in the title bar while a turn runs.
            return
        elif self.status == "waiting":
            text = f"... waiting for engine  (prompt {self.prompt_tokens} tokens)"
            color = "active"
        elif self.status == "starting":
            spin = "|/-\\"[self.spinner]
            if self.demo:
                text = f"{spin} starting demo backend"
            else:
                text = f"{spin} loading checkpoint (this is the 2 GB embed/head read)"
            color = "active"
        elif self.status == "failed":
            text = "engine failed; see the error in the transcript"
            color = "error"
        else:
            text = (f"context {self.context_used}/{self.context_cap}  "
                    f"gen limit {self.gen_limit}  "
                    f"thinking {'on' if self.think else 'off'}  "
                    f"layers {'on' if self.show_layers else 'off'}")
            color = "status"
        self._add(scr, row, 0, text, w - 1, self.attr(color))

    def _wrap(self, text: str, width: int) -> list[str]:
        width = max(8, width)
        out: list[str] = []
        for raw in text.replace("\t", "    ").split("\n"):
            raw = "".join(c if c.isprintable() else "?" for c in raw)
            if not raw:
                out.append("")
                continue
            while len(raw) > width:
                cut = raw.rfind(" ", 0, width + 1)
                if cut <= 0:
                    cut = width
                out.append(raw[:cut].rstrip())
                raw = raw[cut:].lstrip()
            out.append(raw)
        return out

    def chat_lines(self, width: int) -> list[list[Span]]:
        """Build the transcript as lines of styled spans.

        Each line is a list because Markdown can put several styles on one line
        (e.g. a paragraph containing *italic* and **bold**). The speaker prefix
        is a named span so it keeps the role colour even when the body is
        Markdown.
        """
        lines: list[list[Span]] = []
        for msg in self.messages:
            if msg.role == "user":
                label, color = "you", "user"
            elif msg.role == "assistant":
                label, color = "v4", "assistant"
            elif msg.role == "error":
                label, color = "!!", "error"
            else:
                label, color = "··", "system"

            body = msg.text
            if msg.role == "assistant" and msg.think:
                if "</think>" in body:
                    reasoning, body = body.split("</think>", 1)
                elif "<think>" in body:
                    reasoning, body = body.split("<think>", 1)
                else:
                    reasoning, body = body, ""
                reasoning = reasoning.strip()
                if reasoning:
                    for ln in self._wrap(reasoning, width - 4):
                        lines.append([Span("    " + ln, named="reasoning")])
                    lines.append([])

            use_md = (msg.role == "assistant" and self.markdown_enabled
                      and RICH_AVAILABLE)
            if use_md:
                md_width = max(10, width - 4)
                # The body only grows while streaming, so (length, width) is a
                # sufficient cache key for this prototype.
                key = (len(body), md_width)
                if msg.md_cache_key != key:
                    rendered = render_markdown(body, md_width) if body else None
                    if rendered is None:
                        rendered = [[Span(ln)] for ln in self._wrap(body, md_width)]
                    msg.md_cache_lines = rendered
                    msg.md_cache_key = key
                md_lines = msg.md_cache_lines or [[]]
                for i, spans in enumerate(md_lines):
                    prefix = Span(f"{label} › " if i == 0 else "    ",
                                  named=color)
                    lines.append([prefix] + list(spans))
            else:
                for i, ln in enumerate(self._wrap(body, width - 4)):
                    prefix = f"{label} › " if i == 0 else "    "
                    lines.append([Span(prefix + ln, named=color)])

            if msg.meta:
                m = msg.meta
                meta = (f"{m.get('generated', 0)} tok · "
                        f"{float(m.get('seconds') or 0):.1f}s · "
                        f"{float(m.get('s_per_token') or 0):.2f} s/tok · "
                        f"ttft {float(m.get('ttft') or 0):.1f}s · "
                        f"{m.get('stop', '')}")
                lines.append([Span("    " + meta, named="dim")])
            lines.append([])
        return lines

    def _draw_chat(self, scr: "curses.window", top: int, bottom: int,
                   width: int) -> None:
        view_h = bottom - top + 1
        lines = self.chat_lines(max(10, width))
        max_scroll = max(0, len(lines) - view_h)
        self.scroll = min(self.scroll, max_scroll)
        end = max(0, len(lines) - self.scroll)
        start = max(0, end - view_h)
        visible = lines[start:end]

        # Right-align the transcript to the bottom while it fits, so a new chat
        # starts at the bottom edge rather than floating at the top.
        pad = max(0, view_h - len(visible))
        for i, spans in enumerate(visible):
            row = top + pad + i
            if row > bottom:
                break
            x = 0
            for span in spans:
                if x >= width:
                    break
                text = span.text.replace("\t", "    ")
                text = "".join(c if c.isprintable() else "?" for c in text)
                if not text:
                    continue
                self._add(scr, row, x, text, width - x, self.span_attr(span))
                x += len(text)

    def _draw_separator(self, scr: "curses.window", top: int, bottom: int,
                        col: int) -> None:
        for r in range(top, bottom + 1):
            self._add(scr, r, col, "|", 1, self.attr("dim"))

    @staticmethod
    def _short(value: float) -> str:
        """A compact number for the header readout."""
        if value >= 100.0:
            return f"{value:.0f}"
        if value >= 10.0:
            return f"{value:.1f}"
        if value >= 1.0:
            return f"{value:.2f}"
        return f"{value:.3f}"

    @staticmethod
    def _short_spt(value: float) -> str:
        """Seconds per token: two decimals is enough beside tok/s."""
        if value >= 100.0:
            return f"{value:.0f}"
        if value >= 10.0:
            return f"{value:.1f}"
        if value >= 0.01:
            return f"{value:.2f}"
        return f"{value:.3f}"

    def stats_lines(self, width: int,
                    height: int) -> list[list[tuple[str, str]]]:
        """Decode tok/s graph plus live memory and CPU numbers.

        The graph is a right-aligned sliding window: new samples appear at the
        right edge and older ones move out to the left. The header carries the
        current tok/s and s/tok; the axis uses round, linear tick labels. A
        compact MEMORY block below it shows RSS, the expert cache, disk reads
        and the approximate byte inflow to the CPU.
        """
        out: list[list[tuple[str, str]]] = []

        def emit(*segments: tuple[str, str]) -> None:
            out.append(list(segments))

        in_prefill = self.prefill_active
        show_prefill = in_prefill or self.prefill_tokens > 0
        # Keep the last turn's graph on screen after `done`; it is replaced by
        # the next turn's decode data once that turn produces tokens.
        has_decode = bool(self.decode_spt)
        in_decode = (not in_prefill
                     and self.phase not in ("prompt", "resume")
                     and (self.generating or has_decode))

        def emit_prefill() -> None:
            p = self.prefill_progress or {}
            stage = p.get("stage") or "start"
            batch = int(p.get("batch") or self.prefill_batch or 0)
            chunk = int(p.get("chunk") or 0)
            chunks = int(p.get("chunks") or 0)
            layer = int(p.get("layer") or 0)
            layers = int(p.get("layers") or 0)

            final = self.prefill_tokens > 0 and self.prefill_s > 0.0
            if final:
                # The engine's own prompt timer, measured without the head.
                tokens = self.prefill_tokens
                seconds = self.prefill_s
                total = done = tokens
                status = "done"
            else:
                tokens = int(p.get("tokens_done") or 0)
                seconds = float(p.get("elapsed") or 0.0)
                total = int(p.get("tokens_total") or self.prompt_tokens or 0)
                done = tokens
                if stage == "layer":
                    status = f"layers {layer}/{layers}"
                elif stage == "chunk":
                    status = f"chunk {chunk}/{chunks}"
                elif stage == "chunk_done":
                    status = f"chunk {chunk}/{chunks} done"
                elif stage == "head":
                    status = "output head"
                elif stage == "done":
                    status = "prompt done"
                else:
                    status = "preparing"

            seg: list[tuple[str, str]] = [
                (" PREFILL", "section"), (f" {status}", "muted" if status == "done" else "active")]
            if total:
                seg.append((f"  {done}/{total} tok", "value"))
            if batch > 1:
                seg.append((f"  batch {batch}", "dim"))
            emit(*seg)

            if final:
                spt = seconds / tokens
                tps = tokens / seconds
                # The header already carries the batch size when it is > 1.
                mode = "; per-token" if batch <= 1 else ""
                emit((" ", "dim"),
                     (f"{tokens} tokens in {seconds:.2f} s "
                      f"({tps:.2f} tok/s, {spt:.3f} s/tok)",
                      "value"))
            else:
                prog = clamp(float(p.get("progress") or 0.0), 0.0, 1.0)
                elapsed = float(p.get("elapsed") or 0.0)
                bar_w = max(8, min(28, width - 18))
                filled = int(round(prog * bar_w))
                bar = "█" * filled + "░" * (bar_w - filled)
                rate = (prog * total / elapsed) if elapsed > 0.05 else 0.0
                emit((" [", "dim"), (bar, "prefill"), ("]", "dim"))
                emit((" ", "dim"), (f"{prog * 100:3.0f}%", "value"),
                     (f"  {elapsed:5.1f}s", "dim"),
                     ((f"  {self._short(rate)} tok/s", "dim") if rate > 0
                      else ("  -", "dim")))
            emit()

        if show_prefill:
            emit_prefill()

        samples = [("decode", 1.0 / dt) for dt in self.decode_spt
                   if dt > 1e-9]

        cur_spt = 0.0
        if samples:
            cur_spt = 1.0 / samples[-1][1]
        elif not self.generating:
            cur_spt = float(self.last_done.get("s_per_token") or 0.0)
        cur_rate = 1.0 / cur_spt if cur_spt > 0 else 0.0
        rate_txt = self._short(cur_rate) if cur_rate else "-"
        spt_txt = self._short_spt(cur_spt) if cur_spt else "-"

        # The inflow estimate needs a token rate. Decode supplies it once the
        # graph has samples; during prefill use the engine's prompt timing or
        # the live progress estimate instead of showing zero.
        flow_rate = cur_rate
        if flow_rate <= 0.0:
            if self.prefill_tokens > 0 and self.prefill_s > 0.0:
                flow_rate = self.prefill_tokens / self.prefill_s
            else:
                p = self.prefill_progress or {}
                elapsed = float(p.get("elapsed") or 0.0)
                done = int(p.get("tokens_done") or 0)
                if elapsed > 0.05 and done > 0:
                    flow_rate = done / elapsed

        def emit_memory() -> None:
            sysd = self.last.get("sys") or self.ready.get("sys") or {}
            cache = self.last.get("cache") or self.ready.get("cache") or {}
            trunk = self.last.get("trunk") or self.ready.get("trunk") or {}
            cfg = self.ready.get("config") or {}
            rss = float(sysd.get("rss_mb") or 0.0) / 1024.0
            hit = float(cache.get("hit_rate") or 0.0) * 100.0
            avail = (f"{self.mem_available_gb:.2f} GB"
                     if self.mem_available_gb > 0 else "n/a")

            # Per token the CPU reads every dense trunk weight plus the routed
            # experts from RAM. That is a fixed byte count for a given model,
            # so multiplying by the current token rate estimates the flow. The
            # rate is decode tok/s, or the prompt's token rate during prefill.
            layers = int(cfg.get("layers") or 0)
            topk = int(cfg.get("topk") or 0)
            slot_mb = float(cache.get("slot_mb") or 0.0)
            expert_gb = layers * topk * slot_mb / 1024.0
            trunk_gb = float(trunk.get("resident_gb") or 0.0)
            inflow = (expert_gb + trunk_gb) * flow_rate

            emit((" MEMORY", "section"))
            emit((" RSS  ", "dim"), (f"{rss:.2f} GB", "value"),
                 ("  avail ", "dim"), (avail, "value"))
            emit((" cache-hit ", "dim"), (f"{hit:5.1f}%", "value"))
            emit((" disk ", "dim"), (fmt_gbps(self.io_rate), "value"),
                 ("  inflow ", "dim"), (f"~{fmt_gbps(inflow)}", "value"))

        def emit_cpu() -> None:
            def percent(core: int) -> str:
                value = self.core_pct[core]
                return f"{value:3.0f}%" if value is not None else "  - "

            emit((" CPU", "section"))
            segments: list[tuple[str, str]] = []
            for core in range(4):
                label = f" c{core} " if core == 0 else f"  c{core} "
                segments.append((label, "dim"))
                segments.append((percent(core), "value"))
            emit(*segments)
            temp = (f"{self.cpu_temp_c:.0f}°C"
                    if self.cpu_temp_c is not None else "n/a")
            emit((" temp ", "dim"), (temp, "value"))

        def emit_bottom() -> None:
            if out and out[-1]:
                emit()
            emit_memory()
            emit()
            emit_cpu()

        if in_decode:
            emit((" DECODE", "section"),
                 (f" {rate_txt} tok/s", "dim"),
                 (f" ({spt_txt} s/tok)", "dim"))

            if self.decode_spt:
                avg_spt = sum(self.decode_spt) / len(self.decode_spt)
                avg_rate = 1.0 / avg_spt if avg_spt > 0 else 0.0
                emit((" ⌀ avg  ", "dim"),
                     (f"{self._short(avg_rate)} tok/s", "system"),
                     (f" ({self._short_spt(avg_spt)} s/tok)", "value"))
            else:
                emit((" ⌀ avg  ", "dim"), ("-", "value"))

            if samples:
                graph_w = max(6, width - 8)
                visible = samples[-graph_w:]
                cols = [("empty", 0.0)] * (graph_w - len(visible)) + visible

                graph_h = max(3, min(10, height - 12
                                     - (3 if show_prefill else 0)))
                target = 5 if graph_h >= 6 else 2
                peak = max(rate for _, rate in visible)
                step = nice_step(peak, target)
                axis_max = math.ceil(peak / step - 1e-9) * step

                row_labels: dict[int, str] = {}
                for i in range(int(round(axis_max / step)) + 1):
                    value = i * step
                    row = graph_h - 1 - int(round(
                        (value / axis_max) * (graph_h - 1)))
                    row = int(clamp(row, 0, graph_h - 1))
                    row_labels.setdefault(row, axis_label(value, step))

                blocks = " ▁▂▃▄▅▆▇█"
                sub = 8

                for r in range(graph_h):
                    label = f"{row_labels.get(r, ''):>7}"
                    segments: list[tuple[str, str]] = []
                    cur_color: Optional[str] = None
                    cur_text = ""
                    for _, rate in cols:
                        frac = (clamp(rate / axis_max, 0.0, 1.0)
                                if axis_max > 0 else 0.0)
                        filled = (max(1, int(round(frac * graph_h * sub)))
                                  if rate > 1e-9 else 0)
                        n = filled - (graph_h - 1 - r) * sub
                        if n >= sub:
                            ch = "█"
                        elif n > 0:
                            ch = blocks[n]
                        else:
                            ch = " "
                        color = "decode" if ch != " " else "dim"
                        if color != cur_color:
                            if cur_text:
                                segments.append((cur_text, cur_color or "dim"))
                            cur_text = ""
                            cur_color = color
                        cur_text += ch
                    if cur_text:
                        segments.append((cur_text, cur_color or "dim"))
                    emit((label + "┤", "dim"), *segments)

                emit((" " * 7 + "└", "dim"), ("─" * graph_w, "dim"))
            else:
                emit(("  waiting for the first token...", "dim"))
        elif not in_prefill:
            if not self.generating:
                emit((" Waiting for a turn...", "dim"))
            elif self.phase == "resume":
                emit(("  closing previous turn...", "dim"))
            else:
                emit(("  prefill in progress...", "dim"))

        emit_bottom()
        return out

    def _draw_stats(self, scr: "curses.window", top: int, bottom: int,
                    col: int, width: int) -> None:
        if width <= 4:
            return
        # The panel is a 1 Hz instrument. The status line above it stays live
        # for layer/token/prefill progress; repainting a graph on every event
        # only makes it jitter faster than the values move. Prefill is no
        # exception: its progress bar lives in the status line, not here.
        height = bottom - top + 1
        now = time.monotonic()
        if (self.stats_force or not self.stats_cache
                or width != self.stats_cache_width
                or height != self.stats_cache_height
                or now - self.stats_cache_ts >= self.stats_interval):
            self.update_cpu()
            self.mem_available_gb = self._read_mem_available()
            self.stats_cache = self.stats_lines(width, height)
            self.stats_cache_ts = now
            self.stats_cache_width = width
            self.stats_cache_height = height
            self.stats_force = False
        for i, segments in enumerate(self.stats_cache):
            row = top + i
            if row > bottom:
                break
            x = 0
            for text, color in segments:
                if x >= width:
                    break
                text = text[:width - x]
                if text:
                    self._add(scr, row, col + x, text, width - x,
                              self.attr(color))
                    x += len(text)

    def _draw_footer(self, scr: "curses.window", h: int, w: int) -> None:
        self.cancel_rect = None
        if self.generating:
            hint = "PgUp/PgDn scroll  ·  Ctrl-C quit  ·  "
            label = "[cancelling]" if self.cancel_pending else "[cancel]"
            self._add(scr, h - 2, 0, hint, w - 1, self.attr("dim"))
            self._add(scr, h - 2, len(hint), label, w - len(hint) - 1,
                      self.attr("cancel"))
            self.cancel_rect = (h - 2, len(hint), len(label))
        else:
            hint = "Enter send  ·  /help commands  ·  PgUp/PgDn scroll  ·  Ctrl-C quit"
            self._add(scr, h - 2, 0, hint, w - 1, self.attr("dim"))

    def _draw_input(self, scr: "curses.window", h: int, w: int) -> None:
        prompt = "› "
        room = max(1, w - len(prompt) - 1)
        start = 0
        if self.input_cursor > room:
            start = self.input_cursor - room
        shown = self.input[start:start + room]
        color = "input" if not self.generating else "dim"
        self._add(scr, h - 1, 0, prompt + shown, w - 1, self.attr(color))
        cx = len(prompt) + (self.input_cursor - start)
        try:
            scr.move(h - 1, min(w - 2, cx))
            curses.curs_set(1)
        except curses.error:
            pass

    # -- curses helpers ----------------------------------------------------
    def attr(self, name: str) -> int:
        return self.pairs.get(name, 0)

    def _color_index(self, spec: Any) -> int:
        """RGB/xterm-256 -> an index this terminal can actually init_pair()."""
        if spec is None:
            return -1
        if isinstance(spec, tuple):
            rgb = (int(spec[0]), int(spec[1]), int(spec[2]))
        else:
            idx = int(spec)
            if curses.COLORS >= 256 and 0 <= idx < curses.COLORS:
                return idx
            rgb = _xterm_rgb(idx)
        key = (rgb, curses.COLORS)
        if key in self._color_cache:
            return self._color_cache[key]
        if curses.COLORS >= 256:
            out = _rgb_to_xterm(rgb)
        elif curses.COLORS >= 16:
            out = _nearest_basic(rgb, 16)
        elif curses.COLORS >= 8:
            out = _nearest_basic(rgb, 8)
        else:
            out = -1
        self._color_cache[key] = out
        return out

    def pair_for(self, fg: Any, bg: Any) -> int:
        """Lazily allocate a curses colour pair for one Markdown span."""
        if not curses.has_colors():
            return 0
        f = self._color_index(fg)
        b = self._color_index(bg)
        if f < 0 and b < 0:
            return 0
        key = (f, b)
        if key in self._pair_cache:
            return self._pair_cache[key]
        try:
            limit = curses.COLOR_PAIRS
        except Exception:
            limit = 64
        if self._next_pair >= limit:
            return 0
        pid = self._next_pair
        self._next_pair += 1
        try:
            curses.init_pair(pid, f, b)
        except curses.error:
            return 0
        attr = curses.color_pair(pid)
        self._pair_cache[key] = attr
        return attr

    def span_attr(self, span: Span) -> int:
        if span.named:
            return self.attr(span.named)
        attr = self.pair_for(span.fg, span.bg)
        if span.bold:
            attr |= curses.A_BOLD
        if span.italic:
            attr |= getattr(curses, "A_ITALIC", 0)
        if span.underline:
            attr |= curses.A_UNDERLINE
        if span.dim:
            attr |= curses.A_DIM
        if span.reverse:
            attr |= curses.A_REVERSE
        return attr

    def _add(self, scr: "curses.window", row: int, col: int, text: str,
             width: int, attr: int = 0) -> None:
        if not text or width <= 0 or row < 0 or col < 0:
            return
        try:
            scr.addnstr(row, col, text, max(1, width), attr)
        except curses.error:
            pass

    def init_colors(self) -> None:
        self._pair_cache = {}
        self._color_cache = {}
        self._next_pair = 1
        self.pairs: dict[str, int] = {
            "default": 0, "value": 0, "dim": curses.A_DIM,
            "title": curses.A_BOLD, "status": 0, "active": curses.A_BOLD,
            "user": 0, "assistant": 0, "system": 0, "error": curses.A_BOLD,
            "section": curses.A_BOLD, "reasoning": curses.A_DIM,
            "input": 0, "prefill": 0, "decode": 0,
            "muted": curses.A_DIM, "cancel": curses.A_BOLD,
        }
        if not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
            specs = {
                "user": (curses.COLOR_CYAN, -1),
                "assistant": (curses.COLOR_GREEN, -1),
                "system": (curses.COLOR_YELLOW, -1),
                "error": (curses.COLOR_RED, -1),
                "section": (curses.COLOR_BLUE, -1),
                "active": (curses.COLOR_MAGENTA, -1),
                "status": (curses.COLOR_GREEN, -1),
                "title": (curses.COLOR_WHITE, curses.COLOR_BLUE),
                "value": (-1, -1),
                "input": (curses.COLOR_WHITE, -1),
                "prefill": (curses.COLOR_CYAN, -1),
                "decode": (curses.COLOR_GREEN, -1),
                "cancel": (curses.COLOR_RED, -1),
            }
            if curses.COLORS >= 256:
                specs["muted"] = (240, -1)   # dark grey, quieter than A_DIM
                specs["cancel"] = (88, -1)   # dark red
            for i, (name, (fg, bg)) in enumerate(specs.items(), start=1):
                curses.init_pair(i, fg, bg)
                extra = 0
                if name in ("title", "active", "error", "section"):
                    extra = curses.A_BOLD
                self.pairs[name] = curses.color_pair(i) | extra
            self._next_pair = len(specs) + 1
        except curses.error:
            pass

    # -- main loop ---------------------------------------------------------
    def run(self, stdscr: "curses.window") -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        stdscr.timeout(35)
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
            curses.mouseinterval(0)
        except curses.error:
            pass
        self.init_colors()
        self.dirty = True
        self.render(stdscr)

        while self.running:
            self.drain()
            now = time.monotonic()
            active = self.status in ("starting", "waiting") or self.generating
            if (self.dirty or (active and now - self.last_render > 0.12)
                    or now - self.last_render > 0.8):
                self.render(stdscr)

            try:
                ch = stdscr.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:
                self.running = False
                continue
            if ch == curses.KEY_RESIZE:
                stdscr.erase()
                self.dirty = True
                continue
            if ch == curses.KEY_MOUSE:
                self.handle_mouse()
                continue
            self.key(ch)
            if self.running:
                self.render(stdscr)
        self.close()


# -------------------------------------------------------------------- main --

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
    p.add_argument("--prefill-batch-size", type=int, default=16,
                   metavar="N",
                   help="dense prefill chunk size handed to dsv4-serve "
                        "(default: 16; 0 = per-token decode path)")
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
                  f"       run `make bin/dsv4-serve` first, or use --demo.",
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


if __name__ == "__main__":
    sys.exit(main())

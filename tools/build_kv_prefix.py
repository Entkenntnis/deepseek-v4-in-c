#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the precomputed KV cache for the `/tools` preamble.

The TUI sends exactly `BOS + system_prompt() + newline` in front of the first
user turn when tools are enabled. This script runs that same prefix through
the normal prefill path once and writes the resulting layer state to
`DEFAULT_KV` (or `--output`).

Examples:
    make kv
    python3 tools/build_kv_prefix.py
    python3 tools/build_kv_prefix.py --output /tmp/tools-prefix.bin
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tui"))

from dsv4tui import proto  # noqa: E402
from dsv4tui.defaults import (DEFAULT_BINARY, DEFAULT_BUDGET, DEFAULT_CTX,
                              DEFAULT_KV, DEFAULT_MODEL, DEFAULT_TOK,
                              DEFAULT_TRUNK)  # noqa: E402
from dsv4tui.dsml import tools_prefix_text  # noqa: E402
from dsv4tui.tokenizer import Tokenizer  # noqa: E402


def env_path(name: str, default: Path) -> Path:
    return Path(os.path.expanduser(os.environ.get(name, str(default))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=env_path("DSV4_MODEL", DEFAULT_MODEL))
    p.add_argument("--trunk", default=env_path("DSV4_TRUNK", DEFAULT_TRUNK))
    p.add_argument("--tok", default=env_path("DSV4_TOK", DEFAULT_TOK))
    p.add_argument("--binary", default=env_path("DSV4_SERVE_BIN", DEFAULT_BINARY))
    p.add_argument("--budget", default=DEFAULT_BUDGET,
                   help=f"GB or auto (default: {DEFAULT_BUDGET})")
    p.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    p.add_argument("--output", default=None,
                   help=f"cache file to write (default: {DEFAULT_KV})")
    return p.parse_args()


def send(proc: subprocess.Popen, obj: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write(proto.encode_command(obj) + "\n")
    proc.stdin.flush()


def read_event(proc: subprocess.Popen) -> dict | None:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        ev = proto.parse_event(line)
        if ev is None:
            print(f"build_kv: bad event: {line}", file=sys.stderr)
            return None
        return ev


def wait_for(proc: subprocess.Popen, event: str) -> dict:
    while True:
        ev = read_event(proc)
        if ev is None:
            raise RuntimeError("serve exited while waiting for " + event)
        kind = ev.get("event")
        if kind == "error":
            raise RuntimeError(ev.get("message", "KV cache error"))
        if kind == event:
            return ev


def build_command(a: argparse.Namespace) -> list[str]:
    cmd = [
        str(a.binary),
        "--model", str(a.model),
        "--trunk", str(a.trunk),
        "--budget", str(a.budget),
        "--ctx", str(a.ctx),
    ]
    return cmd


def resolve_output(a: argparse.Namespace) -> Path:
    if a.output:
        return Path(os.path.expanduser(str(a.output)))
    env = os.environ.get("DSV4_KV")
    if env:
        return Path(os.path.expanduser(env))
    name = Path(str(a.model).rstrip("/")).name or DEFAULT_MODEL.name
    return DEFAULT_KV.parent / f"{name}-tools-prefix.bin"


def main() -> int:
    a = parse_args()
    out = resolve_output(a)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not Path(a.binary).exists():
        print(f"build_kv: {a.binary} does not exist; run `make serve` first",
              file=sys.stderr)
        return 2

    try:
        tok = Tokenizer(a.tok)
    except (OSError, ValueError) as exc:
        print(f"build_kv: cannot load tokenizer {a.tok}: {exc}",
              file=sys.stderr)
        return 2
    prefix = tools_prefix_text()
    prefix_tokens = tok.encode(prefix)
    proc = subprocess.Popen(
        build_command(a),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        ready = wait_for(proc, "ready")
        cfg = ready.get("config") or {}
        print(f"build_kv: {cfg.get('layers', '?')} layers, "
              f"{cfg.get('hidden', '?')}d; pre-filling tools preamble",
              file=sys.stderr)
        send(proc, {"cmd": "kv_build", "path": str(out),
                    "tokens": prefix_tokens})
        ev = wait_for(proc, "kv_built")
        tokens = int(ev.get("tokens") or 0)
        seconds = float(ev.get("seconds") or 0.0)
        size_mb = float(ev.get("bytes") or 0) / (1024.0 * 1024.0)
        print(f"build_kv: wrote {out}")
        print(f"build_kv: {tokens} tokens, {seconds:.3f}s, {size_mb:.1f} MiB")
        return 0
    except (BrokenPipeError, RuntimeError) as exc:
        print(f"build_kv: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            send(proc, {"cmd": "quit"})
        except Exception:
            pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())

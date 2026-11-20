# SPDX-License-Identifier: Apache-2.0
"""The real engine process and the queue between it and the UI loop.

The child speaks the line protocol in proto.py. Text never crosses that pipe:
this class tokenizes prompts, detokenizes emitted ids in app_events.py, and
keeps the chat framing and system prompt on the Python side.
"""

from __future__ import annotations

import queue
import signal
import subprocess
import threading
from typing import Any, Optional

from . import proto
from .tokenizer import Tokenizer

try:
    import resource
except ImportError:  # non-Unix; the TUI also runs on Windows in demo mode
    resource = None


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


def _disable_core_dumps() -> None:
    """Make a crashed engine exit at once.

    The model process holds the checkpoint and cache resident; letting the
    kernel write a core of that size takes minutes and looks exactly like a
    frozen TUI. Cores are still available when the binary is started under gdb
    by hand.
    """
    if resource is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


class RealBackend:
    def __init__(self, argv: list[str], sink: EventSink,
                 tokenizer: Tokenizer) -> None:
        self.argv = argv
        self.sink = sink
        self.tokenizer = tokenizer
        self.proc: Optional[subprocess.Popen[str]] = None
        # Chat framing state. The engine itself is context-agnostic now.
        self.context_empty = True
        self.prefix_loaded = False
        self.system_prompt: Optional[str] = None

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
            preexec_fn=_disable_core_dumps,
            # Keep terminal Ctrl-C (SIGINT to the foreground process group)
            # away from the engine. The TUI handles cancel explicitly through
            # SIGUSR1; a stray SIGINT would terminate the child instead.
            start_new_session=True,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _track(self, ev: dict) -> None:
        name = ev.get("event")
        if name == "ready":
            self.context_empty = True
            self.prefix_loaded = False
        elif name == "start":
            self.context_empty = False
        elif name == "done":
            self.context_empty = int(ev.get("context_used") or 0) == 0
        elif name == "reset":
            self.context_empty = True
            self.prefix_loaded = False
            self.system_prompt = None
        elif name == "kv_loaded":
            self.context_empty = False
            self.prefix_loaded = True
            self.system_prompt = None

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                ev = proto.parse_event(line)
                if ev is None:
                    self.sink.log(f"bad event from engine: {line[:300]}")
                    continue
                self._track(ev)
                self.sink.event(ev)
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

    def _write(self, obj: dict) -> bool:
        if not self.proc or not self.proc.stdin or self.proc.poll() is not None:
            return False
        try:
            self.proc.stdin.write(proto.encode_command(obj) + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def send(self, obj: dict) -> bool:
        """Translate a UI command into token ids and write it to the engine."""
        name = obj.get("cmd")

        if name == "system":
            self.system_prompt = str(obj.get("text") or "")
            return True

        if name == "generate":
            first = self.context_empty and not self.prefix_loaded
            system = self.system_prompt if first else None
            prompt = str(obj.get("prompt") or "")
            tokens = self.tokenizer.turn_tokens(
                prompt, system=system, first=first)
            self.context_empty = False
            return self._write({
                "cmd": "generate",
                "tokens": tokens,
                "gen": int(obj.get("gen") or 0),
                "eos": self.tokenizer.eos_id,
            })

        if name == "kv_build":
            text = str(obj.get("text") or "")
            return self._write({
                "cmd": "kv_build",
                "path": str(obj.get("path") or ""),
                "tokens": self.tokenizer.encode(text),
            })

        if name == "kv_load":
            return self._write({
                "cmd": "kv_load",
                "path": str(obj.get("path") or ""),
            })

        if name == "reset":
            self.context_empty = True
            self.prefix_loaded = False
            self.system_prompt = None
            return self._write({"cmd": "reset"})

        if name in ("quit", "ping", "stats"):
            return self._write({"cmd": name})

        return False

    def cancel(self) -> bool:
        """Ask the engine to stop (SIGUSR1 in dsv4-serve).

        Decode stops after the current token; prefill stops at its next layer or
        token checkpoint. A mid-prefill stop restores the previous context.
        """
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
            self._write({"cmd": "quit"})
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

# SPDX-License-Identifier: Apache-2.0
"""The real engine process and the queue between it and the UI loop."""

from __future__ import annotations

import json
import queue
import signal
import subprocess
import threading
from typing import Any, Optional

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
            preexec_fn=_disable_core_dumps,
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

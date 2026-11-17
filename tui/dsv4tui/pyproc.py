# SPDX-License-Identifier: Apache-2.0
"""Process plumbing for one persistent interpreter child.

Kept apart from pytool.py so the harness/reply logic stays readable: this file
only knows how to spawn, feed and kill a JSON-lines child and read one line
back with a deadline. No curses, no TUI state.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
import threading
import time
from collections import deque

try:
    import resource
except ImportError:  # non-Unix; the TUI also runs on Windows in demo mode
    resource = None


def _limits() -> None:
    """Keep a runaway script from filling the disk or dumping core."""
    if resource is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    except (OSError, ValueError):
        pass


class PyProc:
    """A python child whose stdin/stdout carry one JSON object per line."""

    def __init__(self, code: str, python: str | None = None) -> None:
        self.code = code
        self.python = python or os.environ.get("DSV4_PYTHON", "python3")
        self.proc: "subprocess.Popen[bytes] | None" = None
        self.started_once = False
        self.cwd = ""
        self._buf = b""
        self._stderr: deque[str] = deque(maxlen=20)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def spawn(self) -> None:
        if not self.cwd:
            self.cwd = tempfile.mkdtemp(prefix="dsv4-py-")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", tempfile.gettempdir()),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.proc = subprocess.Popen(
            [self.python, "-u", "-I", "-q", "-c", self.code],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, cwd=self.cwd, env=env,
            start_new_session=True, preexec_fn=_limits,
        )
        self._buf = b""
        self._stderr.clear()
        self.started_once = True
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        proc = self.proc
        if not proc or not proc.stderr:
            return
        try:
            for line in iter(proc.stderr.readline, b""):
                self._stderr.append(line.decode("utf-8", "replace").rstrip())
        except Exception:
            pass

    def kill(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except Exception:
                pass

    def stderr_tail(self) -> str:
        return " | ".join(list(self._stderr)[-3:])

    def _read_line(self, deadline: float) -> "str | None":
        """One reply line; None on timeout, "" on EOF."""
        proc = self.proc
        if not proc or not proc.stdout:
            return ""
        fd = proc.stdout.fileno()
        while b"\n" not in self._buf:
            remain = deadline - time.monotonic()
            if remain <= 0:
                return None
            try:
                ready, _, _ = select.select([fd], [], [], remain)
            except (OSError, ValueError):
                return ""
            if not ready:
                return None
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return ""
            if not chunk:
                line, self._buf = self._buf, b""
                return line.decode("utf-8", "replace")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode("utf-8", "replace")

    def request(self, payload: dict, timeout: float) -> tuple[str, str]:
        """Send one request. Returns (status, line) with status ok|timeout|eof."""
        proc = self.proc
        if not proc or not proc.stdin:
            return "eof", ""
        started = time.monotonic()
        try:
            proc.stdin.write(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return "eof", ""
        line = self._read_line(started + timeout)
        if line is None:
            return "timeout", ""
        if line == "":
            return "eof", ""
        return "ok", line

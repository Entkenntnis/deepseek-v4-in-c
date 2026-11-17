# SPDX-License-Identifier: Apache-2.0
"""A persistent CPython session behind the JSON-lines pipe in pyproc.py.

One interpreter stays alive for the whole TUI session so variables and imports
survive between calls -- a one-shot `python3 -c` per call would force the model
to rebuild its working state every round. This module owns the harness, the
reply interpretation and the async wrapper; the process plumbing lives in
pyproc.py.

`run_async` is called from the UI thread and its callback runs on a worker, so
callers must hand results back through the event queue (see ToolMixin).
"""

from __future__ import annotations

import json
import threading
import time

from .pyproc import PyProc


HARNESS = r'''
import sys, json, io, ast, traceback, contextlib

_proto = sys.stdin

class _NoStdin:
    def read(self, *a, **k):
        raise RuntimeError("stdin is unavailable; input() is disabled")
    readline = read
    def __iter__(self):
        raise RuntimeError("stdin is unavailable; input() is disabled")

sys.stdin = _NoStdin()
ns = {"__name__": "__main__"}

def _run(code):
    tree = ast.parse(code, "<python>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
        ast.fix_missing_locations(last)
    out, err = io.StringIO(), io.StringIO()
    ok = True
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(tree, "<python>", "exec"), ns)
            if last is not None:
                value = eval(compile(last, "<python>", "eval"), ns)
                ns["_"] = value
                if value is not None:
                    print(repr(value))
    except BaseException:
        ok = False
        err.write(traceback.format_exc())
    return {"ok": ok, "stdout": out.getvalue(), "stderr": err.getvalue()}

for _line in _proto:
    try:
        _code = json.loads(_line)["code"]
        _res = _run(_code)
    except BaseException as _exc:
        _res = {"ok": False, "stdout": "", "stderr": "harness error: %r" % (_exc,)}
    sys.stdout.write(json.dumps(_res) + "\n")
    sys.stdout.flush()
'''


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    tail = limit - head
    return (text[:head]
            + f"\n... [{len(text) - limit} chars elided] ...\n"
            + text[-tail:])


class PySession:
    def __init__(self, timeout: float = 30.0, max_output: int = 6000) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self.proc = PyProc(HARNESS)
        self._io = threading.Lock()
        self._busy = threading.Lock()
        self._cancelled = False

    def stop(self) -> None:
        self.proc.kill()

    def reset(self) -> None:
        """Drop all user state; the next run starts a fresh interpreter."""
        self.proc.kill()

    def cancel(self) -> None:
        self._cancelled = True
        self.proc.kill()

    def run(self, code: str) -> dict:
        """Execute `code`, blocking the calling thread. Never raises."""
        with self._io:
            self._cancelled = False
            note = ""
            if not self.proc.alive():
                if self.proc.started_once:
                    note = "interpreter restarted; earlier variables are gone"
                self.proc.spawn()
            started = time.monotonic()
            status, line = self.proc.request({"code": code}, self.timeout)
            elapsed = time.monotonic() - started
            if self._cancelled:
                self._cancelled = False
                self.proc.kill()
                return self._failure("cancelled by user", started)
            if status == "timeout":
                self.proc.kill()
                return self._failure(
                    f"timed out after {self.timeout:g}s; interpreter restarted, "
                    "variables are gone", started)
            if status == "eof":
                tail = self.proc.stderr_tail()
                self.proc.kill()
                return self._failure(
                    "interpreter exited" + (f": {tail}" if tail else "")
                    + "; restarted, variables are gone", started)
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                self.proc.kill()
                return self._failure("unreadable reply from interpreter; "
                                     "restarted, variables are gone", started)
            if note:
                result["note"] = note
            result["stdout"] = _truncate(
                str(result.get("stdout") or ""), self.max_output)
            result["stderr"] = _truncate(
                str(result.get("stderr") or ""), self.max_output)
            result["seconds"] = round(elapsed, 3)
            return result

    def _failure(self, message: str, started: float) -> dict:
        return {"ok": False, "stdout": "", "stderr": message,
                "seconds": round(time.monotonic() - started, 3)}

    def run_async(self, code: str, on_done) -> bool:
        """Run on a worker thread; `on_done(result)` runs on that thread.

        Returns False when a call is already in flight. Completions are
        delivered exactly once, including for internal failures.
        """
        if not self._busy.acquire(blocking=False):
            return False

        def work() -> None:
            try:
                result = self.run(code)
            except Exception as exc:  # defensive: a bug here must not wedge
                result = self._failure(f"tool error: {exc!r}", time.monotonic())
            finally:
                self._busy.release()
            try:
                on_done(result)
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()
        return True

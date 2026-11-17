# SPDX-License-Identifier: Apache-2.0
"""A one-shot CPython runner behind the JSON-lines pipe in pyproc.py.

Every call starts a new interpreter process with a fresh global namespace, so
variables and imports from one call are never visible to the next. This module
owns the harness, the reply interpretation and the async wrapper; the process
plumbing lives in pyproc.py.

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

def _run(code):
    ns = {"__name__": "__main__"}
    out, err = io.StringIO(), io.StringIO()
    ok = True
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            tree = ast.parse(code, "<python>", "exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
                ast.fix_missing_locations(last)
            exec(compile(tree, "<python>", "exec"), ns)
            if last is not None:
                value = eval(compile(last, "<python>", "eval"), ns)
                if value is not None:
                    print(repr(value))
    except BaseException:
        ok = False
        err.write(traceback.format_exc())
    return {"ok": ok, "stdout": out.getvalue(), "stderr": err.getvalue()}

try:
    _code = json.loads(_proto.readline())["code"]
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
        self.proc: PyProc | None = None
        self._io = threading.Lock()
        self._busy = threading.Lock()
        self._cancelled = False

    def _kill(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None:
            proc.kill()

    def stop(self) -> None:
        self._kill()

    def reset(self) -> None:
        """Stop any in-flight call; every call already starts fresh."""
        self._kill()

    def cancel(self) -> None:
        self._cancelled = True
        self._kill()

    def run(self, code: str) -> dict:
        """Execute `code` in a fresh interpreter. Never raises."""
        with self._io:
            self._cancelled = False
            started = time.monotonic()
            proc = PyProc(HARNESS)
            self.proc = proc
            tail = ""
            try:
                try:
                    proc.spawn()
                except OSError as exc:
                    return self._failure(
                        f"could not start interpreter: {exc}", started)
                if self._cancelled:
                    status, line = "eof", ""
                else:
                    status, line = proc.request({"code": code}, self.timeout)
                    tail = proc.stderr_tail()
            finally:
                proc.kill()
                if self.proc is proc:
                    self.proc = None
            elapsed = time.monotonic() - started
            if self._cancelled:
                self._cancelled = False
                return self._failure("cancelled by user", started)
            if status == "timeout":
                return self._failure(
                    f"timed out after {self.timeout:g}s", started)
            if status == "eof":
                return self._failure(
                    "interpreter exited" + (f": {tail}" if tail else ""),
                    started)
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                return self._failure(
                    "unreadable reply from interpreter", started)
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

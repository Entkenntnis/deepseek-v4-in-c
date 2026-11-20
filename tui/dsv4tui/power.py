# SPDX-License-Identifier: Apache-2.0
"""Keep the machine out of its idle power-saving path while work is running.

The terminal cannot do this with an escape sequence. On Linux with systemd the
right place is a logind inhibitor. We ask ``systemd-inhibit`` to run ``cat`` and
hold its stdin open: while the pipe is open the idle inhibitor is active, and
closing it (normal end, crash, SIGKILL) makes cat exit and releases the lock.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional


class IdleInhibitor:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen[bytes]] = None

    def _start(self) -> None:
        exe = shutil.which("systemd-inhibit")
        if exe is None:
            return
        try:
            self.proc = subprocess.Popen(
                [exe, "--what=idle", "--mode=block",
                 "--who=dsv4-tui", "--why=model turn", "cat"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            self.proc = None

    def begin(self) -> None:
        """Hold the inhibitor. Idempotent."""
        if self.proc is None:
            self._start()

    def end(self) -> None:
        """Release the inhibitor. Safe when it was never acquired."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def sync(self, active: bool) -> None:
        if active:
            self.begin()
        else:
            self.end()

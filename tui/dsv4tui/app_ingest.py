# SPDX-License-Identifier: Apache-2.0
"""Drain the event queue and turn host samples into rates."""

from __future__ import annotations

import time
from pathlib import Path

from .messages import Message


class IngestMixin:
    def drain(self) -> None:
        for kind, payload in self.sink.get_all():
            if kind == "event":
                self.handle_event(payload)
            elif kind == "stderr":
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
        if self.prev_snap is not None:
            dt = now - self.prev_snap_t
            # A forced update (start/done) can land milliseconds after the last
            # sample. Rates over that window are noise, not information; keep
            # the previous ones until a real sampling interval has passed. Do
            # not move the baseline meanwhile, or a stream of sub-0.2 s prefill
            # events would reset the window forever and the disk rate would
            # never be recomputed.
            if dt >= 0.2:
                self.io_rate = max(0.0, (sysd.get("io_read_gb", 0.0)
                                         - self.prev_snap.get("io_read_gb", 0.0))
                                   / dt)
                self.prev_snap = dict(sysd)
                self.prev_snap_t = now
        else:
            self.prev_snap = dict(sysd)
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

# SPDX-License-Identifier: Apache-2.0
"""Synthetic engine with the same event shape as dsv4-serve."""

from __future__ import annotations

import queue
import random
import threading
import time
from typing import Any

from .backend import EventSink
from .defaults import DEFAULT_CTX
from .demo_gen import DemoGenerateMixin


class DemoBackend(DemoGenerateMixin):
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
        self.read_gb = 0.0
        self.t0 = time.monotonic()

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

    def _sys(self, rss_mb: float) -> dict:
        return {
            "rss_mb": rss_mb,
            "io_read_gb": self.read_gb,
        }

    def _cache(self) -> dict:
        n = self.cache_hits + self.cache_miss
        return {
            "slots": 292,
            "hit": self.cache_hits,
            "miss": self.cache_miss,
            "hit_rate": (self.cache_hits / n) if n else 0.0,
            "read_gb": self.read_gb * 0.78,
            "resident_gb": 3.55,
            "slot_mb": 12.77,
        }

    def _trunk(self) -> dict:
        return {
            "layers": 43,
            "resident_gb": 6.43,
        }

    def _base(self, elapsed: float = 0.0, rss: float = 2060.0) -> dict:
        del elapsed
        return {
            "sys": self._sys(rss),
            "cache": self._cache(),
            "trunk": self._trunk(),
        }

    def _event(self, name: str, elapsed: float = 0.0, **fields: Any) -> None:
        obj = {"event": name}
        obj.update(fields)
        obj.update(self._base(elapsed))
        self.sink.event(obj)

    def _run(self) -> None:
        time.sleep(0.35)
        self.sink.event({
            "event": "ready",
            "config": {
                "hidden": 4096, "layers": 43, "experts": 256, "topk": 6,
            },
            "context": {"used": 0, "cap": DEFAULT_CTX},
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
            elif name == "ping":
                self.sink.event({"event": "pong"})
            elif name == "generate":
                self._generate(cmd)
        self.sink.eof(0)

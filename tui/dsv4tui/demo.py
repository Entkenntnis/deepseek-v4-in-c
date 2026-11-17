# SPDX-License-Identifier: Apache-2.0
"""Synthetic engine with the same event shape as dsv4-serve."""

from __future__ import annotations

import queue
import random
import threading
import time
from typing import Any

from .backend import EventSink
from .defaults import DEFAULT_CTX, DEFAULT_GEN
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
        self.cache_evict = 0
        self.trunk_hit = 0
        self.trunk_miss = 0
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
        # Proportions roughly matching the decode profile: disk is the biggest
        # single component, then attention, then expert matmul.
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
            "context": {"used": 0, "cap": DEFAULT_CTX},
            "gen_default": DEFAULT_GEN,
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

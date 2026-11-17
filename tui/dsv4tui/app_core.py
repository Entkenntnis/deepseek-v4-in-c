# SPDX-License-Identifier: Apache-2.0
"""App state: construction, backend selection and shutdown."""

from __future__ import annotations

import argparse
import codecs
from collections import deque
from typing import Any, Optional

from .backend import EventSink, RealBackend
from .defaults import (DEFAULT_BINARY, DEFAULT_BUDGET, DEFAULT_CTX, DEFAULT_GEN,
                       DEFAULT_MODEL, DEFAULT_TOK, DEFAULT_TRUNK)
from .demo import DemoBackend
from .messages import Message


class AppBase:
    def __init__(self, args: argparse.Namespace, sink: EventSink) -> None:
        self.sink = sink
        self.backend: Any = None

        self.messages: list[Message] = []
        self.current_msg: Optional[Message] = None
        self.current_user_msg: Optional[Message] = None
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

        self.ready: dict = {}
        self.last: dict = {}
        self.prev_snap: Optional[dict] = None
        self.prev_snap_t = 0.0
        self.last_stats_ingest = 0.0
        self.stats_interval = 1.0
        self.stats_cache: list[list[tuple[str, str]]] = []
        self.stats_cache_ts = 0.0
        self.stats_cache_width = -1
        self.stats_cache_height = -1
        self.stats_force = True
        self.io_rate = 0.0
        self.mem_available_gb = 0.0
        self.cpu_prev: Optional[dict[int, tuple[int, int]]] = None
        self.cpu_prev_t = 0.0
        self.core_pct: list[Optional[float]] = [None, None, None, None]
        self.cpu_temp_c: Optional[float] = None
        self.cpu_temp_path: Optional[str] = None
        self.cpu_temp_checked = False

        self.status = "starting"
        self.generating = False
        self.cancel_pending = False
        self.cancel_on_start = False
        self.cancel_rect: Optional[tuple[int, int, int]] = None
        self.chat_rect: Optional[tuple[int, int, int]] = None
        self.max_scroll = 0
        self.backend_alive = True
        self.generation = 0
        self.gen_max = DEFAULT_GEN
        self.prompt_tokens = 0
        self.context_used = 0
        self.context_cap = DEFAULT_CTX
        self.turn_start = 0.0
        self.last_token_t = 0.0
        # One duration per decoded token, in turns.
        self.decode_spt: deque[float] = deque(maxlen=4096)
        self.prefill_size = 0
        self.prefill_progress: dict = {}
        self.prefill_active = False
        self.prefill_s = 0.0
        self.prefill_tokens = 0
        # The prompt's elapsed time and rate are 1 Hz samples: the bar and its
        # counters repaint on every prefill event, but a clock or tok/s figure
        # recomputed from sub-second progress windows is noise.
        self.prefill_elapsed = 0.0
        self.prefill_rate = 0.0
        self.prefill_sample_ts = 0.0
        self.ttft = 0.0
        self.current_layer = 0
        self.phase = ""
        self.last_done: dict = {}

        self.gen_limit = DEFAULT_GEN

        self.input = ""
        self.input_cursor = 0
        self.scroll = 0
        self.spinner = 0
        self.stderr_lines: deque[str] = deque(maxlen=8)
        self.running = True
        self.dirty = True
        self.last_render = 0.0
        self.demo = bool(args.demo)
        self.pairs: dict[str, int] = {}
        self.md_warned = False
        self._pair_cache: dict[tuple[int, int], int] = {}
        self._color_cache: dict[Any, int] = {}
        self._next_pair = 1

    def decode_avg_spt(self) -> float:
        """Mean seconds per decoded token, excluding prefill and TTFT."""
        if not self.decode_spt:
            return 0.0
        return sum(self.decode_spt) / len(self.decode_spt)

    def decode_avg_tps(self) -> float:
        """Mean decode rate, matching the sidebar's decode average."""
        spt = self.decode_avg_spt()
        return 1.0 / spt if spt > 0 else 0.0

    def start_backend(self) -> None:
        if self.demo:
            self.backend = DemoBackend(self.sink)
        else:
            argv = [
                str(DEFAULT_BINARY),
                str(DEFAULT_MODEL),
                "--trunk", str(DEFAULT_TRUNK),
                "--tok", str(DEFAULT_TOK),
                "--budget", DEFAULT_BUDGET,
                "--ctx", str(DEFAULT_CTX),
                "--gen", str(DEFAULT_GEN),
            ]
            self.backend = RealBackend(argv, self.sink)
        self.backend.start()

    def close(self) -> None:
        if self.backend:
            self.backend.close()

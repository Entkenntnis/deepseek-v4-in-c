# SPDX-License-Identifier: Apache-2.0
"""App state: construction, backend selection and shutdown."""

from __future__ import annotations

import argparse
import codecs
from collections import deque
from typing import Any, Optional

from .backend import EventSink, RealBackend
from .demo import DemoBackend
from .messages import Message


class AppBase:
    def __init__(self, args: argparse.Namespace, sink: EventSink) -> None:
        self.args = args
        self.sink = sink
        self.backend: Any = None

        self.messages: list[Message] = []
        self.current_msg: Optional[Message] = None
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
        self.cache_rate = 0.0
        self.trunk_rate = 0.0
        self.mem_available_gb = 0.0
        self.cpu_pct = 0.0
        self.cpu_prev: Optional[dict[int, tuple[int, int]]] = None
        self.cpu_prev_t = 0.0
        self.core_pct: list[Optional[float]] = [None, None, None, None]
        self.cpu_temp_c: Optional[float] = None
        self.cpu_temp_path: Optional[str] = None
        self.cpu_temp_checked = False

        self.status = "starting"
        self.status_detail = ""
        self.generating = False
        self.cancel_pending = False
        self.cancel_rect: Optional[tuple[int, int, int]] = None
        self.chat_rect: Optional[tuple[int, int, int]] = None
        self.max_scroll = 0
        self.backend_alive = True
        self.generation = 0
        self.gen_max = args.gen
        self.prompt_tokens = 0
        self.context_used = 0
        self.context_cap = args.ctx
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
        self.top: list = []
        self.last_done: dict = {}
        self.mode = "chat"

        self.system_prompt = args.system or ""
        self.think = bool(args.think)
        self.gen_limit = int(args.gen)
        self.show_layers = not args.no_layers

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
        self.markdown_enabled = not args.no_markdown
        self.md_warned = False
        self._pair_cache: dict[tuple[int, int], int] = {}
        self._color_cache: dict[Any, int] = {}
        self._next_pair = 1

    def start_backend(self) -> None:
        if self.demo:
            self.backend = DemoBackend(self.sink)
        else:
            budget = (self.args.budget if isinstance(self.args.budget, str)
                      else f"{self.args.budget:.2f}")
            argv = [
                str(self.args.binary),
                str(self.args.model),
                "--trunk", str(self.args.trunk),
                "--tok", str(self.args.tok),
                "--budget", budget,
                "--ctx", str(self.args.ctx),
                "--gen", str(self.args.gen),
            ]
            # The TUI follows the engine's thread policy without duplicating it.
            if self.args.threads is not None:
                argv += ["--threads", str(self.args.threads)]
            if self.args.pin is not None:
                argv += ["--pin", self.args.pin]
            if self.args.readers is not None:
                argv += ["--readers", str(self.args.readers)]
            if self.args.reader_pin is not None:
                argv += ["--reader-pin", self.args.reader_pin]
            if self.args.think:
                argv.append("--think")
            if self.args.no_layers:
                argv.append("--no-layers")
            self.backend = RealBackend(argv, self.sink)
        self.backend.start()
        if self.system_prompt:
            self.backend.send({"cmd": "system", "text": self.system_prompt})

    def close(self) -> None:
        if self.backend:
            self.backend.close()

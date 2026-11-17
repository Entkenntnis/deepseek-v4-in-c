# SPDX-License-Identifier: Apache-2.0
"""Synthetic token stream: the demo backend's generate half."""

from __future__ import annotations

import re
import time
from typing import Any


class DemoGenerateMixin:


    def _response(self, prompt: str, think: bool) -> str:
        low = prompt.lower()
        if "code" in low or "c function" in low or "program" in low:
            return (
                "Here is a small C function that adds two integers, with the "
                "kind of checks this repository cares about:\n\n"
                "```c\n"
                "#include <limits.h>\n\n"
                "int add_checked(int a, int b, int *out)\n"
                "{\n"
                "    if ((b > 0 && a > INT_MAX - b) ||\n"
                "        (b < 0 && a < INT_MIN - b)) return 0;\n"
                "    *out = a + b;\n"
                "    return 1;\n"
                "}\n"
                "```\n\n"
                "The overflow check is the part that matters: signed overflow is "
                "undefined behaviour in C, not a wrap.\n"
            )
        if "hello" in low or "hi" in low[:8]:
            return (
                "Hello. This is the demo backend, so no NVMe was harmed. The "
                "numbers in the right-hand panel are synthetic but have the same "
                "shape as the real engine's events: per-layer progress, expert "
                "cache accounting, trunk reads, RSS, disk bytes and a profile "
                "split into I/O and matmul.\n"
            )
        return (
            f'You said: "{prompt.strip()}"\n\n'
            "This is the synthetic backend. Start without --demo and point the "
            "TUI at a checkpoint to stream real tokens from DeepSeek-V4-Flash. "
            "The panel on the right will then show the actual expert-cache hit "
            "rate and the disk traffic, which is where most of the wall clock "
            "goes at this model size.\n"
        )

    def _generate(self, cmd: dict) -> None:
        prompt = cmd.get("prompt") or ""
        ngen = int(cmd.get("gen") or 128)
        think = bool(cmd.get("think"))
        layers = bool(cmd.get("layers", True))
        mode = "thinking" if think else "chat"
        prompt_tokens = 9
        self.cancel_requested = False
        stop = "max"

        self.sink.event({
            "event": "start", "pos": 0, "prompt_tokens": prompt_tokens,
            "gen_max": ngen, "mode": mode, "prefill_tokens": 0,
            "context_cap": 4096, **self._base(),
        })

        text = self._response(prompt, think)
        # Split into plausible token-sized chunks, keeping whitespace attached.
        chunks = re.findall(r"\s+|[^\s]+", text)
        t0 = time.monotonic()
        ttft = 0.0
        produced = 0
        context = prompt_tokens

        def forward(pos: int, phase: str, t_start: float) -> None:
            """Simulate one forward pass, including its per-layer events."""
            time.sleep(0.012)
            for L in range(43):
                if L % 8 == 0:
                    time.sleep(0.002)
                if self.rng.random() < 0.55:
                    self.cache_hits += 1
                else:
                    self.cache_miss += 1
                    if self.rng.random() < 0.8:
                        self.cache_evict += 1
                self.trunk_hit += 1
                self.read_gb += 0.00018
                if layers:
                    # The engine stamps every layer of a forward with the time
                    # the forward *started*, so the TUI can difference prompt
                    # positions into a prefill timeline.
                    self._event("layer", t_start, pos=pos, phase=phase,
                                layer=L, layers=43, t=t_start)

        # Prefill: one forward per prompt token, then the first token event.
        for p in range(prompt_tokens):
            if self.stop.is_set():
                return
            forward(p, "prompt", time.monotonic() - t0)
        ttft = time.monotonic() - t0

        for i, chunk in enumerate(chunks[:ngen]):
            if self.stop.is_set():
                return
            # A token takes ~2.1 s on the reference machine. The demo runs at
            # 1/12 speed so a conversation is watchable.
            if i > 0:
                forward(context - 1, "gen", time.monotonic() - t0)
            if self.cancel_requested:
                stop = "cancel"
                break

            elapsed = time.monotonic() - t0
            self.read_gb += 0.00055
            p = max(0.02, 0.97 - i * 0.004)
            top = [
                [101 + i, p, 31.0 - i * 0.05],
                [102 + i, (1 - p) * 0.5, 29.0],
                [103 + i, (1 - p) * 0.3, 28.0],
                [104 + i, (1 - p) * 0.2, 27.0],
            ]
            self.sink.event({
                "event": "token", "id": 1000 + i,
                "hex": chunk.encode("utf-8").hex(), "index": i,
                "pos": context, "t": elapsed, "ttft": ttft,
                "prefill_s": ttft, "prefill_tokens": prompt_tokens,
                "top": top, **self._base(elapsed),
            })
            context += 1
            produced += 1

        elapsed = time.monotonic() - t0
        self.sink.event({
            "event": "done", "prompt_tokens": prompt_tokens,
            "generated": produced, "seconds": elapsed,
            "tps": produced / elapsed if elapsed else 0.0,
            "s_per_token": elapsed / produced if produced else 0.0,
            "ttft": ttft, "stop": stop, "context_used": context,
            **self._base(elapsed),
        })

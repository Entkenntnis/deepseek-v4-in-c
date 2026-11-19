# SPDX-License-Identifier: Apache-2.0
"""Synthetic token stream: the demo backend's generate half."""

from __future__ import annotations

import re
import time

from .defaults import DEFAULT_CTX, DEFAULT_GEN
from .dsml import DSML, TOOL_NAME


class DemoGenerateMixin:


    def _response(self, prompt: str) -> str:
        low = prompt.lower()
        if prompt.startswith("<tool_result>"):
            return (
                "The interpreter ran the code and returned the result above. "
                "That is the full round trip: model -> tool call -> Python -> "
                "tool result -> model. In a real session the next paragraph "
                "would use those numbers to answer your question.\n"
            )
        if "python" in low or "calculate" in low or "compute" in low:
            return (
                "I will use the interpreter for that.\n\n"
                f"<{DSML}tool_calls>\n"
                f"<{DSML}invoke name=\"{TOOL_NAME}\">\n"
                f"<{DSML}parameter name=\"code\" string=\"true\">"
                "total = sum(range(1, 11))\nprint(total)"
                f"</{DSML}parameter>\n"
                f"</{DSML}invoke>\n"
                f"</{DSML}tool_calls>"
            )
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
                "cache accounting, trunk reads, RSS and disk bytes.\n"
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
        ngen = int(cmd.get("gen") or DEFAULT_GEN)
        prompt_tokens = 9
        self.cancel_requested = False
        stop = "max"

        self.sink.event({
            "event": "start", "pos": 0, "prompt_tokens": prompt_tokens,
            "gen_max": ngen, "prefill_tokens": 0,
            "context_cap": DEFAULT_CTX, **self._base(),
        })

        text = self._response(prompt)
        # Split into plausible token-sized chunks, keeping whitespace attached.
        chunks = re.findall(r"\s+|[^\s]+", text)
        t0 = time.monotonic()
        ttft = 0.0
        produced = 0
        context = prompt_tokens

        def forward(phase: str) -> None:
            """Simulate one forward pass, including its per-layer events."""
            time.sleep(0.012)
            for L in range(43):
                if L % 8 == 0:
                    time.sleep(0.002)
                if self.rng.random() < 0.55:
                    self.cache_hits += 1
                else:
                    self.cache_miss += 1
                self.read_gb += 0.00018
                self._event("layer", phase=phase, layer=L)

        # Prefill: one forward per prompt token, then the first token event.
        for _ in range(prompt_tokens):
            if self.stop.is_set():
                return
            forward("prompt")
        ttft = time.monotonic() - t0

        for i, chunk in enumerate(chunks[:ngen]):
            if self.stop.is_set():
                return
            # Small per-layer sleeps keep the synthetic stream watchable.
            if i > 0:
                forward("gen")
            if self.cancel_requested:
                stop = "cancel"
                break

            elapsed = time.monotonic() - t0
            self.read_gb += 0.00055
            self.sink.event({
                "event": "token", "id": 1000 + i,
                "hex": chunk.encode("utf-8").hex(), "index": i,
                "t": elapsed, "ttft": ttft,
                "prefill_s": ttft, "prefill_tokens": prompt_tokens,
                **self._base(elapsed),
            })
            context += 1
            produced += 1

        elapsed = time.monotonic() - t0
        self.sink.event({
            "event": "done", "prompt_tokens": prompt_tokens,
            "generated": produced, "seconds": elapsed,
            "s_per_token": elapsed / produced if produced else 0.0,
            "ttft": ttft, "stop": stop, "context_used": context,
            "head_s": 0.0, **self._base(elapsed),
        })

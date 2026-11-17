# SPDX-License-Identifier: Apache-2.0
"""Compact numbers and the live PREFILL block."""

from __future__ import annotations

import time

from .defaults import clamp


class StatsMixin:
    @staticmethod
    def _short(value: float) -> str:
        """A compact number for the header readout."""
        if value >= 100.0:
            return f"{value:.0f}"
        if value >= 10.0:
            return f"{value:.1f}"
        if value >= 1.0:
            return f"{value:.2f}"
        return f"{value:.3f}"

    @staticmethod
    def _short_spt(value: float) -> str:
        """Seconds per token: two decimals is enough beside tok/s."""
        if value >= 100.0:
            return f"{value:.0f}"
        if value >= 10.0:
            return f"{value:.1f}"
        if value >= 0.01:
            return f"{value:.2f}"
        return f"{value:.3f}"

    def prefill_lines(self, width: int) -> list[list[tuple[str, str]]]:
        """Build the PREFILL section: bar, percentage, layer/token counters.

        This is rebuilt on every render while a prompt runs, so progress
        follows the engine's prefill events instead of the 1 Hz stats cache.
        The elapsed clock and estimated tok/s are the exceptions: they are
        resampled at most once per `stats_interval`, because values recomputed
        from a sub-second window jitter without telling the reader anything.
        """
        out: list[list[tuple[str, str]]] = []

        def emit(*segments: tuple[str, str]) -> None:
            out.append(list(segments))

        p = self.prefill_progress or {}
        stage = p.get("stage") or "start"
        # The whole prompt is one block. Its token count is fixed for the turn.
        size = int(p.get("tokens_total") or self.prefill_size or 0)
        layer = int(p.get("layer") or 0)
        layers = int(p.get("layers") or 0)

        final = self.prefill_tokens > 0 and self.prefill_s > 0.0
        if final:
            # The engine's own prompt timer, measured without the head.
            tokens = self.prefill_tokens
            seconds = self.prefill_s
            total = done = tokens
            status = "done"
        else:
            tokens = int(p.get("tokens_done") or 0)
            seconds = float(p.get("elapsed") or 0.0)
            total = int(p.get("tokens_total") or self.prompt_tokens or 0)
            done = tokens
            if stage == "layer":
                status = f"layer {layer}/{layers}"
            elif stage == "head":
                status = "output head"
            elif stage == "done":
                status = "prompt done"
            else:
                status = "preparing"

        seg: list[tuple[str, str]] = [
            (" PREFILL", "section"),
            (f" {status}", "muted" if status == "done" else "active")]
        if total:
            seg.append((f"  {total} tokens", "value"))
        emit(*seg)

        if final:
            spt = seconds / tokens
            tps = tokens / seconds
            mode = "; per-token" if size <= 1 else ""
            emit((" ", "dim"),
                 (f"{seconds:.2f} s "
                  f"({tps:.2f} tok/s, {spt:.3f} s/tok{mode})",
                  "value"))
        else:
            prog = clamp(float(p.get("progress") or 0.0), 0.0, 1.0)
            elapsed = float(p.get("elapsed") or 0.0)
            bar_w = max(8, min(28, width - 18))
            filled = int(round(prog * bar_w))
            bar = "█" * filled + "░" * (bar_w - filled)
            now = time.monotonic()
            if now - self.prefill_sample_ts >= self.stats_interval:
                self.prefill_elapsed = elapsed
                self.prefill_sample_ts = now
                if elapsed > 0.05:
                    self.prefill_rate = prog * total / elapsed
            rate = self.prefill_rate
            elapsed = self.prefill_elapsed
            emit((" [", "dim"), (bar, "prefill"), ("]", "dim"))
            emit((" ", "dim"), (f"{prog * 100:3.0f}%", "value"),
                 (f"  {elapsed:5.0f}s", "dim"),
                 ((f"  {self._short(rate)} tok/s", "dim") if rate > 0
                  else ("  -", "dim")))
        emit()
        return out

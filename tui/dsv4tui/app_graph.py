# SPDX-License-Identifier: Apache-2.0
"""Decode tok/s graph plus the MEMORY and CPU blocks."""

from __future__ import annotations

import math
from typing import Optional

from .defaults import axis_label, clamp, fmt_gbps, nice_step


class GraphMixin:
    def stats_lines(self, width: int,
                    height: int) -> list[list[tuple[str, str]]]:
        """Decode tok/s graph plus live memory and CPU numbers.

        The graph is a right-aligned sliding window: new samples appear at the
        right edge and older ones move out to the left. The header carries the
        current tok/s and s/tok; the axis uses round, linear tick labels. A
        compact MEMORY block below it shows RSS, the expert cache, disk reads
        and the approximate byte inflow to the CPU. The PREFILL section is
        built separately by `prefill_lines` so it can repaint faster.
        """
        out: list[list[tuple[str, str]]] = []

        def emit(*segments: tuple[str, str]) -> None:
            out.append(list(segments))

        in_prefill = self.prefill_active
        show_prefill = in_prefill or self.prefill_tokens > 0
        # Keep the last turn's graph on screen after `done`; it is replaced by
        # the next turn's decode data once that turn produces tokens.
        has_decode = bool(self.decode_spt)
        in_decode = (not in_prefill
                     and self.phase not in ("prompt", "resume")
                     and (self.generating or has_decode))

        samples = [("decode", 1.0 / dt) for dt in self.decode_spt
                   if dt > 1e-9]

        cur_spt = 0.0
        if samples:
            cur_spt = 1.0 / samples[-1][1]
        elif not self.generating:
            cur_spt = float(self.last_done.get("s_per_token") or 0.0)
        cur_rate = 1.0 / cur_spt if cur_spt > 0 else 0.0
        rate_txt = self._short(cur_rate) if cur_rate else "-"
        spt_txt = self._short_spt(cur_spt) if cur_spt else "-"

        # The inflow estimate needs a token rate. Decode supplies it once the
        # graph has samples; during prefill use the engine's prompt timing or
        # the live progress estimate instead of showing zero.
        flow_rate = cur_rate
        if flow_rate <= 0.0:
            if self.prefill_tokens > 0 and self.prefill_s > 0.0:
                flow_rate = self.prefill_tokens / self.prefill_s
            else:
                p = self.prefill_progress or {}
                elapsed = float(p.get("elapsed") or 0.0)
                done = int(p.get("tokens_done") or 0)
                if elapsed > 0.05 and done > 0:
                    flow_rate = done / elapsed

        def emit_memory() -> None:
            sysd = self.last.get("sys") or self.ready.get("sys") or {}
            cache = self.last.get("cache") or self.ready.get("cache") or {}
            trunk = self.last.get("trunk") or self.ready.get("trunk") or {}
            cfg = self.ready.get("config") or {}
            rss = float(sysd.get("rss_mb") or 0.0) / 1024.0
            hit = float(cache.get("hit_rate") or 0.0) * 100.0
            avail = (f"{self.mem_available_gb:.2f} GB"
                     if self.mem_available_gb > 0 else "n/a")
            avail_color = ("error" if 0 < self.mem_available_gb < 1.0
                           else "value")

            # Per token the CPU reads every dense trunk weight plus the routed
            # experts from RAM. That is a fixed byte count for a given model,
            # so multiplying by the current token rate estimates the flow. The
            # rate is decode tok/s, or the prompt's token rate during prefill.
            layers = int(cfg.get("layers") or 0)
            topk = int(cfg.get("topk") or 0)
            slot_mb = float(cache.get("slot_mb") or 0.0)
            expert_gb = layers * topk * slot_mb / 1024.0
            trunk_gb = float(trunk.get("resident_gb") or 0.0)
            inflow = (expert_gb + trunk_gb) * flow_rate

            emit((" MEMORY", "section"))
            emit((" RSS  ", "dim"), (f"{rss:.2f} GB", "value"),
                 ("  avail ", "dim"), (avail, avail_color))
            emit((" cache-hit ", "dim"), (f"{hit:5.1f}%", "value"))
            emit((" disk ", "dim"), (fmt_gbps(self.io_rate), "value"),
                 ("  inflow ", "dim"), (f"~{fmt_gbps(inflow)}", "value"))

        def emit_cpu() -> None:
            def percent(core: int) -> str:
                value = self.core_pct[core]
                return f"{value:3.0f}%" if value is not None else "  - "

            emit((" CPU", "section"))
            segments: list[tuple[str, str]] = []
            for core in range(4):
                label = f" c{core} " if core == 0 else f"  c{core} "
                segments.append((label, "dim"))
                segments.append((percent(core), "value"))
            emit(*segments)
            temp = (f"{self.cpu_temp_c:.0f}°C"
                    if self.cpu_temp_c is not None else "n/a")
            temp_color = ("error" if self.cpu_temp_c is not None
                          and self.cpu_temp_c >= 85 else "value")
            emit((" temp ", "dim"), (temp, temp_color))

        def emit_bottom() -> None:
            if out and out[-1]:
                emit()
            emit_memory()
            emit()
            emit_cpu()

        if in_decode:
            emit((" DECODE", "section"),
                 (f" {rate_txt} tok/s", "dim"),
                 (f" ({spt_txt} s/tok)", "dim"))

            if self.decode_spt:
                avg_spt = sum(self.decode_spt) / len(self.decode_spt)
                avg_rate = 1.0 / avg_spt if avg_spt > 0 else 0.0
                emit((" ⌀ avg  ", "dim"),
                     (f"{self._short(avg_rate)} tok/s", "system"),
                     (f" ({self._short_spt(avg_spt)} s/tok)", "value"))
            else:
                emit((" ⌀ avg  ", "dim"), ("-", "value"))

            if samples:
                graph_w = max(6, width - 8)
                visible = samples[-graph_w:]
                cols = [("empty", 0.0)] * (graph_w - len(visible)) + visible

                graph_h = max(3, min(10, height - 12
                                     - (3 if show_prefill else 0)))
                target = 5 if graph_h >= 6 else 2
                peak = max(rate for _, rate in visible)
                step = nice_step(peak, target)
                axis_max = math.ceil(peak / step - 1e-9) * step

                row_labels: dict[int, str] = {}
                for i in range(int(round(axis_max / step)) + 1):
                    value = i * step
                    row = graph_h - 1 - int(round(
                        (value / axis_max) * (graph_h - 1)))
                    row = int(clamp(row, 0, graph_h - 1))
                    row_labels.setdefault(row, axis_label(value, step))

                blocks = " ▁▂▃▄▅▆▇█"
                sub = 8

                for r in range(graph_h):
                    label = f"{row_labels.get(r, ''):>7}"
                    segments: list[tuple[str, str]] = []
                    cur_color: Optional[str] = None
                    cur_text = ""
                    for _, rate in cols:
                        frac = (clamp(rate / axis_max, 0.0, 1.0)
                                if axis_max > 0 else 0.0)
                        filled = (max(1, int(round(frac * graph_h * sub)))
                                  if rate > 1e-9 else 0)
                        n = filled - (graph_h - 1 - r) * sub
                        if n >= sub:
                            ch = "█"
                        elif n > 0:
                            ch = blocks[n]
                        else:
                            ch = " "
                        color = "decode" if ch != " " else "dim"
                        if color != cur_color:
                            if cur_text:
                                segments.append((cur_text, cur_color or "dim"))
                            cur_text = ""
                            cur_color = color
                        cur_text += ch
                    if cur_text:
                        segments.append((cur_text, cur_color or "dim"))
                    emit((label + "┤", "dim"), *segments)

                emit((" " * 7 + "└", "dim"), ("─" * graph_w, "dim"))
            else:
                emit(("  waiting for the first token...", "dim"))
        elif not in_prefill:
            if not self.generating:
                emit((" Waiting for a turn...", "dim"))
            elif self.phase == "resume":
                emit(("  closing previous turn...", "dim"))
            else:
                emit(("  prefill in progress...", "dim"))

        emit_bottom()
        return out

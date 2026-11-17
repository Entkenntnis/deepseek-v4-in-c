# SPDX-License-Identifier: Apache-2.0
"""Engine events -> model/UI state."""

from __future__ import annotations

import time

from .defaults import strip_special_tokens
from .markdown import RICH_AVAILABLE
from .messages import Message


class EventMixin:
    def handle_event(self, ev: dict) -> None:
        name = ev.get("event")
        # Layer events arrive far more often than the panel needs. The panel is
        # sampled at 1 Hz -- rates over a one-second window are more meaningful
        # than rates over a single layer, and a number that changes on every
        # layer is noise. Token text and layer progress below are NOT throttled.
        now = time.monotonic()
        force_stats = name in ("ready", "done")
        # Prefill emits far more progress events than the stats panel needs.
        # Treat them like any other event: MEMORY/CPU and the rate figures are
        # a 1 Hz instrument. The PREFILL bar is drawn separately on every
        # render (see prefill_lines), so the bar and counters still move live
        # without repainting the rest of the panel many times a second.
        if force_stats or now - self.last_stats_ingest >= self.stats_interval:
            self.ingest_stats(ev)
            self.last_stats_ingest = now
            if force_stats:
                self.stats_force = True
        if name == "ready":
            first = not self.ready
            self.ready = ev
            self.context_cap = (ev.get("context") or {}).get("cap",
                                                             self.context_cap)
            self.context_used = (ev.get("context") or {}).get("used", 0)
            if first:
                cfg = ev.get("config") or {}
                cache = ev.get("cache") or {}
                trunk = ev.get("trunk") or {}
                self.messages.append(Message(
                    "system",
                    f"loaded {cfg.get('layers', '?')} layers, "
                    f"{cfg.get('hidden', '?')}d, "
                    f"{cfg.get('experts', '?')} experts top{cfg.get('topk', '?')}; "
                    f"cache {cache.get('slots', '?')} slots "
                    f"({cache.get('resident_gb', 0):+.2f} GB), "
                    f"trunk {trunk.get('pinned', '?')} pinned. "
                    f"Context {self.context_cap} tokens."))
                if not RICH_AVAILABLE and not self.md_warned:
                    self.md_warned = True
                    self.add_system(
                        "Markdown rendering is off because rich is not installed. "
                        "Install it with: python3 -m pip install -r "
                        "tui/requirements.txt")
            if self.status in ("starting", "waiting"):
                self.status = "ready"
            self.dirty = True

        elif name == "start":
            self.generating = True
            self.status = "generating"
            self.generation = 0
            self.gen_max = int(ev.get("gen_max") or self.gen_limit)
            self.prompt_tokens = int(ev.get("prompt_tokens") or 0)
            self.context_cap = int(ev.get("context_cap") or self.context_cap)
            self.context_used = int(ev.get("pos") or 0) + self.prompt_tokens
            self.turn_start = time.monotonic()
            self.last_token_t = 0.0
            # decode_spt is cleared on the first decoded token, so the graph
            # only ever covers the current turn's decode.
            self.prefill_size = int(ev.get("prefill_tokens") or 0)
            self.prefill_progress = {}
            self.prefill_active = self.prefill_size > 1
            self.prefill_s = 0.0
            self.prefill_tokens = 0
            self.prefill_elapsed = 0.0
            self.prefill_rate = 0.0
            self.prefill_sample_ts = 0.0
            self.ttft = 0.0
            self.current_layer = 0
            self.phase = "prompt"
            self.decoder.reset()
            if not self.current_msg or self.current_msg.role != "assistant":
                self.current_msg = Message("assistant", "")
                self.messages.append(self.current_msg)
            if self.cancel_on_start:
                self.cancel_on_start = False
                self.cancel_pending = False
                self.cancel()
            else:
                self.cancel_pending = False
            self.stats_force = True
            self.dirty = True

        elif name == "layer":
            self.current_layer = int(ev.get("layer") or 0) + 1
            self.phase = ev.get("phase", "")
            self.dirty = True

        elif name == "prefill":
            self.prefill_progress = ev
            self.prefill_active = True
            total = int(ev.get("tokens_total") or 0)
            if total:
                self.prefill_size = total
            self.current_layer = int(ev.get("layer") or self.current_layer)
            self.phase = "prompt"
            self.dirty = True

        elif name == "token":
            raw = bytes.fromhex(ev.get("hex") or "")
            text = strip_special_tokens(self.decoder.decode(raw, final=False))
            if text and self.current_msg:
                self.current_msg.text += text
            index = int(ev.get("index") or 0)
            self.generation = index + 1
            self.context_used += 1
            t = float(ev.get("t") or 0.0)
            ttft = float(ev.get("ttft") or self.ttft)
            if index == 0:
                # The first decoded token switches the panel from PREFILL to
                # DECODE; start the decode average fresh.
                self.phase = "gen"
                self.decode_spt.clear()
                ps = float(ev.get("prefill_s") or 0.0)
                pt = int(ev.get("prefill_tokens") or self.prompt_tokens or 0)
                if ps > 0.0 and pt > 0:
                    self.prefill_s = ps
                    self.prefill_tokens = pt
                self.prefill_active = False
            elif self.last_token_t > 0:
                dt = t - self.last_token_t
                if dt > 1e-6:
                    self.decode_spt.append(dt)
            self.last_token_t = t
            self.ttft = ttft
            self.stats_force = True
            self.dirty = True

        elif name == "done":
            stop = ev.get("stop")
            generated = int(ev.get("generated") or 0)
            # No token was emitted, so a cancel rolled the engine back to the
            # previous turn. The message stays in the transcript as a record,
            # but it is not in the model context. A reset event (the
            # no-snapshot fallback) has already taken the whole transcript.
            dropped = (stop == "cancel" and generated == 0
                       and self.current_msg is not None)
            self.generating = False
            self.cancel_pending = False
            self.cancel_on_start = False
            self.prefill_active = False
            self.status = "ready" if stop != "error" else "failed"
            self.last_done = ev
            self.context_used = int(ev.get("context_used") or self.context_used)
            if self.current_msg is not None:
                # The engine's done.tps covers the whole turn, prefill included.
                # The transcript wants decode only, so use the same inter-token
                # samples as the sidebar. A turn with no tokens has no decode.
                self.current_msg.meta = {
                    "generated": generated,
                    "seconds": ev.get("seconds"),
                    "tps": self.decode_avg_tps() if generated > 0 else 0.0,
                    "ttft": ev.get("ttft"),
                    "stop": stop,
                }
                if not self.current_msg.text.strip():
                    self.current_msg.text = ("(cancelled)" if dropped
                                             else "(no tokens)")
            if dropped:
                self.add_system("turn cancelled; context unchanged")
            if self.tool_turn_done(stop):
                self.dirty = True
                return
            self.current_msg = None
            self.current_user_msg = None
            self.dirty = True

        elif name == "tool_done":
            self.tool_event(ev)

        elif name == "reset":
            self.generating = False
            self.cancel_pending = False
            self.cancel_on_start = False
            self.status = "ready"
            self.context_used = 0
            self.current_msg = None
            self.current_user_msg = None
            # An engine-side reset invalidates any tool round in flight: the
            # conversation the model saw no longer exists.
            self.chain_id += 1
            self.tool_running = False
            self.tool_cancelled = False
            self.pending_tool = None
            self.py.reset()
            self.decode_spt.clear()
            self.prefill_active = False
            self.prefill_progress = {}
            self.prefill_s = 0.0
            self.prefill_tokens = 0
            self.prefill_elapsed = 0.0
            self.prefill_rate = 0.0
            self.prefill_sample_ts = 0.0
            if ev.get("reason") == "cancel":
                # The no-snapshot fallback: a partially written block cannot
                # be resumed, so the engine dropped the whole context. The
                # transcript no longer matches it; start a new one.
                self.messages.clear()
                self.add_system("prefill cancelled; context reset")
            self.stats_force = True
            self.dirty = True

        elif name == "error":
            msg = str(ev.get("message") or "unknown engine error")
            # If an assistant bubble was opened for a turn that never started,
            # do not leave an empty one above the error.
            if self.current_msg is not None and not self.current_msg.text:
                try:
                    self.messages.remove(self.current_msg)
                except ValueError:
                    pass
            self.current_msg = None
            self.current_user_msg = None
            self.generating = False
            self.cancel_pending = False
            self.cancel_on_start = False
            self.prefill_active = False
            if self.status not in ("starting",):
                self.status = "ready" if self.backend_alive else "failed"
            self.messages.append(Message("error", msg))
            self.dirty = True

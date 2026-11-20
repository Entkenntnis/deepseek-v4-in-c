# SPDX-License-Identifier: Apache-2.0
"""The tool-call loop: one user turn may span several model/tool rounds.

The engine's KV cache is the conversation, so a round is just another
`generate` whose prompt is a <tool_result> block. This mixin owns that state
machine and nothing else: parsing lives in dsml.py, Python execution in
pytool.py, and web search in websearch.py.
"""

from __future__ import annotations

import time

from .defaults import (SEARCH_DEFAULT_RESULTS, SEARCH_MAX_RESULTS,
                       TOOL_MAX_ROUNDS)
from .dsml import (DSML, TOOL_NAME, WEB_SEARCH_NAME, code_fence,
                   format_result, parse_tool_calls, render_tool_text,
                   render_web_text, system_prompt)
from .messages import Message

TOOL_NAMES = (TOOL_NAME, WEB_SEARCH_NAME)


class ToolMixin:
    # -- starting a turn ---------------------------------------------------

    def tool_begin(self, text: str) -> None:
        """Open a turn (user message + first generate) and reset chain state."""
        self.chain_id += 1
        self.chain_round = 0
        self.tool_cancelled = False
        self.tool_running = False
        self.pending_tool = None
        self.pending_tool_name = ""
        self.pending_code = ""
        self.pending_query = ""
        self.current_user_msg = Message("user", text)
        self.messages.append(self.current_user_msg)
        # The turn clock covers the whole exchange: model rounds, tool runs and
        # the gaps between them. The conversation clock starts at the first
        # user message and survives until the transcript is reset.
        self.turn_start = time.monotonic()
        self.round_prefill_s = 0.0
        if not self.conv_started:
            self.conv_started = time.time()
        self.stats_force = True
        self._open_assistant()
        self.scroll = 0
        self.generating = True
        self.status = "waiting"
        self._ensure_system()
        if not self.backend.send({"cmd": "generate", "prompt": text,
                                  "gen": self.gen_limit}):
            self.generating = False
            self.status = "failed"
            self.end_turn_clock()
            self.add_system("could not write to the engine")

    def _ensure_system(self) -> None:
        """The engine only applies a system prompt while its context is empty."""
        if not self.tool_enabled or not self.system_armed:
            return
        if self.backend.send({"cmd": "system", "text": system_prompt()}):
            self.system_armed = False

    def _open_assistant(self) -> None:
        self.current_msg = Message("assistant", "")
        self.messages.append(self.current_msg)
        self.decoder.reset()

    def _settle_content(self, msg: Message, content: str) -> None:
        """Keep the prose before a tool block; drop the bubble if there is none."""
        if content.strip():
            msg.text = content
            return
        try:
            self.messages.remove(msg)
        except ValueError:
            pass
        if self.current_msg is msg:
            self.current_msg = None

    # -- end of one model round -------------------------------------------

    def tool_turn_done(self, stop: str) -> bool:
        """Called from the `done` event. True means the chain continues."""
        if stop in ("cancel", "error"):
            return False
        msg = self.current_msg
        if msg is None:
            return False

        raw = msg.text
        content, calls = parse_tool_calls(raw)
        if not self.tool_enabled:
            # The tools preamble may still be in the KV cache after startup.
            if calls or f"<{DSML}" in raw or "<|DSML|" in raw:
                self._settle_content(msg, content)
                self.add_system("tools are off; the call was ignored "
                                "(/tools to allow them)")
            return False
        if not calls:
            if raw != content:
                msg.text = content
            if f"<{DSML}" in raw or "<|DSML|" in raw:
                self.add_system(
                    "the tool call was cut off (token limit); ask me to retry")
            return False

        self._settle_content(msg, content)

        call = next((c for c in calls if c.get("name") in TOOL_NAMES), None)
        if call is None:
            names = ", ".join(str(c.get("name")) for c in calls)
            self.add_system(f"unknown tool call: {names}")
            return False
        args = call.get("arguments")
        if not isinstance(args, dict):
            args = {}
        name = str(call.get("name"))

        if name == TOOL_NAME:
            code = args.get("code", "")
            if not isinstance(code, str) or not code.strip():
                self.add_system("run_python call had no code; ignored")
                return False
            self.pending_code = code
            pending_text = code_fence(code) + "\n\n_running…_"
        else:
            query = args.get("query", "")
            if not isinstance(query, str) or not query.strip():
                self.add_system("web_search call had no query; ignored")
                return False
            try:
                count = int(args.get("count", SEARCH_DEFAULT_RESULTS))
            except (TypeError, ValueError):
                count = SEARCH_DEFAULT_RESULTS
            self.pending_query = query.strip()
            self.pending_count = max(1, min(SEARCH_MAX_RESULTS, count))
            pending_text = (f"**Search** `{self.pending_query}`\n\n"
                            "_running…_")

        if len(calls) > 1:
            self.add_system(f"{len(calls) - 1} extra tool call(s) skipped")

        self.chain_round += 1
        if self.chain_round > TOOL_MAX_ROUNDS:
            self.add_system(f"stopped after {TOOL_MAX_ROUNDS} tool rounds")
            return False
        if self.context_used >= self.context_cap - 512:
            self.add_system("stopped: context is nearly full; /reset to continue")
            return False

        self.pending_tool_name = name
        self.pending_tool = Message("tool", pending_text)
        self.messages.append(self.pending_tool)
        self.tool_running = True
        # Keep the turn "live" while the tool runs: the input line refuses new
        # turns, the header repaints, and Ctrl-C routes to tool_abort().
        self.generating = True
        self.status = "generating"
        chain = self.chain_id

        def done(result: dict) -> None:
            self.sink.event({"event": "tool_done", "chain": chain,
                             "result": result})

        if name == TOOL_NAME:
            started = self.py.run_async(self.pending_code, done)
            busy = "interpreter is busy; try again"
        else:
            started = self.web.run_async(self.pending_query,
                                         self.pending_count, done)
            busy = "web search is busy; try again"
        if not started:
            self.tool_running = False
            self.pending_tool = None
            self.add_system(busy)
            return False
        self.dirty = True
        return True

    # -- tool result arrives ----------------------------------------------

    def tool_event(self, ev: dict) -> None:
        if ev.get("chain") != self.chain_id or self.tool_cancelled:
            return
        self.tool_running = False
        result = ev.get("result") or {}
        if self.pending_tool is not None:
            if self.pending_tool_name == WEB_SEARCH_NAME:
                self.pending_tool.text = render_web_text(
                    self.pending_query, result)
            else:
                self.pending_tool.text = render_tool_text(
                    self.pending_code, result)
        self.pending_tool = None
        self.pending_tool_name = ""
        self.dirty = True

        self._open_assistant()
        self.generating = True
        self.status = "waiting"
        if not self.backend.send({"cmd": "generate",
                                  "prompt": format_result([result]),
                                  "gen": self.gen_limit}):
            self.generating = False
            self.status = "failed"
            self.end_turn_clock()
            self.add_system("could not write to the engine")

    # -- cancellation ------------------------------------------------------

    def tool_abort(self, note: str = "tool cancelled; chain stopped") -> None:
        """Kill a running tool and invalidate its late completion."""
        if not self.tool_running and not self.tool_cancelled:
            return
        self.tool_cancelled = True
        self.tool_running = False
        self.py.cancel()
        self.web.cancel()
        if self.pending_tool is not None:
            if self.pending_tool_name == WEB_SEARCH_NAME:
                text = (f"**Search** `{self.pending_query}`\n\n"
                        "_cancelled_")
            else:
                text = code_fence(self.pending_code) + "\n\n_cancelled_"
            self.pending_tool.text = text
        self.pending_tool = None
        self.pending_tool_name = ""
        self.chain_id += 1
        self.generating = False
        self.end_turn_clock()
        self.current_msg = None
        self.current_user_msg = None
        self.status = "ready" if self.backend_alive else "failed"
        self.add_system(note)

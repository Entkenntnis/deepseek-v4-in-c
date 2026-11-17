# SPDX-License-Identifier: Apache-2.0
"""The tool-call loop: one user turn may span several model/tool rounds.

The engine's KV cache is the conversation, so a round is just another
`generate` whose prompt is a <tool_result> block. This mixin owns that state
machine and nothing else: parsing lives in dsml.py, execution in pytool.py.
"""

from __future__ import annotations

from .defaults import TOOL_MAX_ROUNDS
from .dsml import (DSML, TOOL_NAME, code_fence, format_result,
                   parse_tool_calls, render_tool_text, system_prompt)
from .messages import Message


class ToolMixin:
    # -- starting a turn ---------------------------------------------------

    def tool_begin(self, text: str) -> None:
        """Open a turn (user message + first generate) and reset chain state."""
        self.chain_id += 1
        self.chain_round = 0
        self.tool_cancelled = False
        self.tool_running = False
        self.pending_tool = None
        self.pending_code = ""
        self.current_user_msg = Message("user", text)
        self.messages.append(self.current_user_msg)
        self._open_assistant()
        self.scroll = 0
        self.generating = True
        self.status = "waiting"
        self._ensure_system()
        if not self.backend.send({"cmd": "generate", "prompt": text,
                                  "gen": self.gen_limit}):
            self.generating = False
            self.status = "failed"
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
        content, _reasoning, calls = parse_tool_calls(raw)
        if not self.tool_enabled:
            # The tools preamble may still be in the KV cache after startup.
            if calls or f"<{DSML}" in raw or "<|DSML|" in raw:
                self._settle_content(msg, content)
                self.add_system("python is off; the call was ignored "
                                "(/py to allow it)")
            return False
        if not calls:
            if raw != content:
                msg.text = content
            if f"<{DSML}" in raw or "<|DSML|" in raw:
                self.add_system(
                    "the tool call was cut off (token limit); ask me to retry")
            return False

        self._settle_content(msg, content)

        call = next((c for c in calls if c.get("name") == TOOL_NAME), None)
        if call is None:
            names = ", ".join(str(c.get("name")) for c in calls)
            self.add_system(f"unknown tool call: {names}")
            return False
        code = call.get("arguments", {}).get("code", "")
        if not isinstance(code, str) or not code.strip():
            self.add_system("tool call had no code; ignored")
            return False
        if len(calls) > 1:
            self.add_system(f"{len(calls) - 1} extra tool call(s) skipped")

        self.chain_round += 1
        if self.chain_round > TOOL_MAX_ROUNDS:
            self.add_system(f"stopped after {TOOL_MAX_ROUNDS} tool rounds")
            return False
        if self.context_used >= self.context_cap - 512:
            self.add_system("stopped: context is nearly full; /reset to continue")
            return False

        self.pending_code = code
        self.pending_tool = Message("tool", code_fence(code) + "\n\n_running…_")
        self.messages.append(self.pending_tool)
        self.tool_running = True
        # Keep the turn "live" while the child runs: the input line refuses new
        # turns, the header repaints, and Ctrl-C routes to tool_abort().
        self.generating = True
        self.status = "generating"
        chain = self.chain_id

        def done(result: dict) -> None:
            self.sink.event({"event": "tool_done", "chain": chain,
                             "result": result})

        if not self.py.run_async(code, done):
            self.tool_running = False
            self.pending_tool = None
            self.add_system("interpreter is busy; try again")
            return False
        self.dirty = True
        return True

    # -- interpreter result arrives ---------------------------------------

    def tool_event(self, ev: dict) -> None:
        if ev.get("chain") != self.chain_id or self.tool_cancelled:
            return
        self.tool_running = False
        result = ev.get("result") or {}
        if self.pending_tool is not None:
            self.pending_tool.text = render_tool_text(self.pending_code, result)
        self.pending_tool = None
        self.dirty = True

        self._open_assistant()
        self.generating = True
        self.status = "waiting"
        if not self.backend.send({"cmd": "generate",
                                  "prompt": format_result([result]),
                                  "gen": self.gen_limit}):
            self.generating = False
            self.status = "failed"
            self.add_system("could not write to the engine")

    # -- cancellation ------------------------------------------------------

    def tool_abort(self, note: str = "python cancelled; chain stopped") -> None:
        """Kill a running script and invalidate its late completion."""
        if not self.tool_running and not self.tool_cancelled:
            return
        self.tool_cancelled = True
        self.tool_running = False
        self.py.cancel()
        if self.pending_tool is not None:
            self.pending_tool.text = (
                code_fence(self.pending_code) + "\n\n_cancelled_")
        self.pending_tool = None
        self.chain_id += 1
        self.generating = False
        self.current_msg = None
        self.current_user_msg = None
        self.status = "ready" if self.backend_alive else "failed"
        self.add_system(note)

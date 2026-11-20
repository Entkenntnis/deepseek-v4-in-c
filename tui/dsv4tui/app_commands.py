# SPDX-License-Identifier: Apache-2.0
"""Submitting a turn, cancelling one, and the slash commands."""

from __future__ import annotations

from .defaults import DEFAULT_ENV, DEFAULT_KV
from .messages import Message


class CommandMixin:
    def add_system(self, text: str) -> None:
        self.messages.append(Message("system", text))
        self.dirty = True

    def _tools_note(self, text: str) -> str:
        """Append the one-time hint when web search has no key yet."""
        if not self.web.key_configured():
            return (f"{text}; web search needs DEEPSEEK_API_KEY in "
                    f"{DEFAULT_ENV}")
        return text

    def submit(self) -> None:
        text = self.input.strip()
        if not text:
            return
        self.input = ""
        self.input_cursor = 0

        if text.startswith("/"):
            self.command(text)
            return

        if not self.backend_alive:
            self.add_system("engine is not running; restart the TUI")
            return
        if self.kv_pending:
            self.add_system("cached tools preamble is still loading; try again")
            return
        if self.status in ("starting", "waiting"):
            self.add_system("model is still loading; wait for the ready event")
            return
        if self.generating:
            self.add_system("a turn is already generating")
            return

        self.tool_begin(text)
        self.dirty = True

    def cancel(self) -> None:
        """Stop at the next safe point.

        Decode keeps the turn in the model context. A prefill stop rolls the
        context back to the previous turn; the message stays in the transcript
        but is not sent to the model.
        """
        if not self.generating or self.cancel_pending:
            return
        if self.tool_running:
            self.tool_abort()
            return
        if self.status != "generating":
            # The generate command is queued but the engine has not emitted
            # `start` yet. Send the actual cancel from that event instead.
            self.cancel_on_start = True
            self.cancel_pending = True
            self.dirty = True
            return
        if self.backend and self.backend.cancel():
            self.cancel_pending = True
        else:
            self.add_system("could not request cancellation")
        self.dirty = True

    def command(self, text: str) -> None:
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            self.running = False
        elif cmd == "/help":
            self.add_system(
                "commands:\n"
                "  /help                 this list\n"
                "  /reset                clear model context\n"
                "  /gen N                max tokens per turn\n"
                "  /tools                allow run_python and web_search; loads\n"
                "                        the cached tools preamble when empty\n"
                "  /exit                 quit")
        elif cmd == "/reset":
            if self.tool_running:
                self.tool_abort("tool cancelled for /reset")
            elif self.generating:
                self.add_system("finish the current turn before resetting")
                return
            if self.backend_alive:
                self.backend.send({"cmd": "reset"})
            self.py.reset()
            self.web.reset()
            self.chain_id += 1
            self.tool_cancelled = False
            self.system_armed = True
            self.kv_loaded = False
            self.kv_pending = False
            self.messages.clear()
            self.current_msg = None
            self.current_user_msg = None
            self.generating = False
            self.status = "ready"
            self.context_used = 0
            self.decode_spt.clear()
            self.stats_force = True
            self.add_system("context reset; cache stays warm")
        elif cmd == "/tools":
            if arg:
                state = "on" if self.tool_enabled else "off"
                self.add_system(f"tools are {state}; usage: /tools")
            elif self.kv_pending:
                self.add_system("cached tools preamble is still loading")
            elif self.generating and not self.tool_running:
                self.add_system(
                    "finish the current turn before changing tools")
                return
            elif self.tool_enabled and self.kv_loaded:
                self.add_system(self._tools_note(
                    "tools are already on; cached preamble loaded"))
            elif self.context_used == 0 and \
                    (not self.tool_enabled or not self.kv_loaded):
                # Empty context and no restored preamble yet: try the cache.
                # A miss is harmless, the first turn then falls back to the
                # normal `system` command and a full prefill.
                self.tool_enabled = True
                if DEFAULT_KV.exists() and self.backend_alive and self.backend.send(
                        {"cmd": "kv_load", "path": str(DEFAULT_KV)}):
                    self.kv_pending = True
                    self.add_system(self._tools_note(
                        f"tools on; loading cached preamble from "
                        f"{DEFAULT_KV}…"))
                else:
                    self.system_armed = True
                    if DEFAULT_KV.exists():
                        self.add_system(self._tools_note(
                            "tools on; could not request the cached preamble"))
                    else:
                        self.add_system(self._tools_note(
                            "tools on; no cached preamble yet (run `make kv`); "
                            "the model may call run_python and web_search"))
            elif self.tool_enabled:
                self.add_system(self._tools_note("tools are already on"))
            else:
                self.tool_enabled = True
                if self.context_used == 0:
                    self.system_armed = True
                    self.add_system(self._tools_note(
                        "tools on; the model may call run_python and web_search"))
                else:
                    self.add_system(self._tools_note(
                        "tools on, but the tools preamble needs an empty "
                        "context; /reset before the model can see it"))
        elif cmd == "/gen":
            try:
                n = int(arg)
                if n < 1:
                    raise ValueError
                self.gen_limit = n
                self.add_system(f"max {n} tokens per turn")
            except ValueError:
                self.add_system("usage: /gen N   (N >= 1)")
        else:
            self.add_system(f"unknown command {cmd}; /help for the list")
        self.dirty = True

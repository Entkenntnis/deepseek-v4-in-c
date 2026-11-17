# SPDX-License-Identifier: Apache-2.0
"""Submitting a turn, cancelling one, and the slash commands."""

from __future__ import annotations

from .messages import Message


class CommandMixin:
    def add_system(self, text: str) -> None:
        self.messages.append(Message("system", text))
        self.dirty = True

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
        if self.status in ("starting", "waiting"):
            self.add_system("model is still loading; wait for the ready event")
            return
        if self.generating:
            self.add_system("a turn is already generating")
            return

        self.messages.append(Message("user", text))
        self.current_msg = Message("assistant", "")
        self.messages.append(self.current_msg)
        self.decoder.reset()
        self.scroll = 0
        self.generating = True
        self.status = "waiting"
        sent = self.backend.send({
            "cmd": "generate",
            "prompt": text,
            "gen": self.gen_limit,
        })
        if not sent:
            self.generating = False
            self.status = "failed"
            self.add_system("could not write to the engine")
        self.dirty = True

    def cancel(self) -> None:
        """Stop at the next token; the turn stays in the model context."""
        if not self.generating or self.cancel_pending:
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
                "  /reset                clear model context (keeps the disk cache warm)\n"
                "  /gen N                max tokens per turn\n"
                "  /exit                 quit")
        elif cmd == "/reset":
            if self.generating:
                self.add_system("finish the current turn before resetting")
                return
            if self.backend_alive:
                self.backend.send({"cmd": "reset"})
            self.messages.clear()
            self.current_msg = None
            self.generating = False
            self.status = "ready"
            self.context_used = 0
            self.decode_spt.clear()
            self.stats_force = True
            self.add_system("context reset; expert cache stays warm")
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

# SPDX-License-Identifier: Apache-2.0
"""Submitting a turn, cancelling one, and the slash commands."""

from __future__ import annotations

from .markdown import RICH_AVAILABLE
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
        self.current_msg = Message("assistant", "", think=self.think)
        self.messages.append(self.current_msg)
        self.decoder.reset()
        self.scroll = 0
        self.generating = True
        self.status = "waiting"
        sent = self.backend.send({
            "cmd": "generate",
            "prompt": text,
            "gen": self.gen_limit,
            "think": self.think,
            "layers": self.show_layers,
        })
        if not sent:
            self.generating = False
            self.status = "failed"
            self.add_system("could not write to the engine")
        self.dirty = True

    def cancel(self) -> None:
        """Stop at the next token; the turn stays in the model context."""
        if self.status != "generating" or self.cancel_pending:
            return
        if self.backend and self.backend.cancel():
            self.cancel_pending = True
            self.status_detail = "cancelling"
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
                "  /clear                clear the chat display only\n"
                "  /system TEXT          system prompt for the NEXT reset\n"
                "  /think [on|off]       toggle thinking mode for the next turn\n"
                "  /gen N                max tokens per turn\n"
                "  /layers [on|off]      stream per-layer progress events\n"
                "  /markdown [on|off]    Markdown rendering (rich, optional)\n"
                "  /stats                ask the engine for a fresh snapshot\n"
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
        elif cmd == "/clear":
            self.messages.clear()
            self.add_system("display cleared")
        elif cmd == "/system":
            if not arg:
                self.add_system(f"system prompt: {self.system_prompt or '(empty)'}")
            else:
                self.system_prompt = arg
                if self.context_used > 0:
                    self.add_system("system prompt stored; /reset for it to take effect")
                if self.backend_alive:
                    self.backend.send({"cmd": "system", "text": arg})
                else:
                    self.add_system("engine is not running")
        elif cmd == "/think":
            if arg.lower() in ("on", "1", "true", "yes"):
                self.think = True
            elif arg.lower() in ("off", "0", "false", "no"):
                self.think = False
            else:
                self.think = not self.think
            self.add_system(f"thinking mode {'on' if self.think else 'off'} "
                            f"for the next turn")
        elif cmd == "/gen":
            try:
                n = int(arg)
                if n < 1:
                    raise ValueError
                self.gen_limit = n
                self.add_system(f"max {n} tokens per turn")
            except ValueError:
                self.add_system("usage: /gen N   (N >= 1)")
        elif cmd == "/layers":
            if arg.lower() in ("on", "1", "true", "yes"):
                self.show_layers = True
            elif arg.lower() in ("off", "0", "false", "no"):
                self.show_layers = False
            else:
                self.show_layers = not self.show_layers
            self.add_system(f"layer events {'on' if self.show_layers else 'off'}")
        elif cmd == "/markdown":
            if arg.lower() in ("on", "1", "true", "yes"):
                self.markdown_enabled = True
            elif arg.lower() in ("off", "0", "false", "no"):
                self.markdown_enabled = False
            else:
                self.markdown_enabled = not self.markdown_enabled
            if self.markdown_enabled and not RICH_AVAILABLE:
                self.add_system("Markdown needs rich: python3 -m pip install -r "
                                "tui/requirements.txt")
            else:
                self.add_system(f"Markdown rendering "
                                f"{'on' if self.markdown_enabled else 'off'}")
        elif cmd == "/stats":
            if self.backend_alive:
                self.backend.send({"cmd": "stats"})
            self.add_system("requested a fresh snapshot")
        else:
            self.add_system(f"unknown command {cmd}; /help for the list")
        self.dirty = True

# SPDX-License-Identifier: Apache-2.0
"""The composed application class.

Each mixin owns one slice of the old monolith; App inherits them in dependency
order and adds nothing of its own.
"""

from __future__ import annotations

from .agent import ToolMixin
from .app_chat import ChatMixin
from .app_commands import CommandMixin
from .app_core import AppBase
from .app_cpu import CpuMixin
from .app_curses import CursesMixin
from .app_draw import DrawMixin
from .app_events import EventMixin
from .app_graph import GraphMixin
from .app_ingest import IngestMixin
from .app_input import InputMixin
from .app_loop import LoopMixin
from .app_stats import StatsMixin
from .app_view import ViewMixin


class App(EventMixin, IngestMixin, CpuMixin, ToolMixin, CommandMixin,
          InputMixin, ViewMixin, ChatMixin, StatsMixin, GraphMixin, DrawMixin,
          CursesMixin, LoopMixin, AppBase):
    """deepseek-v4-in-c TUI."""

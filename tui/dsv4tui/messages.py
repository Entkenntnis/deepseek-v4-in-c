# SPDX-License-Identifier: Apache-2.0
"""One transcript entry. Shared by the event, command and render mixins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .markdown import Span

@dataclass
class Message:
    role: str                     # user | assistant | system | error
    text: str = ""
    meta: dict = field(default_factory=dict)
    md_cache_key: tuple = ()
    md_cache_lines: list[list[Span]] = field(default_factory=list)

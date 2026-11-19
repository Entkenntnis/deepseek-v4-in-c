# SPDX-License-Identifier: Apache-2.0
"""The line protocol spoken by bin/dsv4-serve.

Wire shape is deliberately trivial: one event or command per line, the first
whitespace-separated token is the name, and the rest are ``key=value`` fields.
String fields carry raw bytes as ``key=hex:<hex>`` so neither side needs to
quote, escape, or parse Unicode.

``parse_event`` also reconstructs the small nested dictionaries the TUI uses
(``sys``, ``cache``, ``trunk``, ``config``, ``context``) from their flattened
wire prefixes, so event handlers keep their existing shape.
"""

from __future__ import annotations

from typing import Any

_NESTED = ("config", "sys", "cache", "trunk")


def hex_field(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "hex:" + raw.hex()


def encode_command(cmd: dict[str, Any]) -> str:
    """Turn a command dict into one wire line, without the trailing newline."""
    name = str(cmd.get("cmd") or "")
    if not name:
        raise ValueError("command has no cmd")

    fields: list[str] = []

    if name == "generate":
        tokens = cmd.get("tokens")
        if tokens is None:
            raise ValueError("generate needs a token list")
        fields.append("tokens=" + ",".join(str(int(i)) for i in tokens))
        fields.append(f"gen={int(cmd.get('gen', 0))}")
        fields.append(f"eos={int(cmd.get('eos', -1))}")
        fields.append(f"think={1 if cmd.get('think') else 0}")
        fields.append(f"layers={1 if cmd.get('layers') else 0}")
    elif name in ("kv_build", "kv_load"):
        path = cmd.get("path")
        if path is not None:
            fields.append("path=" + hex_field(str(path)))
        tokens = cmd.get("tokens")
        if tokens is not None:
            fields.append("tokens=" + ",".join(str(int(i)) for i in tokens))
    elif name in ("reset", "stats", "ping", "quit"):
        pass
    else:
        raise ValueError(f"unknown command {name!r}")

    return " ".join((name, *fields))


def _number(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_event(line: str) -> dict[str, Any] | None:
    """Parse one server event line into the dict shape the TUI consumes."""
    parts = line.strip().split()
    if not parts:
        return None
    ev: dict[str, Any] = {"event": parts[0]}

    for field in parts[1:]:
        key, sep, value = field.partition("=")
        if not sep:
            continue
        if value.startswith("hex:"):
            ev[key] = bytes.fromhex(value[4:]).decode("utf-8", "replace")
        else:
            ev[key] = _number(value)

    prefixes = list(_NESTED)
    # Only `ready` carries the context as a pair. Elsewhere context_used /
    # context_cap are ordinary top-level fields (start, done, kv_loaded).
    if ev["event"] in ("ready", "stats") and "context" not in prefixes:
        prefixes.append("context")
    for prefix in prefixes:
        head = prefix + "_"
        bucket = {k[len(head):]: ev.pop(k) for k in list(ev)
                  if k.startswith(head)}
        if bucket:
            ev[prefix] = bucket
    return ev

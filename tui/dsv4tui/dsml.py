# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4's DSML tool-call format and the one tool the TUI offers.

The format is the checkpoint's own (see encoding/README.md next to the
weights), not an invention of this TUI: the model was trained to emit
<｜DSML｜>tool_calls> blocks and to read <tool_result> blocks back. The engine
already wraps every prompt in <｜User｜>...<｜Assistant｜>, so a follow-up round
is just a normal generate whose prompt is the tool_result block.
"""

from __future__ import annotations

import json
import re

from .defaults import BOS

DSML = "｜DSML｜"
TOOL_NAME = "run_python"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": ("Run Python 3 in a fresh interpreter for every call. "
                    "Variables and imports never carry over between calls. "
                    "Returns stdout/stderr; input() is disabled; output is "
                    "truncated beyond 750 chars"),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
        },
        "required": ["code"],
    },
}

# Compact rendering of the checkpoint's tools preamble. The tag syntax is the
# part that must match training exactly; the prose around it is instructions
# the model already knows, so it costs context for little gain.
_TOOLS_TEMPLATE = """## Tools

Call a tool by writing a <{dsml}tool_calls> block:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
</{dsml}invoke>
</{dsml}tool_calls>

String parameters are raw text with string="true"; other types are JSON with string="false".

### Available Tool Schemas

{schemas}

You MUST strictly follow the above tool name and parameter schema.
"""


def system_prompt() -> str:
    """The tools preamble, sent once while the engine context is empty."""
    schema = json.dumps(TOOL_SCHEMA, ensure_ascii=False,
                        separators=(",", ":"))
    return _TOOLS_TEMPLATE.format(dsml=DSML, schemas=schema)


def py_prefix_text() -> str:
    """Exactly the prefix the engine puts in front of the first user turn.

    `dsv4_engine_generate()` builds BOS + system prompt + newline when the
    context is empty. The precomputed cache must contain those same tokens and
    nothing else; the first user turn is appended after it.
    """
    return BOS + system_prompt() + "\n"


_BLOCK_RE = re.compile(
    re.escape(f"<{DSML}tool_calls>") + r"(.*?)"
    + re.escape(f"</{DSML}tool_calls>"), re.S)
_INVOKE_RE = re.compile(
    re.escape(f"<{DSML}invoke") + r"""\s+name=["']([^"']*)["']\s*>(.*?)"""
    + re.escape(f"</{DSML}invoke>"), re.S)
_PARAM_RE = re.compile(
    re.escape(f"<{DSML}parameter") + r"""\s+name=["']([^"']*)["']"""
    + r"""\s+string=["'](true|false)["']\s*>(.*?)"""
    + re.escape(f"</{DSML}parameter>"), re.S)
_THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)


def parse_tool_calls(text: str) -> tuple[str, str, list[dict]]:
    """Split an assistant turn into (content, reasoning, tool calls).

    `content` is the prose before the DSML block, with the reasoning and the
    markup removed. `reasoning` is the <think> body when one is present. Each
    call is {"name": str, "arguments": dict}. Parsing is deliberately lenient:
    greedy decode occasionally drops a newline or uses ASCII bars.
    """
    text = text.replace("|DSML|", DSML)
    reasoning = ""
    m = _THINK_RE.search(text)
    if m:
        reasoning = m.group(1).strip()
        # The reasoning block (closed or truncated) sits at the start.
        text = text[m.end():] if m.group(0).endswith("</think>") else ""

    block = _BLOCK_RE.search(text)
    content = text[:block.start()] if block else text
    content = content.replace(DSML, "").strip()

    calls: list[dict] = []
    if block:
        for im in _INVOKE_RE.finditer(block.group(1)):
            args: dict = {}
            for pm in _PARAM_RE.finditer(im.group(2)):
                key, is_string, value = pm.groups()
                value = value.strip("\n")
                if is_string == "false":
                    try:
                        value = json.loads(value.strip())
                    except (ValueError, TypeError):
                        pass
                args[key] = value
            calls.append({"name": im.group(1), "arguments": args})
    return content, reasoning, calls


def format_result(results: list[dict]) -> str:
    """Wrap execution results the way the model expects to read them back."""
    parts = []
    for result in results:
        payload = json.dumps(result, ensure_ascii=False)
        # The closing tag is the framing; a program that prints it would end
        # the block early, so escape it before it reaches the model.
        payload = payload.replace("</tool_result>", r"<\/tool_result>")
        parts.append(f"<tool_result>{payload}</tool_result>")
    return "\n\n".join(parts)


def code_fence(code: str) -> str:
    """A Markdown fence that survives code containing backticks."""
    fence = "~~~~" if "```" in code else "```"
    return f"{fence}python\n{code.rstrip()}\n{fence}"


def render_tool_text(code: str, result: dict) -> str:
    """The transcript bubble: what ran, what came back, and how long it took."""
    out = str(result.get("stdout") or "")
    err = str(result.get("stderr") or "")
    body = out + (("\n" + err) if err else "")
    if not body.strip():
        body = "(no output)"
    fence = "~~~~" if "```" in body else "```"
    meta = ("_ok" if result.get("ok") else "_error")
    meta += f" · {float(result.get('seconds') or 0):.2f}s"
    if result.get("note"):
        meta += f" · {result['note']}"
    meta += "_"
    return (code_fence(code) + f"\n\n{fence}text\n{body.rstrip()}\n{fence}"
            + "\n\n" + meta)

"""Prompt rendering (full and incremental) and tool-call parsing for the agy text protocol."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_MARKER_START = "<<<TOOL_CALLS>>>"
TOOL_MARKER_END = "<<<END_TOOL_CALLS>>>"
_CLOSE_FOR = {TOOL_CALL_OPEN: TOOL_CALL_CLOSE, TOOL_MARKER_START: TOOL_MARKER_END}
_OPEN_TAGS = tuple(_CLOSE_FOR)

# Pruning of tool output in full prompts (persistent workers receive later results in full).
RECENT_TOOL_RESULTS_KEPT = 8
OLD_TOOL_RESULT_CHARS = 400
TOOL_RESULT_CAP_CHARS = 20_000

# Non-confrontational framing: telling the model it is "not Antigravity" makes some models treat the
# preamble as an injection and fall back to agy's built-in tools; redirecting the protocol works.
_HEADLESS_INTRO = (
    "You are running in HEADLESS INFERENCE MODE as the language model behind Hermes Agent. "
    "Your built-in tools (run_command, view_file, write_to_file, edit_file, browser and similar) are "
    "disabled in this mode and every attempt to use them is rejected. Hermes Agent performs all actions "
    "on the host."
)
_TOOL_PROTOCOL = f"""To act, call one of the Hermes tools listed below by writing a tool call block in your reply:
{TOOL_CALL_OPEN}
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
{TOOL_CALL_CLOSE}

Rules:
- Exactly one JSON object per {TOOL_CALL_OPEN} block; write several blocks in a row for parallel calls.
- "arguments" is a JSON object that matches the tool's parameter schema.
- Stop right after your last tool call block. Hermes runs the tools and sends the real results in the next message. Never invent, predict or simulate tool results.
- If no tool is needed, answer normally in plain text."""
_NO_TOOLS_NOTE = "No tools are available for this request: answer in plain text."
DELTA_REMINDER = (
    "(Headless mode: built-in tools stay disabled. Call Hermes tools only with "
    f"{TOOL_CALL_OPEN}...{TOOL_CALL_CLOSE} blocks, then stop.)"
)


# ── Message rendering ───────────────────────────────────────────────────────


def render_content(content: Any) -> str:
    """Flatten string / multipart content into text; images become an explicit placeholder."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind in ("text", "input_text", "output_text") or (kind is None and "text" in item):
                    parts.append(str(item.get("text") or ""))
                elif kind in ("image_url", "input_image", "image"):
                    parts.append("[image attached: this model cannot view images]")
                elif "content" in item:
                    parts.append(render_content(item.get("content")))
        return "\n".join(p for p in parts if p)
    return str(content)


def _role(msg: dict[str, Any]) -> str:
    role = str(msg.get("role") or "user").strip().lower()
    if role == "developer":
        return "system"
    if role == "function":
        return "tool"
    return role


def _tool_call_parts(tc: Any) -> tuple[str, str, str]:
    """(id, name, arguments-json) from a dict or attribute-style tool call."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        call_id = tc.get("id") or ""
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
    else:
        fn = getattr(tc, "function", None)
        call_id = getattr(tc, "id", "") or ""
        name = getattr(fn, "name", "") if fn is not None else ""
        args = getattr(fn, "arguments", "") if fn is not None else ""
    if not isinstance(args, str):
        args = json.dumps(args if args is not None else {}, ensure_ascii=False)
    return str(call_id or ""), str(name or ""), args


def _canonical_args(args: str) -> str:
    try:
        return json.dumps(json.loads(args), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (ValueError, TypeError):
        return args.strip()


def _render_tool_call(tc: Any) -> str:
    _, name, args = _tool_call_parts(tc)
    try:
        args_obj: Any = json.loads(args) if args.strip() else {}
    except ValueError:
        args_obj = args
    return f"{TOOL_CALL_OPEN}{json.dumps({'name': name, 'arguments': args_obj}, ensure_ascii=False)}{TOOL_CALL_CLOSE}"


def truncate_middle(text: str, limit: int) -> str:
    """Keep ~65% head and ~35% tail of an over-long text."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.65)
    tail = max(0, limit - head)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} characters of tool output omitted ...]\n{text[-tail:] if tail else ''}"


def _tool_names_by_id(messages: Iterable[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, dict) and _role(msg) == "assistant":
            for tc in msg.get("tool_calls") or []:
                call_id, name, _ = _tool_call_parts(tc)
                if call_id:
                    names[call_id] = name
    return names


def _render_message(msg: dict[str, Any], *, names: dict[str, str], tool_limit: int | None) -> str:
    role = _role(msg)
    text = render_content(msg.get("content")).strip()
    if role == "assistant":
        parts = [text] if text else []
        parts += [_render_tool_call(tc) for tc in msg.get("tool_calls") or []]
        return "[Assistant]\n" + "\n".join(parts) if parts else ""
    if role == "tool":
        call_id = str(msg.get("tool_call_id") or "")
        name = str(msg.get("name") or names.get(call_id) or "tool")
        failed = bool(msg.get("is_error")) or text.startswith(("Error:", "error:", "ERROR:"))
        label = f"[Tool Result: {name}{' (failed)' if failed else ''}{f' id={call_id}' if call_id else ''}]"
        body = truncate_middle(text, tool_limit if tool_limit is not None else TOOL_RESULT_CAP_CHARS)
        return f"{label}\n{body or '(no output)'}"
    if role == "system":
        return f"[System]\n{text}" if text else ""
    if role == "user":
        return f"[User]\n{text}" if text else ""
    return f"[{role}]\n{text}" if text else ""


# ── Tools ───────────────────────────────────────────────────────────────────


def _tool_specs(tools: Sequence[Any] | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name.strip():
            continue
        specs.append(
            {
                "name": name.strip(),
                "description": str(fn.get("description") or "").strip(),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return specs


def tool_names(tools: Sequence[Any] | None) -> frozenset[str]:
    return frozenset(spec["name"] for spec in _tool_specs(tools))


def tools_signature(tools: Sequence[Any] | None, tool_choice: Any = None) -> str:
    """Stable fingerprint of the tool set, to notice when Hermes changes it mid-session."""
    payload = json.dumps([_tool_specs(tools), tool_choice], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _tool_choice_hint(tool_choice: Any) -> str:
    if tool_choice in (None, "auto"):
        return ""
    if tool_choice == "none":
        return "Do not call any tool for this request."
    if tool_choice == "required":
        return "You must call at least one tool in this reply."
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else tool_choice
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            return f"You must call the tool `{name}` in this reply."
    return ""


def render_tools(tools: Sequence[Any] | None, tool_choice: Any = None) -> str:
    """Tool catalogue with full JSON schemas (sent once per persistent worker, so nested schemas fit)."""
    specs = _tool_specs(tools)
    if not specs:
        return ""
    lines = ["## Available Hermes tools"]
    for spec in specs:
        lines.append(f"\n### {spec['name']}")
        if spec["description"]:
            lines.append(spec["description"])
        schema = json.dumps(spec["parameters"], ensure_ascii=False, separators=(",", ":"))
        lines.append(f"Parameters (JSON Schema): {schema}")
    hint = _tool_choice_hint(tool_choice)
    if hint:
        lines.append(f"\n{hint}")
    return "\n".join(lines)


# ── Prompts ─────────────────────────────────────────────────────────────────


def _closing_instruction(last: dict[str, Any] | None) -> str:
    role = _role(last) if last else ""
    if role == "tool":
        return (
            "## Next step\nThe latest tool results are above. Continue the task: call more tools if needed, "
            "otherwise answer the user. Do not repeat earlier summaries."
        )
    if role == "user":
        return (
            "## Latest user message\n" + render_content(last.get("content")).strip() + "\n\n"  # type: ignore[union-attr]
            "Respond to this latest message. If it asks about or comments on earlier work instead of asking to "
            "continue it, address that directly rather than resuming the earlier task."
        )
    return "## Next step\nContinue the conversation from the latest message."


def format_full_prompt(
    messages: Sequence[dict[str, Any]] | None,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
) -> str:
    """Complete prompt for a fresh agy conversation."""
    msgs = [m for m in messages or [] if isinstance(m, dict)]
    sections = [_HEADLESS_INTRO]
    tool_section = render_tools(tools, tool_choice)
    sections.append(f"{_TOOL_PROTOCOL}\n\n{tool_section}" if tool_section else _NO_TOOLS_NOTE)

    system_texts = [render_content(m.get("content")).strip() for m in msgs if _role(m) == "system"]
    system_texts = [t for t in system_texts if t]
    if system_texts:
        sections.append("## Hermes system instructions\n" + "\n\n".join(system_texts))

    dialogue = [m for m in msgs if _role(m) != "system"]
    last = dialogue[-1] if dialogue else None
    history = dialogue[:-1] if last is not None and _role(last) == "user" else dialogue
    tool_positions = [i for i, m in enumerate(history) if _role(m) == "tool"]
    recent = set(tool_positions[-RECENT_TOOL_RESULTS_KEPT:])
    names = _tool_names_by_id(dialogue)
    rendered = [
        _render_message(m, names=names, tool_limit=None if i in recent else OLD_TOOL_RESULT_CHARS)
        for i, m in enumerate(history)
    ]
    rendered = [r for r in rendered if r]
    if rendered:
        sections.append("## Conversation so far\n\n" + "\n\n".join(rendered))
    sections.append(_closing_instruction(last))
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def format_delta_prompt(
    new_messages: Sequence[dict[str, Any]],
    *,
    context_messages: Sequence[dict[str, Any]] = (),
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
    tools_changed: bool = False,
) -> str:
    """Only what happened since the worker's last turn. ``context_messages`` resolves tool names."""
    msgs = [m for m in new_messages if isinstance(m, dict)]
    sections: list[str] = []
    if tools_changed:
        tool_section = render_tools(tools, tool_choice)
        sections.append(
            "## Tool list updated\nThe available Hermes tools changed; this list replaces the previous one.\n\n"
            + tool_section
            if tool_section
            else "## Tool list updated\n" + _NO_TOOLS_NOTE
        )
    last = msgs[-1] if msgs else None
    body = msgs[:-1] if last is not None and _role(last) == "user" else msgs
    names = _tool_names_by_id([*context_messages, *msgs])
    rendered = [r for r in (_render_message(m, names=names, tool_limit=None) for m in body) if r]
    if rendered:
        sections.append("\n\n".join(rendered))
    sections.append(_closing_instruction(last))
    if tools:
        sections.append(DELTA_REMINDER)
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


# ── Conversation continuity ─────────────────────────────────────────────────


def normalize_message(msg: Any) -> tuple[Any, ...]:
    """The parts of a message that matter to the model, in a comparison-stable form."""
    if not isinstance(msg, dict):
        return ("?", repr(msg))
    calls = tuple(
        (name, _canonical_args(args)) for _, name, args in (_tool_call_parts(tc) for tc in msg.get("tool_calls") or [])
    )
    return (
        _role(msg),
        render_content(msg.get("content")).strip(),
        calls,
        str(msg.get("tool_call_id") or ""),
    )


def message_key(msg: Any) -> bytes:
    """Compact digest of :func:`normalize_message`; workers keep these instead of whole transcripts."""
    payload = json.dumps(normalize_message(msg), ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def history_keys(messages: Sequence[Any]) -> tuple[bytes, ...]:
    return tuple(message_key(m) for m in messages)


def continuation_delta(history: Sequence[bytes], messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """New messages when ``messages`` strictly extends the recorded ``history`` keys, else None."""
    if not history or len(messages) <= len(history):
        return None
    for recorded, incoming in zip(history, messages):
        if recorded != message_key(incoming):
            return None
    return list(messages[len(history) :])


@dataclass(frozen=True)
class Reply:
    """What the worker itself answered on its last turn (already in agy's trajectory)."""

    content: str
    tool_names: tuple[str, ...]

    def matches(self, msg: dict[str, Any]) -> bool:
        if not isinstance(msg, dict) or _role(msg) != "assistant":
            return False
        names = tuple(_tool_call_parts(tc)[1] for tc in msg.get("tool_calls") or [])
        if names or self.tool_names:
            return names == self.tool_names
        mine = " ".join(self.content.split())
        theirs = " ".join(render_content(msg.get("content")).split())
        return mine == theirs or bool(mine and theirs and (theirs in mine or mine in theirs))


# ── Tool-call parsing ───────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_BARE_JSON_START_RE = re.compile(r"[\[{]|```json\b|```\s*\n\s*[\[{]", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.match(text)
    return (match.group(1) if match else text).strip()


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _call_from_object(obj: Any) -> SimpleNamespace | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    args: Any = obj.get("arguments", obj.get("parameters", obj.get("args", obj.get("input"))))
    fn = obj.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters", args))
    if not isinstance(name, str) or not name.strip():
        return None
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args_text = json.dumps(json.loads(args), ensure_ascii=False) if args.strip() else "{}"
        except ValueError:
            args_text = args
    else:
        args_text = json.dumps(args, ensure_ascii=False)
    # Ids are always minted here: model-written ids ("call_1") repeat across turns.
    return SimpleNamespace(
        id=new_call_id(), type="function", function=SimpleNamespace(name=name.strip(), arguments=args_text)
    )


def calls_from_payload(payload: str) -> list[SimpleNamespace] | None:
    """Tool calls in a JSON payload (object or list), or None when it is not valid tool JSON."""
    raw = _strip_fences(payload)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    items = parsed if isinstance(parsed, list) else [parsed]
    calls = [c for c in (_call_from_object(item) for item in items) if c is not None]
    return calls or None


def _partial_prefix_len(text: str) -> int:
    """Length of the longest suffix of ``text`` that could start an opening tag."""
    best = 0
    for tag in _OPEN_TAGS:
        for k in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:k]):
                best = max(best, k)
                break
    return best


class ToolCallStreamParser:
    """Splits streamed model text into display text and tool calls.

    Text is released as soon as it cannot be the start of a tool block. Text after the first tool
    call is dropped (the model was told to stop there; Hermes would suppress it anyway). A reply that
    opens with bare JSON is held back and accepted as tool calls only when every name is a known tool.
    """

    def __init__(self, *, enabled: bool = True, known_tools: frozenset[str] = frozenset()) -> None:
        self.enabled = enabled
        self.known_tools = known_tools
        self._buf = ""
        self._block: str | None = None
        self._started = False
        self._bare_json = False
        self.saw_calls = False
        self.dropped_text = ""

    def _text(self, out: list[tuple[str, Any]], text: str) -> None:
        if not text:
            return
        if self.saw_calls:
            self.dropped_text += text
            return
        out.append(("text", text))

    def feed(self, text: str) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        if not text:
            return out
        if not self.enabled:
            out.append(("text", text))
            return out
        self._buf += text
        if not self._started:
            stripped = self._buf.lstrip()
            if not stripped:
                return out
            if stripped[0] == "`" and len(stripped) < 8 and "\n" not in stripped:
                return out  # not enough text yet to tell a ```json fence from a code answer
            self._started = True
            self._bare_json = bool(self.known_tools) and bool(_BARE_JSON_START_RE.match(stripped))
        if self._bare_json:
            return out  # decided in finish()
        self._scan(out)
        return out

    def _scan(self, out: list[tuple[str, Any]]) -> None:
        while self._buf:
            if self._block is None:
                hits = [(self._buf.find(tag), tag) for tag in _OPEN_TAGS]
                hits = [(i, t) for i, t in hits if i != -1]
                if hits:
                    idx, tag = min(hits)
                    self._text(out, self._buf[:idx])
                    self._buf = self._buf[idx + len(tag) :]
                    self._block = tag
                    continue
                keep = _partial_prefix_len(self._buf)
                self._text(out, self._buf[: len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep :]
                return
            close = _CLOSE_FOR[self._block]
            end = self._buf.find(close)
            if end == -1:
                return
            payload, tag = self._buf[:end], self._block
            self._buf = self._buf[end + len(close) :]
            self._block = None
            self._emit_block(out, tag, payload, close)

    def _emit_block(self, out: list[tuple[str, Any]], tag: str, payload: str, close: str) -> None:
        calls = calls_from_payload(payload)
        if calls:
            self.saw_calls = True
            out.extend(("call", c) for c in calls)
        else:
            logger.debug("Unparseable tool block from agy: %.200s", payload)
            self._text(out, f"{tag}{payload}{close}")

    def finish(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        if not self.enabled:
            return out
        if not self._started:
            self._started = True
            self._scan(out)
        elif self._bare_json:
            self._bare_json = False
            calls = calls_from_payload(self._buf)
            if calls and all(c.function.name in self.known_tools for c in calls):
                self._buf = ""
                self.saw_calls = True
                out.extend(("call", c) for c in calls)
                return out
            self._scan(out)
        if self._block is not None:
            # Unterminated block: accept it if the payload is complete JSON.
            tag, payload = self._block, self._buf
            self._block, self._buf = None, ""
            calls = calls_from_payload(payload)
            if calls:
                self.saw_calls = True
                out.extend(("call", c) for c in calls)
            else:
                self._text(out, f"{tag}{payload}")
        elif self._buf:
            self._text(out, self._buf)
            self._buf = ""
        return out


def parse_tool_calls(text: str, known_tools: frozenset[str] = frozenset()) -> tuple[str, list[SimpleNamespace] | None]:
    """Whole-text variant of :class:`ToolCallStreamParser`: (display text, tool calls or None)."""
    parser = ToolCallStreamParser(enabled=True, known_tools=known_tools)
    events = parser.feed(text or "") + parser.finish()
    cleaned = "".join(v for k, v in events if k == "text").strip()
    calls = [v for k, v in events if k == "call"]
    return cleaned, calls or None

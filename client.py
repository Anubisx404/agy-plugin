from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Iterator

try:
    from .models import find_agy, resolve_model
except ImportError:
    from models import find_agy, resolve_model


TOOL_MARKER_START = "<<<TOOL_CALLS>>>"
TOOL_MARKER_END = "<<<END_TOOL_CALLS>>>"
TAG_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

TOOL_SYSTEM_INSTRUCTIONS = """You have access to tools. When you need to call a tool, you MUST emit your tool call in one of these EXACT formats:

<tool_call>
{"name": "function_name", "arguments": {"param": "value"}}
</tool_call>

Or:
<<<TOOL_CALLS>>>
[{"name": "function_name", "arguments": {"param": "value"}}]
<<<END_TOOL_CALLS>>>

Rules:
- Output ONLY the tool call block when calling a tool.
- Do not execute local shell commands or files directly; only emit tool call blocks so Hermes executes them safely.
- Never hallucinate tool results. Hermes Agent executes the tools and provides real results in the next turn.
- If no tool is needed, respond directly with text.

Available tools:
"""


def _clean_markdown_code_blocks(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _tools_to_description(tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(func, dict):
            continue
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"\n### {name}")
        if desc:
            lines.append(str(desc))
        if isinstance(params, dict) and params.get("properties"):
            lines.append("Parameters:")
            required = set(params.get("required", []))
            for pname, pschema in params["properties"].items():
                if isinstance(pschema, dict):
                    ptype = pschema.get("type", "any")
                    pdesc = pschema.get("description", "")
                    req = " (required)" if pname in required else ""
                    lines.append(f"  - {pname}: {ptype}{req} — {pdesc}")
    return "\n".join(lines)


def _prune_text(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    tail = max_chars - half - 30
    return f"{text[:half]}\n... [output truncated {len(text) - max_chars} characters] ...\n{text[-tail:]}"



def _format_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
        return "\n".join(parts)
    return str(content)


def _parse_single_tool_object(raw_obj: Any) -> SimpleNamespace | None:
    if not isinstance(raw_obj, dict):
        return None
    call_id = raw_obj.get("id") or f"call_{uuid.uuid4().hex[:12]}"
    name = raw_obj.get("name")
    arguments = raw_obj.get("arguments", {})
    if not name and "function" in raw_obj and isinstance(raw_obj["function"], dict):
        name = raw_obj["function"].get("name")
        arguments = raw_obj["function"].get("arguments", {})
    if not name or not isinstance(name, str):
        return None
    if isinstance(arguments, dict):
        arg_str = json.dumps(arguments, ensure_ascii=False)
    elif isinstance(arguments, str):
        arg_str = arguments
    else:
        arg_str = json.dumps(arguments, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arg_str),
    )


def parse_tool_calls(text: str) -> tuple[str, list[SimpleNamespace] | None]:
    if not text:
        return "", None
    tool_calls: list[SimpleNamespace] = []
    cleaned_text = text

    for match in TAG_TOOL_CALL_RE.finditer(text):
        raw_inner = _clean_markdown_code_blocks(match.group(1))
        try:
            parsed = json.loads(raw_inner)
            if isinstance(parsed, list):
                for item in parsed:
                    tc = _parse_single_tool_object(item)
                    if tc:
                        tool_calls.append(tc)
            else:
                tc = _parse_single_tool_object(parsed)
                if tc:
                    tool_calls.append(tc)
        except Exception:
            continue

    if tool_calls:
        cleaned_text = TAG_TOOL_CALL_RE.sub("", cleaned_text).strip()

    start_idx = cleaned_text.find(TOOL_MARKER_START)
    while start_idx != -1:
        end_idx = cleaned_text.find(TOOL_MARKER_END, start_idx)
        if end_idx == -1:
            break
        raw_payload = cleaned_text[start_idx + len(TOOL_MARKER_START) : end_idx]
        cleaned_text = (cleaned_text[:start_idx] + cleaned_text[end_idx + len(TOOL_MARKER_END) :]).strip()
        raw_payload = _clean_markdown_code_blocks(raw_payload)
        try:
            parsed = json.loads(raw_payload)
            if isinstance(parsed, list):
                for item in parsed:
                    tc = _parse_single_tool_object(item)
                    if tc:
                        tool_calls.append(tc)
            else:
                tc = _parse_single_tool_object(parsed)
                if tc:
                    tool_calls.append(tc)
        except Exception:
            pass
        start_idx = cleaned_text.find(TOOL_MARKER_START)

    return cleaned_text.strip(), (tool_calls if tool_calls else None)


def format_messages_as_prompt(
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = [
        "You are operating in HEADLESS INFERENCE MODE for Hermes Agent.",
        "All file editing, command executions, and interactions are performed on the host by Hermes.",
        "You must NEVER execute native tools or commands yourself. Only generate text or tool-call blocks.",
    ]
    if tools:
        tool_desc = _tools_to_description(tools)
        parts.append(f"{TOOL_SYSTEM_INSTRUCTIONS}{tool_desc}")

    msgs = list(messages or [])
    tool_message_indices = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    preserve_indices = set(tool_message_indices[-8:]) if len(tool_message_indices) > 8 else set(tool_message_indices)

    for i, msg in enumerate(msgs):
        role = msg.get("role", "user")
        content = _format_message_content(msg.get("content"))
        if role == "system":
            parts.append(f"[System Instructions]\n{content}")
        elif role == "assistant":
            tc_list = msg.get("tool_calls")
            if tc_list and isinstance(tc_list, list):
                actions: list[str] = []
                for tc in tc_list:
                    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                    if fn:
                        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
                        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                        call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                        id_str = f" id={call_id}" if call_id else ""
                        actions.append(f"Called {name}({args}){id_str}")
                parts.append("[Previous Assistant Action]\n" + "\n".join(actions))
            elif content:
                parts.append(f"[Previous Assistant Response]\n{content}")
        elif role == "tool":
            tool_name = msg.get("name", "tool")
            is_err = bool(msg.get("is_error")) or str(content).strip().startswith("Error:")
            status_label = " (Failed)" if is_err else ""
            call_id = msg.get("tool_call_id")
            id_label = f" id={call_id}" if call_id else ""
            body = content if i in preserve_indices else _prune_text(content, 400 if is_err else 300)
            parts.append(f"[Tool Result: {tool_name}{status_label}{id_label}]\n{body}")
        elif role == "user":
            parts.append(f"[User]\n{content}" if len(msgs) > 1 else content)
        else:
            parts.append(f"[{role}]\n{content}")

    parts.append("Continue the conversation from the latest turn. Respond directly or emit a tool call.")
    return "\n\n".join(p for p in parts if p)



class AntigravityClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or "antigravity"
        self.base_url = base_url or "agy://local"
        self._command = command or find_agy()
        self._default_args = list(args or [])
        self._active_processes: set[subprocess.Popen[str]] = set()
        self._lock = threading.Lock()
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    def _terminate_proc(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        if sys.platform == "win32":
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            return
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=2)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()

    def close(self) -> None:
        with self._lock:
            procs = tuple(self._active_processes)
            self._active_processes.clear()
            self.is_closed = True
        for p in procs:
            self._terminate_proc(p)

    def _track_proc(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            self._active_processes.add(proc)

    def _untrack_proc(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            self._active_processes.discard(proc)

    def _build_cmd(self, model: str, timeout_seconds: int) -> list[str]:
        cmd = [
            self._command,
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--disable-slash-commands",
            "--print-timeout",
            f"{timeout_seconds}s",
        ]
        cmd.extend(self._default_args)
        mode = os.environ.get("AGY_MODE")
        if mode:
            cmd.extend(["--mode", mode])
        return cmd

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        timeout_sec = int(timeout) if timeout and timeout > 0 else 300
        resolved = resolve_model(model or "gemini-3.8-flash-medium", reasoning_effort=reasoning_effort)
        prompt = format_messages_as_prompt(messages, tools=tools)
        cmd = self._build_cmd(resolved, timeout_sec)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        sandbox = tempfile.mkdtemp(prefix="hermes_agy_")

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=sandbox,
            creationflags=flags,
        )
        self._track_proc(proc)

        if proc.stdin:
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except Exception:
                pass

        if stream:
            return self._stream_generator(proc, resolved, prompt, sandbox, tools=bool(tools))

        output_chunks: list[str] = []
        usage_data: dict[str, Any] = {}
        try:
            for line in iter(proc.stdout.readline, "") if proc.stdout else []:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                event = data.get("event")
                if event == "step_update":
                    su = data.get("step_update", {})
                    step_type = su.get("step_type")
                    if step_type == "tool":
                        self._terminate_proc(proc)
                        raise RuntimeError(
                            f"Antigravity native tool execution intercepted ({su.get('tool_name')}). "
                            "Native tools are blocked to preserve Hermes orchestration."
                        )
                    if step_type == "agent_response" and "text_delta" in su:
                        output_chunks.append(su["text_delta"])
                    if "usage" in su:
                        usage_data.update(su["usage"])
                elif event == "result":
                    res = data.get("result", {})
                    if "usage" in res:
                        usage_data.update(res["usage"])
                    if not output_chunks and res.get("response"):
                        output_chunks.append(res["response"])
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self._terminate_proc(proc)
            raise TimeoutError(f"Antigravity process timed out after {timeout_sec}s")
        finally:
            self._untrack_proc(proc)
            self._terminate_proc(proc)
            shutil.rmtree(sandbox, ignore_errors=True)

        output = "".join(output_chunks)
        cleaned_content, tool_calls = parse_tool_calls(output)
        finish_reason = "tool_calls" if tool_calls else "stop"

        p_tokens = usage_data.get("input_tokens") or len(prompt.split())
        c_tokens = usage_data.get("output_tokens") or len(output.split())
        t_tokens = usage_data.get("total_tokens") or (p_tokens + c_tokens)
        cached_tokens = usage_data.get("cache_read_tokens") or 0

        message = SimpleNamespace(
            role="assistant",
            content=cleaned_content if cleaned_content else None,
            tool_calls=tool_calls,
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
        usage = SimpleNamespace(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        )
        return SimpleNamespace(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=resolved,
            choices=[choice],
            usage=usage,
        )

    def _stream_generator(
        self,
        proc: subprocess.Popen[str],
        model: str,
        prompt: str,
        sandbox: str,
        tools: bool = False,
    ) -> Iterator[SimpleNamespace]:
        request_id = uuid.uuid4().hex[:12]
        accumulated_text: list[str] = []
        usage_data: dict[str, Any] = {}
        try:
            for line in iter(proc.stdout.readline, "") if proc.stdout else []:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                event = data.get("event")
                if event == "step_update":
                    su = data.get("step_update", {})
                    step_type = su.get("step_type")
                    if step_type == "tool":
                        self._terminate_proc(proc)
                        raise RuntimeError(
                            f"Antigravity native tool execution intercepted ({su.get('tool_name')}). "
                            "Native tools are blocked to preserve Hermes orchestration."
                        )
                    if step_type == "agent_response" and "text_delta" in su:
                        chunk = su["text_delta"]
                        accumulated_text.append(chunk)
                        if not tools:
                            delta = SimpleNamespace(role="assistant", content=chunk, tool_calls=None)
                            yield SimpleNamespace(
                                id=f"chatcmpl-{request_id}",
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model,
                                choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
                            )
                    if "usage" in su:
                        usage_data.update(su["usage"])
                elif event == "result":
                    res = data.get("result", {})
                    if "usage" in res:
                        usage_data.update(res["usage"])

            proc.wait()

            full_text = "".join(accumulated_text)
            cleaned_text, tool_calls = parse_tool_calls(full_text)

            if tools and tool_calls:
                if cleaned_text:
                    delta = SimpleNamespace(role="assistant", content=cleaned_text, tool_calls=None)
                    yield SimpleNamespace(
                        id=f"chatcmpl-{request_id}",
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
                    )
                deltas = [
                    SimpleNamespace(
                        index=i,
                        id=tc.id,
                        type="function",
                        function=SimpleNamespace(name=tc.function.name, arguments=tc.function.arguments),
                    )
                    for i, tc in enumerate(tool_calls)
                ]
                delta = SimpleNamespace(role=None, content=None, tool_calls=deltas)
                yield SimpleNamespace(
                    id=f"chatcmpl-{request_id}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[SimpleNamespace(index=0, delta=delta, finish_reason="tool_calls")],
                )
            elif tools:
                delta = SimpleNamespace(role="assistant", content=full_text, tool_calls=None)
                yield SimpleNamespace(
                    id=f"chatcmpl-{request_id}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[SimpleNamespace(index=0, delta=delta, finish_reason="stop")],
                )
            else:
                final_delta = SimpleNamespace(role=None, content=None, tool_calls=None)
                yield SimpleNamespace(
                    id=f"chatcmpl-{request_id}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[SimpleNamespace(index=0, delta=final_delta, finish_reason="stop")],
                )
        finally:
            self._untrack_proc(proc)
            self._terminate_proc(proc)
            shutil.rmtree(sandbox, ignore_errors=True)


from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
DEFAULT_PORT = 8642
DEFAULT_HOST = "127.0.0.1"
MODEL_CACHE_TTL = 300

TOOL_CALL_START = "<<<TOOL_CALLS>>>"
TOOL_CALL_END = "<<<END_TOOL_CALLS>>>"

TOOL_CALLING_INSTRUCTIONS = """
You have access to tools. When you need to use a tool, you MUST output your tool calls in this EXACT format — no other format will work:

<<<TOOL_CALLS>>>
[{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}]
<<<END_TOOL_CALLS>>>

Rules:
- Output ONLY the tool call block when calling tools — no text before or after it.
- You may call multiple tools at once by putting multiple objects in the array.
- "arguments" must be a JSON object matching the function's parameters.
- If you want to respond with text (no tool call), just write your response normally WITHOUT the markers.
- NEVER put tool calls inside markdown code blocks. Use the raw markers directly.

Available tools:
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [agy-proxy] %(message)s",
)
log = logging.getLogger("agy_proxy")

app = FastAPI(title="AGY Proxy", version="2.0.0")

from dashboard import router as dashboard_router, tracker as usage_tracker  # noqa: E402

app.include_router(dashboard_router)

_model_cache: dict[str, Any] = {"ts": 0.0, "models": []}


def _find_agy() -> str:
    override = os.environ.get("AGY_CLI_PATH")
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"AGY_CLI_PATH set but not found: {p}")
    found = shutil.which("agy") or shutil.which("agy.exe")
    if not found:
        raise FileNotFoundError(
            "agy CLI not found in PATH. Install it from https://github.com/google/anthropic-cli "
            "or set AGY_CLI_PATH to the binary location."
        )
    return found


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _subprocess_flags() -> int:
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def _fetch_models() -> list[dict[str, str]]:
    now = time.time()
    if now - _model_cache["ts"] < MODEL_CACHE_TTL and _model_cache["models"]:
        return _model_cache["models"]
    try:
        agy = _find_agy()
        result = subprocess.run(
            [agy, "models"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_subprocess_flags(),
        )
        raw = _strip_ansi(result.stdout + result.stderr)
        models = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            model_id = parts[0]
            display = parts[1].strip() if len(parts) > 1 else model_id
            if any(model_id.startswith(p) for p in ("gemini", "claude", "gpt")):
                models.append({"id": model_id, "name": display})
        if models:
            _model_cache["ts"] = now
            _model_cache["models"] = models
        return models
    except Exception as exc:
        log.error("Failed to fetch models: %s", exc)
        return _model_cache.get("models", [])


def _tools_to_description(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"\n### {name}")
        if desc:
            lines.append(desc)
        if params.get("properties"):
            lines.append("Parameters:")
            required = set(params.get("required", []))
            for pname, pschema in params["properties"].items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req = " (required)" if pname in required else ""
                lines.append(f"  - {pname}: {ptype}{req} — {pdesc}")
    return "\n".join(lines)


def _parse_tool_calls(text: str) -> tuple[str | None, list[dict] | None]:
    start_idx = text.find(TOOL_CALL_START)
    if start_idx == -1:
        return text, None

    end_idx = text.find(TOOL_CALL_END, start_idx)
    if end_idx == -1:
        return text, None

    before = text[:start_idx].strip()
    json_str = text[start_idx + len(TOOL_CALL_START) : end_idx].strip()

    try:
        calls_raw = json.loads(json_str)
    except json.JSONDecodeError:
        log.warning("Failed to parse tool calls JSON: %s", json_str[:200])
        return text, None

    if not isinstance(calls_raw, list):
        calls_raw = [calls_raw]

    tool_calls = []
    for call in calls_raw:
        if not isinstance(call, dict) or "name" not in call:
            continue
        arguments = call.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": arguments,
                },
            }
        )

    if not tool_calls:
        return text, None

    return (before if before else None), tool_calls


def _messages_to_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    parts = []

    if tools:
        tool_desc = _tools_to_description(tools)
        parts.append(f"[System Instructions]\n{TOOL_CALLING_INSTRUCTIONS}{tool_desc}")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        if content is None:
            content = ""

        if role == "system":
            parts.append(f"[System Instructions]\n{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                calls_summary = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    calls_summary.append(f"Called {fn.get('name', '?')}({fn.get('arguments', '{}')})")
                parts.append("[Previous Assistant Action]\n" + "\n".join(calls_summary))
            elif content:
                parts.append(f"[Previous Assistant Response]\n{content}")
        elif role == "tool":
            tool_name = msg.get("name", "unknown_tool")
            parts.append(f"[Tool Result: {tool_name}]\n{content}")
        elif role == "user":
            parts.append(content)
        else:
            parts.append(f"[{role}]\n{content}")

    return "\n\n".join(parts)


def _get_cwd(working_directory: str | None = None) -> str:
    return working_directory or os.environ.get("AGY_PROXY_CWD") or str(Path.home())


def _run_agy_sync(
    prompt: str,
    model: str,
    timeout_seconds: int = 300,
    working_directory: str | None = None,
) -> str:
    agy = _find_agy()
    cmd = [
        agy,
        "--model",
        model,
        "--mode",
        "plan",
        "--output-format",
        "text",
        "--print-timeout",
        f"{timeout_seconds}s",
    ]

    cwd = _get_cwd(working_directory)
    log.info("Running AGY: model=%s cwd=%s prompt_len=%d", model, cwd, len(prompt))

    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds + 30,
        cwd=cwd,
        creationflags=_subprocess_flags(),
    )

    output = _strip_ansi(proc.stdout or "")
    if not output.strip() and proc.stderr:
        stderr_clean = _strip_ansi(proc.stderr)
        filtered = "\n".join(
            line for line in stderr_clean.splitlines() if not line.strip().startswith("Fetching") and line.strip()
        )
        if filtered.strip():
            output = filtered

    return output.strip()


async def _run_agy_streaming(
    prompt: str,
    model: str,
    timeout_seconds: int = 300,
    working_directory: str | None = None,
    request_id: str = "",
) -> AsyncIterator[str]:
    agy = _find_agy()
    cmd = [
        agy,
        "--model",
        model,
        "--mode",
        "plan",
        "--output-format",
        "text",
        "--print-timeout",
        f"{timeout_seconds}s",
    ]

    cwd = _get_cwd(working_directory)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        creationflags=_subprocess_flags(),
    )

    if proc.stdin:
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()

    buffer = ""

    try:
        while True:
            chunk = await asyncio.wait_for(
                proc.stdout.read(256),
                timeout=timeout_seconds + 30,
            )
            if not chunk:
                break
            text = _strip_ansi(chunk.decode("utf-8", errors="replace"))
            buffer += text

            while "\n" in buffer or len(buffer) > 80:
                if "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    content = line + "\n"
                else:
                    content = buffer
                    buffer = ""

                delta = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(delta)}\n\n"

        if buffer.strip():
            delta = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": buffer},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(delta)}\n\n"

        final = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        if proc.returncode is None:
            try:
                proc.terminate()
            except Exception:
                pass


def _usage(prompt: str, output: str) -> dict:
    return {
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": len(output.split()),
        "total_tokens": len(prompt.split()) + len(output.split()),
    }


def _completion_response(
    request_id: str,
    model: str,
    content: str | None,
    tool_calls: list[dict] | None = None,
    prompt: str = "",
    output: str = "",
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": _usage(prompt, output),
    }


@app.get("/v1/models")
async def list_models():
    models = _fetch_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": "antigravity",
                "permission": [],
                "root": m["id"],
                "parent": None,
            }
            for m in models
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    models = _fetch_models()
    for m in models:
        if m["id"] == model_id:
            return {
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": "antigravity",
                "permission": [],
                "root": m["id"],
                "parent": None,
            }
    raise HTTPException(404, f"Model not found: {model_id}")


@app.get("/health")
async def health():
    try:
        _find_agy()
        return {"status": "ok", "version": "2.0.0", "tool_calling": True, "models": len(_fetch_models())}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    model = body.get("model", "gemini-3.8-flash-medium")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    tools = body.get("tools")
    timeout = min(body.get("timeout", 300), 1800)

    if not messages:
        raise HTTPException(400, "messages is required")

    known = {m["id"] for m in _fetch_models()}
    if model not in known:
        available = ", ".join(sorted(known)) if known else "none available"
        raise HTTPException(400, f"Unknown model: {model}. Available: {available}")

    has_tools = bool(tools and isinstance(tools, list) and len(tools) > 0)
    prompt = _messages_to_prompt(messages, tools if has_tools else None)
    request_id = uuid.uuid4().hex[:12]

    if has_tools:
        t0 = time.time()
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, _run_agy_sync, prompt, model, timeout)
        duration_ms = (time.time() - t0) * 1000

        content, tool_calls = _parse_tool_calls(output)
        log.info("Tool call parse: has_tool_calls=%s content_len=%s", tool_calls is not None, len(content or ""))

        response = _completion_response(request_id, model, content, tool_calls, prompt, output)
        usage_tracker.record(
            model, response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
            messages=messages, duration_ms=duration_ms, stream=stream,
        )

        if stream:

            async def _stream_response():
                yield f"data: {json.dumps(response)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return response

    if stream:
        usage_tracker.record(
            model, len(prompt.split()), 0,  # ponytail: completion tokens unknown in stream
            messages=messages, stream=True,
        )
        return StreamingResponse(
            _run_agy_streaming(prompt, model, timeout, request_id=request_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Request-Id": request_id},
        )

    t0 = time.time()
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, _run_agy_sync, prompt, model, timeout)
    duration_ms = (time.time() - t0) * 1000
    resp = _completion_response(request_id, model, output, prompt=prompt, output=output)
    usage_tracker.record(
        model, resp["usage"]["prompt_tokens"], resp["usage"]["completion_tokens"],
        messages=messages, duration_ms=duration_ms, stream=False,
    )
    return resp


def main():
    parser = argparse.ArgumentParser(description="AGY Proxy — OpenAI-compatible API for Antigravity CLI")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    args = parser.parse_args()

    try:
        agy = _find_agy()
        log.info("Found AGY at: %s", agy)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    models = _fetch_models()
    log.info("Available models: %s", [m["id"] for m in models])
    log.info("Tool calling: enabled")
    log.info("Starting proxy on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

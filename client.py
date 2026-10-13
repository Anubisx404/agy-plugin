"""OpenAI-shaped client facade that runs Hermes turns on the agy CLI.

A turn runs on a persistent pooled worker when possible (see ``worker.py``) and falls back to a
one-shot ``agy`` process otherwise. Either way agy's stream-json events are translated into OpenAI
``chat.completion.chunk`` objects; tool calls written in the text protocol become real tool calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

try:
    from .models import DEFAULT_MODEL, resolve_model, strip_ansi
    from .process import (
        ABORTED,
        EOF,
        LineReader,
        QuotaLogWatch,
        StderrTail,
        build_child_env,
        env_float,
        find_quota_error,
        getenv_any,
        kill_tree,
        kill_tree_in_background,
        remove_dir,
        spawn,
        which_agy,
    )
    from .prompt import (
        Reply,
        ToolCallStreamParser,
        format_delta_prompt,
        format_full_prompt,
        tool_names,
        tools_signature,
    )
    from .worker import CLI_CAPS, POOL, Lease, effort_args, mode_args
except ImportError:
    from models import DEFAULT_MODEL, resolve_model, strip_ansi
    from process import (
        ABORTED,
        EOF,
        LineReader,
        QuotaLogWatch,
        StderrTail,
        build_child_env,
        env_float,
        find_quota_error,
        getenv_any,
        kill_tree,
        kill_tree_in_background,
        remove_dir,
        spawn,
        which_agy,
    )
    from prompt import (
        Reply,
        ToolCallStreamParser,
        format_delta_prompt,
        format_full_prompt,
        tool_names,
        tools_signature,
    )
    from worker import CLI_CAPS, POOL, Lease, effort_args, mode_args

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
ONESHOT_PRINT_TIMEOUT_SECONDS = 1800
SESSION_KEY_FIELD = "agy_session_id"
_CONTENT_STEP_TYPES = frozenset({"", "agent_response", "response", "text", "message", "assistant"})
_FAILED_STATUSES = frozenset({"ERROR", "FAILED", "FAILURE", "CANCELLED", "CANCELED", "ABORTED", "TIMEOUT"})
_UNKNOWN_FLAG = r"(?:unknown (?:flag|option|shorthand)|flag provided but not defined|unrecognized (?:argument|option))"
_INPUT_FORMAT_REJECTED_RE = re.compile(
    rf"{_UNKNOWN_FLAG}.*input-format|input-format.*(?:invalid|unsupported|unknown)", re.IGNORECASE
)
_EFFORT_REJECTED_RE = re.compile(rf"{_UNKNOWN_FLAG}.*effort", re.IGNORECASE)
_AUTH_RE = re.compile(
    r"not (?:authenticated|logged in|signed in)|authentication required|please (?:log|sign) in", re.IGNORECASE
)


class AgyError(RuntimeError):
    """A failed Antigravity turn. ``kind`` drives Hermes' error classification."""

    def __init__(self, message: str, *, kind: str = "execution") -> None:
        super().__init__(message)
        self.kind = kind


class _RetryTurn(Exception):
    def __init__(self, mode: str, error: AgyError) -> None:
        super().__init__(str(error))
        self.mode = mode  # "fresh" (new worker, full prompt) or "oneshot"
        self.error = error


def coerce_timeout(timeout: Any) -> float:
    """Seconds of silence tolerated from agy. Accepts numbers and httpx.Timeout-like objects."""
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
        return float(timeout)
    for attr in ("read", "timeout"):
        value = getattr(timeout, attr, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    return env_float("AGY_TIMEOUT_SECONDS", default=DEFAULT_TIMEOUT_SECONDS)


def _as_dict(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        with contextlib.suppress(Exception):
            return message.model_dump()
    if hasattr(message, "__dict__"):
        return dict(vars(message))
    return None


def _dump_prompt(prompt: str, *, branch: str, model: str) -> None:
    """Write the outgoing prompt to AGY_DEBUG_PROMPT_DIR (0600) for offline diagnosis."""
    target = getenv_any("AGY_DEBUG_PROMPT_DIR", "ANTIGRAVITY_DEBUG_PROMPT_DIR")
    if not target:
        return
    try:
        directory = Path(target)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / f"{stamp}_{branch}.txt"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(f"=== agy prompt ({branch}) model={model} chars={len(prompt)} ===\n{prompt}")
    except Exception as exc:
        logger.debug("Prompt dump failed: %s", exc)


def _retire_oneshot(proc: Any, sandbox: str | None, grace: float) -> None:
    if proc is not None:
        if grace > 0:
            with contextlib.suppress(Exception):
                proc.wait(timeout=grace)
        kill_tree(proc)
    remove_dir(sandbox)


async def _ready(value: Any) -> Any:
    return value


class Completion(SimpleNamespace):
    """A finished ChatCompletion that can also be awaited (Hermes' async auxiliary path)."""

    def __await__(self) -> Any:
        return _ready(self).__await__()


def _counter_delta(raw: dict[str, Any], baseline: dict[str, int], name: str) -> int | None:
    """Per-turn value of a cumulative worker counter (agy reports session totals)."""
    if name not in raw:
        return None
    try:
        value = int(raw.get(name) or 0)
    except (TypeError, ValueError):
        return None
    previous = baseline.get(name)
    baseline[name] = value
    if previous is None or value < previous:
        return value  # first report, or agy restarted its counters
    return value - previous


class _TurnRunner:
    """One chat completion request against agy, from prompt to verified result."""

    def __init__(
        self,
        client: AntigravityClient,
        *,
        model: str,
        effort: str | None,
        messages: list[dict[str, Any]],
        tools: list[Any] | None,
        tool_choice: Any,
        timeout: float,
        session_key: str | None,
        output_instruction: str = "",
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.messages = messages
        self.tools = tools or None
        self.tool_choice = tool_choice
        self.timeout = timeout
        self.session_key = session_key
        self.output_instruction = output_instruction
        self.known_tools = tool_names(self.tools)
        self.tools_sig = tools_signature(self.tools, tool_choice)
        self.response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())

        # RLock: an abandoned stream's generator may be finalized by the cyclic GC on a thread that
        # already holds this lock; its cleanup re-enters it.
        self._lock = threading.RLock()
        self._aborted: str | None = None
        self._committed = False
        self._cleaned = False
        self._mode = "auto"  # auto | fresh | oneshot
        self.lease: Lease | None = None
        self.proc: Any = None
        self.sandbox: str | None = None
        self.reader: LineReader | None = None
        self.stderr: StderrTail | None = None
        self.quota: QuotaLogWatch | None = None
        self.stderr_mark = 0
        self.prompt_chars = 0
        self.yielded = False

    # ── lifecycle ──

    def abort(self, reason: str) -> None:
        """Stop the turn from any thread. No-op once the turn has completed."""
        with self._lock:
            if self._committed or self._cleaned or self._aborted:
                return
            self._aborted = reason
            lease, proc, reader = self.lease, self.proc, self.reader
        if reader is not None:
            reader.abort()
        if lease is not None:
            POOL.discard(lease.worker)
        elif proc is not None:
            kill_tree_in_background(proc)

    def _release_resources(self, *, success: bool, reply: Reply | None = None, conversation_id: str = "") -> None:
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True
            lease, proc, sandbox = self.lease, self.proc, self.sandbox
            self.lease = self.proc = self.reader = None
            self.sandbox = None
        if lease is not None:
            if success and reply is not None:
                POOL.release(
                    lease,
                    messages=self.messages,
                    reply=reply,
                    tools_sig=self.tools_sig,
                    conversation_id=conversation_id or None,
                )
            else:
                POOL.discard(lease.worker)
        if proc is not None or sandbox:
            # Off-thread: a finished one-shot agy gets a moment to exit on its own (it may still be
            # writing its conversation store); a failed one is killed at once. The caller never waits.
            threading.Thread(
                target=_retire_oneshot,
                args=(proc, sandbox, 10.0 if success else 0.0),
                name="agy-oneshot-exit",
                daemon=True,
            ).start()

    def _reset_for_retry(self) -> None:
        self._release_resources(success=False)
        with self._lock:
            self._cleaned = False

    # ── opening a turn ──

    def _prompt_for(self, lease: Lease | None) -> tuple[str, str]:
        if lease is not None and lease.delta is not None:
            delta = lease.delta[1:] if lease.skip_reply else lease.delta
            context = self.messages[: len(self.messages) - len(delta)]  # includes a skipped own reply
            prompt = format_delta_prompt(
                delta,
                context_messages=context,
                tools=self.tools,
                tool_choice=self.tool_choice,
                tools_changed=lease.tools_changed,
            )
            branch = "delta"
        else:
            prompt = format_full_prompt(self.messages, tools=self.tools, tool_choice=self.tool_choice)
            branch = "full" if lease is not None else "oneshot"
        if self.output_instruction:
            prompt = f"{prompt}\n\n{self.output_instruction}"
        return prompt, branch

    def _open(self) -> None:
        command = self.client.resolve_command()
        if self._mode != "oneshot":
            if self._mode == "fresh" and self.session_key:
                POOL.forget_session(self.session_key)
            for _ in range(2):
                lease = POOL.checkout(
                    command=command,
                    extra_args=self.client.extra_args,
                    model=self.model,
                    effort=self.effort,
                    session_key=self.session_key,
                    messages=self.messages,
                    tools_sig=self.tools_sig,
                )
                if lease is None:
                    break
                worker = lease.worker
                prompt, branch = self._prompt_for(lease)
                stale = worker.stdout.drain()
                if stale:
                    logger.debug("Discarded %d stray line(s) from agy worker #%d", len(stale), worker.id)
                with self._lock:
                    self.lease, self.reader, self.stderr, self.quota = lease, worker.stdout, worker.stderr, worker.quota
                self.stderr_mark = worker.stderr.mark()
                worker.quota.mark_turn_start()
                _dump_prompt(prompt, branch=branch, model=self.model)
                logger.debug("agy turn on worker #%d (%s, %d chars)", worker.id, branch, len(prompt))
                try:
                    worker.send(prompt)
                except OSError:
                    # The worker died while idle: drop it and check out again (fresh spawn).
                    with self._lock:
                        self.lease = self.reader = None
                    POOL.discard(worker)
                    continue
                self.prompt_chars = len(prompt)
                return
        self._open_oneshot(command)

    def _open_oneshot(self, command: str) -> None:
        prompt, branch = self._prompt_for(None)
        cmd = [command, "--output-format", "stream-json", "--disable-slash-commands", "--model", self.model]
        cmd += effort_args(self.effort)
        # agy's own cap is a total-time limit; hangs are caught by the idle timeout in _consume.
        print_timeout = max(int(self.timeout), ONESHOT_PRINT_TIMEOUT_SECONDS)
        cmd += ["--print-timeout", f"{print_timeout}s", *mode_args(), *self.client.extra_args]
        sandbox = tempfile.mkdtemp(prefix="hermes_agy_")
        quota = QuotaLogWatch()
        try:
            proc = spawn(cmd, cwd=sandbox, env=build_child_env())
        except BaseException:
            remove_dir(sandbox)
            raise
        with self._lock:
            self.proc, self.sandbox, self.quota = proc, sandbox, quota
            self.reader = LineReader(proc.stdout, name="agy-oneshot")
            self.stderr = StderrTail(proc.stderr)
        _dump_prompt(prompt, branch=branch, model=self.model)
        self.prompt_chars = len(prompt)
        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt)
                proc.stdin.close()
        except OSError:
            pass  # the process already failed; its exit is reported by the event loop

    # ── running ──

    def chunks(self) -> Iterator[Any]:
        try:
            attempts = 0
            while True:
                self._open()
                try:
                    yield from self._consume()
                    return
                except _RetryTurn as retry:
                    attempts += 1
                    if attempts > 1 or self.yielded:
                        raise retry.error from None
                    logger.info("Retrying agy turn (%s): %s", retry.mode, retry.error)
                    self._reset_for_retry()
                    self._mode = retry.mode
        finally:
            self._release_resources(success=False)
            self.client._forget(self)

    def _stderr_text(self, wait: float = 0.0) -> str:
        if self.stderr is None:
            return ""
        return strip_ansi(self.stderr.text(self.stderr_mark, wait=wait)).strip()

    def _process_exit(self) -> int | None:
        proc = self.lease.worker.proc if self.lease is not None else self.proc
        if proc is None:
            return None
        with contextlib.suppress(Exception):
            return proc.poll()
        return None

    def _check_quota(self) -> None:
        detail = (self.quota.poll() if self.quota is not None else None) or find_quota_error(self._stderr_text())
        if detail:
            raise AgyError(f"Antigravity quota exhausted: {detail}", kind="quota")

    def _failure(self, detail: str, *, kind: str = "execution") -> AgyError:
        stderr = self._stderr_text(wait=0.5)
        quota = find_quota_error(stderr)
        if quota:
            return AgyError(f"Antigravity quota exhausted: {quota}", kind="quota")
        if _AUTH_RE.search(stderr) or _AUTH_RE.search(detail):
            return AgyError(
                "Antigravity CLI is not signed in. Run `agy` in a terminal and sign in with your Google account."
                + (f" ({stderr[-500:]})" if stderr else ""),
                kind="auth",
            )
        if stderr:
            detail = f"{detail}: {stderr[-1500:]}"
        return AgyError(f"Antigravity execution failed: {detail}", kind=kind)

    def _chunk(
        self,
        *,
        content: str | None = None,
        reasoning: str | None = None,
        tool_calls: list[Any] | None = None,
        finish_reason: str | None = None,
    ) -> Any:
        delta = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning,
            reasoning_content=reasoning,
            refusal=None,
        )
        return SimpleNamespace(
            id=self.response_id,
            object="chat.completion.chunk",
            created=self.created,
            model=self.model,
            choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason, logprobs=None)],
            usage=None,
        )

    def _consume(self) -> Iterator[Any]:
        parser = ToolCallStreamParser(enabled=bool(self.tools), known_tools=self.known_tools)
        content: list[str] = []
        reasoning: list[str] = []
        calls: list[Any] = []
        usage_raw: dict[str, Any] = {}
        status, error, conversation_id = "", "", ""
        saw_event = saw_result = got_output = False
        start = time.monotonic()
        deadline = start + self.timeout
        next_quota_check = start + 1.0
        reader = self.reader
        assert reader is not None

        def emit(events: list[tuple[str, Any]]) -> Iterator[Any]:
            for kind, value in events:
                if kind == "text":
                    content.append(value)
                    self.yielded = True
                    yield self._chunk(content=value)
                else:
                    value_delta = SimpleNamespace(
                        index=len(calls),
                        id=value.id,
                        type="function",
                        function=SimpleNamespace(name=value.function.name, arguments=value.function.arguments),
                    )
                    calls.append(value)
                    self.yielded = True
                    yield self._chunk(tool_calls=[value_delta])

        while True:
            if self._aborted:
                raise AgyError(f"Antigravity request cancelled ({self._aborted})", kind="cancelled")
            now = time.monotonic()
            if now >= deadline:
                raise self._failure(f"no output from agy for {self.timeout:.0f}s", kind="timeout")
            item = reader.get(timeout=min(0.5, deadline - now))
            if item is None:
                # agy retries 429s silently for minutes; watch for quota errors until output starts.
                if not got_output and time.monotonic() >= next_quota_check:
                    next_quota_check = time.monotonic() + 1.0
                    self._check_quota()
                continue
            if item is ABORTED:
                raise AgyError(f"Antigravity request cancelled ({self._aborted or 'worker stopped'})", kind="cancelled")
            if item is EOF:
                break
            line = item.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            saw_event = True
            deadline = time.monotonic() + self.timeout
            etype = event.get("event") or event.get("type")
            body = event.get(etype) if isinstance(etype, str) and isinstance(event.get(etype), dict) else {}
            conversation_id = conversation_id or str(event.get("conversation_id") or body.get("conversation_id") or "")

            got_output = got_output or etype in ("step_update", "result")
            if etype == "step_update":
                step_type = str(body.get("step_type") or "")
                if step_type == "tool" or step_type.startswith("tool_"):
                    tool = body.get("tool_name") or body.get("name") or "unknown"
                    logger.warning("agy attempted native tool %r; terminating it", tool)
                    self.abort("native tool")
                    raise AgyError(
                        f"Antigravity native tool execution intercepted ({tool}). "
                        "Native tools are blocked to preserve Hermes orchestration.",
                        kind="native_tool",
                    )
                if isinstance(body.get("usage"), dict):
                    usage_raw = body["usage"]
                thinking = body.get("thinking_delta") or body.get("reasoning_delta")
                text = body.get("text_delta")
                if text and ("think" in step_type or "reason" in step_type):
                    thinking, text = text, None
                if thinking:
                    reasoning.append(str(thinking))
                    self.yielded = True
                    yield self._chunk(reasoning=str(thinking))
                if text and step_type in _CONTENT_STEP_TYPES:
                    yield from emit(parser.feed(str(text)))
            elif etype == "result":
                saw_result = True
                status = str(body.get("status") or "").upper()
                error = str(body.get("error") or "")
                if isinstance(body.get("usage"), dict):
                    usage_raw = body["usage"]
                response = body.get("response")
                if response and not content and not calls and not parser.saw_calls:
                    yield from emit(parser.feed(str(response)))
                break
            elif etype == "error":
                error = str(body.get("message") or event.get("message") or event.get("error") or "unknown error")

        yield from emit(parser.finish())
        if parser.dropped_text.strip():
            logger.debug("Dropped %d chars agy wrote after its tool calls", len(parser.dropped_text))

        if self._aborted:
            raise AgyError(f"Antigravity request cancelled ({self._aborted})", kind="cancelled")
        exit_code = self._process_exit()
        if not saw_event and not self.yielded:
            self._retry_before_output(exit_code)
        if error or status in _FAILED_STATUSES:
            failure = self._failure(f"{error or 'turn failed'} (status='{status or 'unknown'}')")
            if failure.kind == "execution":
                failure = AgyError(
                    str(failure).replace("Antigravity execution failed", "Antigravity model error", 1), kind="model"
                )
            raise failure
        if not saw_result:
            # An old one-shot agy that ends without a result event still counts if it exited cleanly.
            if not (self.lease is None and exit_code == 0 and (content or calls)):
                state = "still running" if exit_code is None else f"exited with code {exit_code}"
                raise self._failure(f"no result event (process {state})")
        if not content and not calls and not reasoning:
            # agy reports SUCCESS with nothing in it when its trajectory hit the context limit.
            raise self._failure(
                f"empty result (status='{status or 'unknown'}')", kind="empty" if status == "SUCCESS" else "execution"
            )

        baseline = self.lease.worker.usage_baseline if self.lease is not None else None
        usage = self._usage(usage_raw, baseline, "".join(content))
        reply = Reply(content="".join(content), tool_names=tuple(c.function.name for c in calls))
        with self._lock:
            self._committed = True
        # Release before the final chunks: consumers often stop reading at finish_reason.
        self._release_resources(success=True, reply=reply, conversation_id=conversation_id)
        yield self._chunk(finish_reason="tool_calls" if calls else "stop")
        yield SimpleNamespace(
            id=self.response_id,
            object="chat.completion.chunk",
            created=self.created,
            model=self.model,
            choices=[],
            usage=usage,
        )

    def _retry_before_output(self, exit_code: int | None) -> None:
        """agy died before saying anything: recover transparently where a retry can help."""
        stderr = self._stderr_text(wait=0.5)
        if CLI_CAPS["effort_flag"] and self.effort and _EFFORT_REJECTED_RE.search(stderr):
            CLI_CAPS["effort_flag"] = False
            logger.warning("This agy does not accept --effort; selecting effort through the model id only.")
            raise _RetryTurn("fresh" if self.lease is not None else "oneshot", self._failure("--effort rejected"))
        lease = self.lease
        if lease is None:
            return
        if lease.fresh and not lease.worker.resumed and _INPUT_FORMAT_REJECTED_RE.search(stderr):
            POOL.mark_unsupported(stderr.splitlines()[-1][:200] if stderr else "--input-format rejected")
            raise _RetryTurn("oneshot", self._failure("persistent mode rejected by agy"))
        if lease.worker.resumed or not lease.fresh:
            # A resumed conversation that agy refused, or a reused worker that died while idle.
            raise _RetryTurn("fresh", self._failure(f"agy worker exited ({exit_code}) before answering"))

    def _usage(self, raw: dict[str, Any], baseline: dict[str, int] | None, output: str) -> Any:
        def value(name: str) -> int | None:
            if baseline is not None:
                return _counter_delta(raw, baseline, name)
            with contextlib.suppress(TypeError, ValueError):
                return int(raw[name]) if name in raw else None
            return None

        prompt = value("input_tokens")
        completion = value("output_tokens")
        total = value("total_tokens")
        cached = value("cache_read_tokens") or 0
        thinking = value("thinking_tokens") or 0
        if prompt is None:
            prompt = max(1, self.prompt_chars // 4)
        if completion is None:
            completion = max(0, len(output) // 4)
        if total is None:
            total = prompt + completion
        return SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=thinking),
        )


class AgyStream:
    """Iterator of chat.completion.chunk objects; also async-iterable, awaitable and closeable."""

    response = None  # Hermes reads ``stream.response`` when an SDK stream fails to decode

    def __init__(self, runner: _TurnRunner) -> None:
        self._runner = runner
        self._gen = runner.chunks()
        self._closed = False
        self._executor: ThreadPoolExecutor | None = None
        self.model = runner.model

    def __iter__(self) -> AgyStream:
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            return next(self._gen)
        except StopIteration:
            self._finish()
            raise
        except BaseException:
            self._finish()
            raise

    def _finish(self) -> None:
        self._closed = True
        self._runner.client._forget(self._runner)
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def close(self) -> None:
        if self._closed:
            return
        self._runner.abort("stream closed")
        with contextlib.suppress(ValueError):  # generator running on another thread: the abort ends it
            self._gen.close()
        self._finish()

    def __enter__(self) -> AgyStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __await__(self) -> Any:
        return _ready(self).__await__()

    def __aiter__(self) -> AgyStream:
        return self

    def _pull(self) -> tuple[bool, Any]:
        try:
            return False, self.__next__()
        except StopIteration:
            return True, None

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        if self._executor is None:
            # Private thread: one blocking read must not occupy the loop's shared default executor.
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agy-stream")
        try:
            done, chunk = await asyncio.get_running_loop().run_in_executor(self._executor, self._pull)
        except asyncio.CancelledError:
            self.close()
            raise
        if done:
            raise StopAsyncIteration
        return chunk

    async def aclose(self) -> None:
        self.close()


def collect_completion(stream: AgyStream) -> Completion:
    """Drain a stream into a non-streaming ChatCompletion."""
    content: list[str] = []
    reasoning: list[str] = []
    calls: list[Any] = []
    finish_reason = "stop"
    usage = None
    response_id, created, model = "", int(time.time()), stream.model
    for chunk in stream:
        response_id = response_id or getattr(chunk, "id", "")
        created = getattr(chunk, "created", created)
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            delta = choice.delta
            if delta.content:
                content.append(delta.content)
            if delta.reasoning_content:
                reasoning.append(delta.reasoning_content)
            for tc in delta.tool_calls or []:
                calls.append(
                    SimpleNamespace(
                        id=tc.id,
                        type="function",
                        function=SimpleNamespace(name=tc.function.name, arguments=tc.function.arguments),
                    )
                )
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    text = "".join(content).strip()
    message = SimpleNamespace(
        role="assistant",
        content=text or None,
        tool_calls=calls or None,
        refusal=None,
        reasoning_content="".join(reasoning) or None,
        reasoning="".join(reasoning) or None,
    )
    if usage is None:
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
    return Completion(
        id=response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        object="chat.completion",
        created=created,
        model=model,
        choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason, logprobs=None)],
        usage=usage,
    )


def _output_instruction(response_format: Any) -> str:
    if not isinstance(response_format, dict):
        return ""
    kind = response_format.get("type")
    if kind == "json_object":
        return "## Output format\nReply with a single valid JSON object only: no prose, no code fences."
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        suffix = f" matching this JSON Schema: {json.dumps(schema, ensure_ascii=False)}" if schema else ""
        return f"## Output format\nReply with a single valid JSON object{suffix}. No prose, no code fences."
    return ""


class AntigravityClient:
    """``client.chat.completions.create(...)`` over the agy CLI."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | tuple[str, ...] | None = None,
        timeout: Any = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or "antigravity"
        self.base_url = base_url or "acp://antigravity"
        self._command_hint = command
        self._command: str | None = None
        if args:
            self.extra_args = [str(a) for a in args]
        else:
            self.extra_args = shlex.split(getenv_any("AGY_ARGS", "ANTIGRAVITY_ARGS", default="") or "")
        if "--dangerously-skip-permissions" in self.extra_args:
            logger.warning("AGY extra args include --dangerously-skip-permissions: agy may execute tools itself.")
        self._default_timeout = timeout
        self._active: set[_TurnRunner] = set()
        self._lock = threading.RLock()  # see _TurnRunner._lock
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    def resolve_command(self) -> str:
        if self._command is None:
            found = which_agy(self._command_hint)
            if not found:
                raise AgyError(
                    "agy CLI not found. Install Google Antigravity CLI, put `agy` on PATH, or set AGY_CLI_PATH.",
                    kind="missing",
                )
            self._command = found
        return self._command

    # ── lifecycle ──

    def _forget(self, runner: _TurnRunner) -> None:
        with self._lock:
            self._active.discard(runner)

    def cancel(self) -> None:
        """Abort in-flight turns (Hermes calls this from its interrupt thread). Pooled idle workers stay."""
        with self._lock:
            runners = list(self._active)
        for runner in runners:
            runner.abort("cancelled")

    def close(self) -> None:
        self.is_closed = True
        self.cancel()

    def __enter__(self) -> AntigravityClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── requests ──

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[Any] | None = None,
        stream: bool = False,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        timeout: Any = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: Any = None,
        **_: Any,
    ) -> Any:
        if self.is_closed:
            raise RuntimeError("AntigravityClient is closed.")
        self.resolve_command()  # fail at create() with a clear message, not on the first chunk
        extra = extra_body if isinstance(extra_body, dict) else {}
        effort = reasoning_effort or extra.get("reasoning_effort")
        if effort is None and isinstance(extra.get("reasoning"), dict):
            effort = extra["reasoning"].get("effort")
        resolved, agy_effort = resolve_model(model or DEFAULT_MODEL, effort)
        msgs = [d for d in (_as_dict(m) for m in messages or []) if d is not None]
        session = extra.get(SESSION_KEY_FIELD)
        if tool_choice == "none":
            tools = None
        runner = _TurnRunner(
            self,
            model=resolved,
            effort=agy_effort,
            messages=msgs,
            tools=list(tools) if tools else None,
            tool_choice=tool_choice,
            timeout=coerce_timeout(timeout if timeout is not None else self._default_timeout),
            session_key=str(session) if session else None,
            output_instruction=_output_instruction(response_format),
        )
        with self._lock:
            self._active.add(runner)
        result = AgyStream(runner)
        if stream:
            return result
        return collect_completion(result)


def classify_error(error: Exception, message: str = "") -> dict[str, Any] | None:
    """Hermes failover hints for agy failures (``ProviderProfile.classify_api_error``)."""
    kind = getattr(error, "kind", None)
    text = f"{error} {message}".lower()
    if kind == "quota" or any(p in text for p in ("credits balance is too low", "billing cap")):
        if any(p in text for p in ("credits balance is too low", "billing cap")):
            return {"reason": "billing", "retryable": False, "should_fallback": True}
        return {"reason": "rate_limit", "retryable": True, "should_fallback": True}
    if any(p in text for p in ("resource_exhausted", "quota exceeded", "quota reached", "code 429")):
        return {"reason": "rate_limit", "retryable": True, "should_fallback": True}
    if any(p in text for p in ("verify your account", "terms of service")):
        return {"reason": "auth_permanent", "retryable": False, "should_fallback": False}
    if kind == "auth" or "not signed in" in text:
        return {"reason": "auth", "retryable": False, "should_fallback": True}
    if kind == "missing":
        return {"reason": "auth_permanent", "retryable": False, "should_fallback": True}
    if kind == "empty" or any(
        p in text
        for p in (
            "subscriber fell behind updates",
            "stalled for 5s",
            "max_trajectory_tokens",
            "max trajectory tokens",
            "context length exceeded",
            "prompt is too long",
        )
    ):
        return {"reason": "context_overflow", "retryable": True, "should_compress": True}
    if any(p in text for p in ("unknown model", "model not found", "invalid model", "unsupported model")):
        return {"reason": "model_not_found", "retryable": False, "should_fallback": True}
    if kind == "timeout":
        return {"reason": "timeout", "retryable": True}
    return None


__all__ = [
    "AgyError",
    "AgyStream",
    "AntigravityClient",
    "Completion",
    "POOL",
    "classify_error",
    "coerce_timeout",
    "collect_completion",
]

"""Persistent agy workers, pooled per Hermes session and kept alive across conversation turns.

One worker is one long-lived ``agy --input-format stream-json`` process. Each turn writes a single
``{"event": "user", ...}`` line and reads stream-json events until the turn's ``result``. Because agy
keeps the conversation (its trajectory) in memory, a continuing Hermes conversation only sends the
messages added since the previous turn instead of re-sending the whole transcript.

Hermes builds and closes request-scoped clients freely, so workers belong to this process-wide pool,
not to a client. They are matched to requests by Hermes session id (passed through ``extra_body``)
or, for requests without one, by the conversation history they already hold. Idle workers are reaped
after ``AGY_WORKER_IDLE_SECONDS``; a reaped session can later resume its agy conversation with
``--conversation <id>`` and still send only the delta.
"""

from __future__ import annotations

import atexit
import contextlib
import itertools
import json
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    from .process import (
        LineReader,
        QuotaLogWatch,
        StderrTail,
        build_child_env,
        env_flag,
        env_float,
        kill_tree,
        remove_dir,
        spawn,
    )
    from .prompt import Reply, continuation_delta, history_keys
except ImportError:
    from process import (
        LineReader,
        QuotaLogWatch,
        StderrTail,
        build_child_env,
        env_flag,
        env_float,
        kill_tree,
        remove_dir,
        spawn,
    )
    from prompt import Reply, continuation_delta, history_keys

logger = logging.getLogger(__name__)

DEFAULT_IDLE_SECONDS = 900.0
DEFAULT_MAX_WORKERS = 4
DORMANT_LIMIT = 64
DORMANT_TTL_SECONDS = 24 * 3600.0

_ids = itertools.count(1)


def idle_seconds() -> float:
    return env_float("AGY_WORKER_IDLE_SECONDS", "ANTIGRAVITY_WORKER_IDLE_SECONDS", default=DEFAULT_IDLE_SECONDS)


def max_workers() -> int:
    return int(env_float("AGY_MAX_WORKERS", default=float(DEFAULT_MAX_WORKERS)))


def mode_args() -> list[str]:
    mode = os.environ.get("AGY_MODE", "").strip()
    return ["--mode", mode] if mode else []


# CLI features detected at runtime. agy 1.2+ wants `--effort` for effort variants; an older agy
# that rejects the flag gets it dropped for the rest of the process.
CLI_CAPS = {"effort_flag": True}


def effort_args(effort: str | None) -> list[str]:
    return ["--effort", effort] if effort and CLI_CAPS["effort_flag"] else []


def worker_command(
    command: str, model: str, effort: str | None, extra_args: Sequence[str], conversation_id: str | None
) -> list[str]:
    cmd = [
        command,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--disable-slash-commands",
        "--model",
        model,
    ]
    cmd += effort_args(effort)
    if conversation_id:
        cmd += ["--conversation", conversation_id]
    return cmd + mode_args() + list(extra_args)


class AgyWorker:
    """One persistent agy process plus the conversation state it holds."""

    def __init__(
        self,
        *,
        command: str,
        model: str,
        effort: str | None,
        extra_args: Sequence[str] = (),
        session_key: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.id = next(_ids)
        self.command = command
        self.model = model
        self.effort = effort
        self.extra_args = tuple(extra_args)
        self.session_key = session_key
        self.conversation_id = conversation_id
        self.resumed = conversation_id is not None
        self.history: tuple[bytes, ...] = ()  # one digest per message agy has seen
        self.reply: Reply | None = None
        self.tools_sig: str | None = None
        self.usage_baseline: dict[str, int] = {}
        self.busy = False
        self.turns = 0
        self.last_used = time.monotonic()
        self._terminated = False
        self.sandbox = tempfile.mkdtemp(prefix="hermes_agy_")
        self.quota = QuotaLogWatch()
        try:
            self.proc = spawn(
                worker_command(command, model, effort, self.extra_args, conversation_id),
                cwd=self.sandbox,
                env=build_child_env(),
            )
        except BaseException:
            remove_dir(self.sandbox)
            raise
        self.stdout = LineReader(self.proc.stdout, name=f"agy-worker-{self.id}")
        self.stderr = StderrTail(self.proc.stderr)
        logger.info(
            "Started agy worker #%d pid=%s model=%s effort=%s%s",
            self.id,
            getattr(self.proc, "pid", "?"),
            model,
            effort,
            f" resuming conversation {conversation_id}" if conversation_id else "",
        )

    @property
    def pool_key(self) -> tuple[str, tuple[str, ...]]:
        return (self.command, self.extra_args)

    def alive(self) -> bool:
        return not self._terminated and self.proc.poll() is None

    def send(self, prompt: str) -> None:
        """Start a turn. Raises OSError (BrokenPipeError) when the process is gone."""
        stdin = self.proc.stdin
        if stdin is None:
            raise BrokenPipeError("agy worker has no stdin")
        stdin.write(json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n")
        stdin.flush()

    def terminate(self, *, background: bool = False, grace: float = 0.0) -> None:
        if self._terminated:
            return
        self._terminated = True
        self.stdout.abort()
        logger.info(
            "Stopping agy worker #%d pid=%s after %d turn(s)", self.id, getattr(self.proc, "pid", "?"), self.turns
        )

        def _stop() -> None:
            kill_tree(self.proc, grace=grace)
            remove_dir(self.sandbox)

        if background:
            threading.Thread(target=_stop, name=f"agy-stop-{self.id}", daemon=True).start()
        else:
            _stop()


@dataclass
class _Dormant:
    conversation_id: str
    model: str
    effort: str | None
    history: tuple[bytes, ...]
    reply: Reply | None
    tools_sig: str | None
    stored_at: float = field(default_factory=time.monotonic)


@dataclass
class Lease:
    """A worker checked out for one turn."""

    worker: AgyWorker
    delta: list[dict[str, Any]] | None  # None: the worker needs the full prompt
    skip_reply: bool = False  # delta[0] is the worker's own previous reply
    tools_changed: bool = False
    fresh: bool = False  # spawned for this request (so a pre-output failure can be retried)


class WorkerPool:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: list[AgyWorker] = []
        self._dormant: OrderedDict[str, _Dormant] = OrderedDict()
        self._wake = threading.Event()
        self._reaper: threading.Thread | None = None
        self._closed = False
        self.unsupported_reason: str | None = None

    # ── configuration ──

    def enabled(self) -> bool:
        if self._closed or self.unsupported_reason is not None:
            return False
        if env_flag("AGY_DISABLE_PERSISTENT", "AGY_ONESHOT"):
            return False
        return max_workers() > 0

    def mark_unsupported(self, reason: str) -> None:
        if self.unsupported_reason is None:
            logger.warning("agy persistent workers unavailable (%s); using one process per turn.", reason)
        self.unsupported_reason = reason

    # ── checkout / return ──

    def checkout(
        self,
        *,
        command: str,
        extra_args: Sequence[str],
        model: str,
        effort: str | None,
        session_key: str | None,
        messages: Sequence[dict[str, Any]],
        tools_sig: str | None,
    ) -> Lease | None:
        """A worker for this request, or None to run it as a one-shot process."""
        if not self.enabled():
            return None
        key = (command, tuple(extra_args))
        with self._lock:
            self._ensure_reaper()
            self._drop_dead()
            if session_key:
                return self._checkout_session(key, model, effort, session_key, messages, tools_sig)
            return self._checkout_anonymous(key, model, effort, messages, tools_sig)

    def _checkout_session(
        self,
        key: Any,
        model: str,
        effort: str | None,
        session_key: str,
        messages: Sequence[dict[str, Any]],
        tools_sig: str | None,
    ) -> Lease | None:
        current = next((w for w in self._workers if w.session_key == session_key), None)
        if current is not None and current.busy:
            return None  # a second concurrent request in the same session runs one-shot
        if current is not None and current.pool_key != key:
            self._retire(current, keep_conversation=False)
            current = None
        if current is not None:
            delta = continuation_delta(current.history, messages)
            if delta is not None and (current.model, current.effort) == (model, effort):
                return self._lease(current, delta, tools_sig)
            if delta is not None and current.conversation_id:
                # Model or effort switch mid-conversation: same agy conversation, new process.
                state = (current.history, current.reply, current.tools_sig)
                cid = current.conversation_id
                self._workers.remove(current)
                # Synchronous: the old process must have let go of the conversation before the new one opens it.
                current.terminate(background=False, grace=2.0)
                worker = self._spawn(key, model, effort, session_key, conversation_id=cid)
                if worker is None:
                    return None
                worker.history, worker.reply, worker.tools_sig = state
                return self._lease(worker, delta, tools_sig, fresh=True)
            # History diverged (compression, retry, edit, /new): start over.
            self._retire(current, keep_conversation=False)
            self._dormant.pop(session_key, None)
        else:
            dormant = self._dormant.pop(session_key, None)
            if dormant is not None and time.monotonic() - dormant.stored_at < DORMANT_TTL_SECONDS:
                delta = continuation_delta(dormant.history, messages)
                if delta is not None:
                    worker = self._spawn(key, model, effort, session_key, conversation_id=dormant.conversation_id)
                    if worker is None:
                        return None
                    worker.history, worker.reply, worker.tools_sig = dormant.history, dormant.reply, dormant.tools_sig
                    return self._lease(worker, delta, tools_sig, fresh=True)
        worker = self._spawn(key, model, effort, session_key)
        return None if worker is None else self._lease(worker, None, tools_sig, fresh=True)

    def _checkout_anonymous(
        self,
        key: Any,
        model: str,
        effort: str | None,
        messages: Sequence[dict[str, Any]],
        tools_sig: str | None,
    ) -> Lease | None:
        best: AgyWorker | None = None
        best_delta: list[dict[str, Any]] | None = None
        for worker in self._workers:
            if (
                worker.busy
                or worker.session_key is not None
                or worker.pool_key != key
                or (worker.model, worker.effort) != (model, effort)
            ):
                continue
            delta = continuation_delta(worker.history, messages)
            if delta is not None and (best is None or len(worker.history) > len(best.history)):
                best, best_delta = worker, delta
        if best is not None:
            return self._lease(best, best_delta, tools_sig)
        worker = self._spawn(key, model, effort, None)
        return None if worker is None else self._lease(worker, None, tools_sig, fresh=True)

    def _lease(
        self, worker: AgyWorker, delta: list[dict[str, Any]] | None, tools_sig: str | None, *, fresh: bool = False
    ) -> Lease:
        worker.busy = True
        worker.last_used = time.monotonic()
        skip = bool(delta) and worker.reply is not None and worker.reply.matches(delta[0])  # type: ignore[index]
        changed = delta is not None and worker.tools_sig is not None and worker.tools_sig != tools_sig
        return Lease(worker=worker, delta=delta, skip_reply=skip, tools_changed=changed, fresh=fresh)

    def release(
        self,
        lease: Lease,
        *,
        messages: Sequence[dict[str, Any]],
        reply: Reply,
        tools_sig: str | None,
        conversation_id: str | None,
    ) -> None:
        """Return a worker after a successful turn, recording what its trajectory now holds."""
        worker = lease.worker
        with self._lock:
            worker.history = history_keys(messages)
            worker.reply = reply
            worker.tools_sig = tools_sig
            if conversation_id:
                worker.conversation_id = conversation_id
            worker.turns += 1
            worker.last_used = time.monotonic()
            worker.busy = False
            if not worker.alive():
                self._retire(worker, keep_conversation=True)
        self._wake.set()

    def forget_session(self, session_key: str) -> None:
        """Start the session over: drop its idle worker and any resumable conversation."""
        with self._lock:
            self._dormant.pop(session_key, None)
            for worker in [w for w in self._workers if w.session_key == session_key and not w.busy]:
                self._retire(worker, keep_conversation=False)

    def discard(self, worker: AgyWorker, *, keep_conversation: bool = False) -> None:
        """Drop a worker whose turn failed or was cancelled: its trajectory can no longer be trusted."""
        with self._lock:
            worker.busy = False
            self._retire(worker, keep_conversation=keep_conversation)
            if not keep_conversation and worker.session_key:
                self._dormant.pop(worker.session_key, None)

    # ── internals (lock held) ──

    def _spawn(
        self, key: Any, model: str, effort: str | None, session_key: str | None, *, conversation_id: str | None = None
    ) -> AgyWorker | None:
        limit = max_workers()
        live = [w for w in self._workers if w.alive()]
        while len(live) >= limit:
            idle = [w for w in live if not w.busy]
            if not idle:
                return None
            victim = min(idle, key=lambda w: (w.session_key is not None, w.last_used))
            self._retire(victim, keep_conversation=True)
            live.remove(victim)
        command, extra_args = key
        worker = AgyWorker(
            command=command,
            model=model,
            effort=effort,
            extra_args=extra_args,
            session_key=session_key,
            conversation_id=conversation_id,
        )
        self._workers.append(worker)
        return worker

    def _retire(self, worker: AgyWorker, *, keep_conversation: bool) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        if keep_conversation and worker.session_key and worker.conversation_id and worker.history:
            self._dormant[worker.session_key] = _Dormant(
                conversation_id=worker.conversation_id,
                model=worker.model,
                effort=worker.effort,
                history=worker.history,
                reply=worker.reply,
                tools_sig=worker.tools_sig,
            )
            self._dormant.move_to_end(worker.session_key)
            while len(self._dormant) > DORMANT_LIMIT:
                self._dormant.popitem(last=False)
        worker.terminate(background=True, grace=2.0 if keep_conversation else 0.0)

    def _drop_dead(self) -> None:
        for worker in [w for w in self._workers if not w.busy and not w.alive()]:
            self._retire(worker, keep_conversation=True)

    # ── idle reaping ──

    def _ensure_reaper(self) -> None:
        if self._reaper is None or not self._reaper.is_alive():
            self._reaper = threading.Thread(target=self._reap_loop, name="agy-worker-reaper", daemon=True)
            self._reaper.start()

    def _reap_loop(self) -> None:
        while not self._closed:
            limit = idle_seconds()
            self._wake.wait(timeout=min(30.0, max(0.5, limit / 4)) if limit > 0 else 30.0)
            self._wake.clear()
            self.reap_idle()

    def reap_idle(self, now: float | None = None) -> int:
        limit = idle_seconds()
        now = time.monotonic() if now is None else now
        with self._lock:
            victims = [
                w for w in self._workers if not w.busy and (not w.alive() or (limit > 0 and now - w.last_used >= limit))
            ]
            for worker in victims:
                logger.info("Reaping idle agy worker #%d (session=%s)", worker.id, worker.session_key)
                self._retire(worker, keep_conversation=True)
        return len(victims)

    # ── introspection / shutdown ──

    def workers(self) -> list[AgyWorker]:
        with self._lock:
            return list(self._workers)

    def dormant_sessions(self) -> list[str]:
        with self._lock:
            return list(self._dormant)

    def shutdown(self) -> None:
        """Stop every worker (interpreter exit). Synchronous so nothing outlives Hermes."""
        with self._lock:
            workers, self._workers = list(self._workers), []
            self._dormant.clear()
        for worker in workers:
            with contextlib.suppress(Exception):
                worker.terminate(background=False)
        self._wake.set()

    def reset(self) -> None:
        """Test helper: stop everything and forget detected capabilities."""
        self.shutdown()
        self.unsupported_reason = None
        CLI_CAPS["effort_flag"] = True


POOL = WorkerPool()
atexit.register(POOL.shutdown)

__all__ = [
    "CLI_CAPS",
    "POOL",
    "AgyWorker",
    "Lease",
    "WorkerPool",
    "effort_args",
    "idle_seconds",
    "max_workers",
    "mode_args",
    "worker_command",
]

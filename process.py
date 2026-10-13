"""Process plumbing for the agy CLI: discovery, environment, spawning, I/O and teardown."""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

# Older plugin releases and the soyelmismo plugin used these names; all are honoured.
COMMAND_ENV_VARS = ("AGY_CLI_PATH", "HERMES_AGY_COMMAND", "ANTIGRAVITY_COMMAND", "ANTIGRAVITY_CLI_PATH")
_TOKEN_FILENAMES = ("jetski-standalone-oauth-token", "antigravity-oauth-token")
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def getenv_any(*names: str, default: str | None = None) -> str | None:
    """First non-blank value among ``names``."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def env_flag(*names: str, default: bool = False) -> bool:
    value = getenv_any(*names)
    if value is None:
        return default
    return value.lower() in _TRUTHY


def env_float(*names: str, default: float) -> float:
    value = getenv_any(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ── Discovery ───────────────────────────────────────────────────────────────


def _is_executable(path: str | Path) -> bool:
    try:
        p = Path(path)
        return p.is_file() and (os.name == "nt" or os.access(p, os.X_OK))
    except OSError:
        return False


def which_agy(preferred: str | None = None) -> str | None:
    """Locate the agy binary, or None when it is not installed."""
    for name in COMMAND_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value and _is_executable(value):
            return value
    if preferred:
        if _is_executable(preferred):
            return preferred
        found = shutil.which(preferred)
        if found:
            return found
    found = shutil.which("agy") or shutil.which("agy.exe")
    if found:
        return found
    binary = "agy.exe" if os.name == "nt" else "agy"
    candidates = [
        Path.home() / ".gemini" / "antigravity-cli" / "bin" / binary,
        Path.home() / ".local" / "bin" / binary,
        Path("/usr/local/bin") / binary,
    ]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates += [
            Path(local_appdata) / "agy" / "bin" / binary,
            Path(local_appdata) / "Programs" / "agy" / binary,
            Path(local_appdata) / "Microsoft" / "WinGet" / "Links" / binary,
        ]
    for candidate in candidates:
        if _is_executable(candidate):
            return str(candidate)
    return None


def find_agy(preferred: str | None = None) -> str:
    found = which_agy(preferred)
    if not found:
        raise FileNotFoundError(
            "agy CLI not found. Install Google Antigravity CLI, put it on PATH, or set AGY_CLI_PATH."
        )
    return found


def agy_config_dir() -> Path:
    override = getenv_any("AGY_CONFIG_DIR", "ANTIGRAVITY_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".gemini" / "antigravity-cli"


def detect_login() -> bool | None:
    """Whether agy holds a session: True/False when determinable, None when unknown.

    Only presence is checked (file size, credential-store metadata). Token contents are never read.
    """
    config_dir = agy_config_dir()
    for name in _TOKEN_FILENAMES:
        with contextlib.suppress(OSError):
            token = config_dir / name
            if token.is_file() and token.stat().st_size > 10:
                return True
    try:
        if os.name == "nt":
            # cmdkey prints target metadata only, never the secret.
            out = subprocess.run(
                ["cmdkey", "/list:gemini:antigravity"],
                capture_output=True,
                timeout=5,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return out.returncode == 0 and b"gemini:antigravity" in (out.stdout or b"").lower()
        if sys.platform == "darwin":
            # Without -g/-w only item attributes are printed; no keychain prompt, no secret.
            out = subprocess.run(
                ["/usr/bin/security", "find-generic-password", "-s", "gemini", "-a", "antigravity"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None
    # Linux keyring sessions cannot be probed without loading the secret; report unknown.
    return None


# ── Child environment ───────────────────────────────────────────────────────

# Hermes runs with gateway/dashboard secrets in its environment. agy is a closed binary that needs
# none of them, so credential-looking names are removed by default (a name heuristic; use strict
# mode for a guarantee).
_SECRET_NAME_RE = re.compile(
    r"(^|_)(TOKENS?|SECRETS?|PASSWORDS?|PASSWD|PASSPHRASE|API_?KEYS?|ACCESS_?KEY|PRIVATE_?KEY|"
    r"CREDENTIALS?|AUTH|WEBHOOKS?|DSN|COOKIES?|SESSION_?KEY)($|_)"
    r"|(^|_)(DATABASE|DB|REDIS|MONGO\w*|POSTGRES\w*|MYSQL|AMQP|BROKER|CACHE)_(URL|URI)($|_)",
    re.IGNORECASE,
)
_SECRET_NAME_ALLOW = frozenset({"SSH_AUTH_SOCK", "GPG_AGENT_INFO"})
_STRICT_BASELINE = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "USERNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TZ",
        "TERM",
        "TMPDIR",
        "TEMP",
        "TMP",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "SSH_AUTH_SOCK",
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "AGY_CLI_DISABLE_AUTO_UPDATE",
        "AGY_CLI_MODEL_API_MAX_RETRIES",
    }
)
_LOOPBACK = ("localhost", "127.0.0.1", "::1")


def _norm_name(name: str) -> str:
    return name.upper() if os.name == "nt" else name


def _name_set(*env_vars: str) -> set[str]:
    raw = ",".join(os.environ.get(v, "") for v in env_vars)
    return {_norm_name(n.strip()) for n in raw.split(",") if n.strip()}


def build_child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for agy: the parent's minus credential-looking variables.

    ``AGY_ENV_PASSTHROUGH`` keeps named variables; ``AGY_ENV_STRICT=1`` passes only a runtime
    baseline plus ``AGY_ENV_ALLOWLIST``; ``AGY_PROXY`` routes agy through an explicit proxy.
    """
    source = dict(os.environ if base is None else base)
    keep = _name_set("AGY_ENV_PASSTHROUGH", "ANTIGRAVITY_ENV_PASSTHROUGH")
    if env_flag("AGY_ENV_STRICT", "ANTIGRAVITY_ENV_STRICT"):
        allowed = {_norm_name(n) for n in _STRICT_BASELINE}
        allowed |= _name_set("AGY_ENV_ALLOWLIST", "ANTIGRAVITY_ENV_ALLOWLIST") | keep
        env = {
            k: v
            for k, v in source.items()
            if _norm_name(k) in allowed or _norm_name(k).upper() in allowed or k.upper().startswith("LC_")
        }
    else:
        safe = {_norm_name(n) for n in _SECRET_NAME_ALLOW}
        env = {
            k: v
            for k, v in source.items()
            if _norm_name(k) in keep or _norm_name(k) in safe or not _SECRET_NAME_RE.search(k)
        }
    proxy = getenv_any("AGY_PROXY", "ANTIGRAVITY_PROXY")
    if proxy:
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            env[key] = proxy
        entries: dict[str, str] = {}
        raw = ",".join(filter(None, (env.get("NO_PROXY"), env.get("no_proxy"))))
        for entry in (*raw.split(","), *_LOOPBACK):
            if entry.strip():
                entries.setdefault(entry.strip().casefold(), entry.strip())
        env["NO_PROXY"] = env["no_proxy"] = ",".join(entries.values())
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


# ── Spawning and teardown ───────────────────────────────────────────────────


def popen_kwargs() -> dict[str, Any]:
    """Own process group (so the whole tree can be killed) and no console window on Windows."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": flags}
    return {"start_new_session": True}


def spawn(cmd: list[str], *, cwd: str, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=cwd,
        env=env,
        **popen_kwargs(),
    )


def kill_tree(proc: subprocess.Popen, *, grace: float = 0.0) -> None:
    """Terminate ``proc`` and every descendant (agy runs a language-server child). Never raises."""
    pid = getattr(proc, "pid", None)
    if os.name == "nt":
        if proc.poll() is None and isinstance(pid, int):
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        with contextlib.suppress(Exception):
            if proc.poll() is None:
                proc.kill()
    else:
        group_killed = False
        if isinstance(pid, int) and pid > 0:
            # The group id equals the child's pid (start_new_session); signal it even if the
            # leader already exited, so an orphaned language server is not left behind.
            if grace > 0 and proc.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(pid, signal.SIGTERM)
                with contextlib.suppress(Exception):
                    proc.wait(timeout=grace)
            try:
                os.killpg(pid, signal.SIGKILL)  # windows-footgun: ok — POSIX branch only
                group_killed = True
            except (ProcessLookupError, PermissionError, OSError, AttributeError):
                pass
        if not group_killed:
            with contextlib.suppress(Exception):
                if proc.poll() is None:
                    proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=2)
    for stream in (getattr(proc, "stdin", None), getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
        with contextlib.suppress(Exception):
            if stream is not None and hasattr(stream, "close"):
                stream.close()


def kill_tree_in_background(proc: subprocess.Popen, *, grace: float = 0.0) -> None:
    """kill_tree without blocking the caller (safe from event-loop threads and interrupt handlers)."""
    threading.Thread(target=kill_tree, args=(proc,), kwargs={"grace": grace}, name="agy-kill", daemon=True).start()


def remove_dir(path: str | None) -> None:
    """Remove a scratch directory, retrying briefly for Windows handle release. Never raises."""
    if not path:
        return
    for delay in (0.0, 0.1, 0.25):
        if delay:
            time.sleep(delay)
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
    logger.debug("Could not fully remove agy scratch dir %s", path)


# ── Pipe readers ────────────────────────────────────────────────────────────

EOF = object()
ABORTED = object()


class LineReader:
    """Reads a text pipe on a daemon thread so the consumer can wait with a timeout."""

    def __init__(self, stream: IO[str] | None, name: str = "agy-stdout") -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stream = stream
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        stream = self._stream
        try:
            if stream is not None:
                for line in iter(stream.readline, ""):
                    self._queue.put(line)
        except Exception:
            pass
        finally:
            self._queue.put(EOF)

    def get(self, timeout: float) -> Any:
        """A line, ``EOF``, ``ABORTED``, or None when nothing arrived within ``timeout``."""
        try:
            return self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def drain(self) -> list[str]:
        """Discard and return whatever is buffered (lines left over from a previous turn)."""
        leftovers: list[str] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return leftovers
            if item is EOF or item is ABORTED:
                self._queue.put(item)
                return leftovers
            leftovers.append(item)

    def abort(self) -> None:
        self._queue.put(ABORTED)


class StderrTail:
    """Drains stderr continuously (an unread pipe blocks agy once full) and keeps a bounded tail."""

    def __init__(self, stream: IO[str] | None, max_chars: int = 32 * 1024) -> None:
        self._stream = stream
        self._max = max_chars
        self._chunks: deque[str] = deque()
        self._size = 0
        self._total = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if stream is not None and callable(getattr(stream, "read", None)):
            self._thread = threading.Thread(target=self._run, name="agy-stderr", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        stream = self._stream
        reader = getattr(stream, "readline", None) or getattr(stream, "read", None)
        while True:
            try:
                chunk = reader()  # type: ignore[misc]
            except Exception:
                return
            if not chunk or not isinstance(chunk, str):
                return
            with self._lock:
                self._chunks.append(chunk)
                self._size += len(chunk)
                self._total += len(chunk)
                while self._chunks and self._size - len(self._chunks[0]) >= self._max:
                    self._size -= len(self._chunks.popleft())

    def mark(self) -> int:
        with self._lock:
            return self._total

    def text(self, since: int = 0, *, wait: float = 0.0) -> str:
        if wait and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=wait)
        with self._lock:
            data = "".join(self._chunks)
            available_from = self._total - len(data)
        start = max(0, since - available_from)
        return data[start:][-self._max :]


# ── Quota detection ─────────────────────────────────────────────────────────

_QUOTA_LINE_RE = re.compile(
    r"RESOURCE_EXHAUSTED|quota (?:exceeded|reached)|code 429|credits balance is too low|"
    r"daily or billing cap",
    re.IGNORECASE,
)


def find_quota_error(text: str) -> str | None:
    """The last quota/credit-exhaustion line in ``text``, trimmed of retry chatter."""
    for line in reversed((text or "").splitlines()):
        if not _QUOTA_LINE_RE.search(line):
            continue
        detail = line.strip()
        start = detail.find("RESOURCE_EXHAUSTED")
        if start != -1:
            detail = detail[start:]
        retrying = detail.rfind("), retrying in")
        if retrying != -1:
            detail = detail[:retrying]
        return detail.strip()[:500]
    return None


class QuotaLogWatch:
    """Watches the agy log file that belongs to one process for quota exhaustion.

    agy retries 429s internally for minutes without writing to stdout. Its log directory is shared
    with every other agy process the user runs, so the file is attributed only when exactly one
    new log appeared after this process was spawned; otherwise the watch stays silent.
    """

    def __init__(self) -> None:
        self._dir = agy_config_dir() / "log"
        self._before = self._snapshot()
        self._spawned_at = time.time()
        self._path: Path | None = None
        self._ambiguous = False
        self._offset = 0

    def _snapshot(self) -> set[str]:
        try:
            return {p.name for p in self._dir.glob("cli-*.log")}
        except OSError:
            return set()

    def _attach(self) -> Path | None:
        if self._path is not None or self._ambiguous:
            return self._path
        fresh = [n for n in self._snapshot() - self._before]
        if len(fresh) == 1:
            self._path = self._dir / fresh[0]
        elif len(fresh) > 1:
            self._ambiguous = True
        return self._path

    def mark_turn_start(self) -> None:
        path = self._attach()
        if path is not None:
            with contextlib.suppress(OSError):
                self._offset = path.stat().st_size

    def poll(self) -> str | None:
        path = self._attach()
        if path is None:
            return None
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                start = max(self._offset, size - 64 * 1024)
                fh.seek(start)
                return find_quota_error(fh.read())
        except OSError:
            return None

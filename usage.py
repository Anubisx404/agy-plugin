"""Subscription quota for Hermes' /usage and status bar, read from `agy -p /usage --output-format json`."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
except ImportError:  # running without Hermes (tests)

    @dataclass(frozen=True)
    class AccountUsageWindow:  # type: ignore[no-redef]
        label: str
        used_percent: Optional[float] = None
        reset_at: Optional[datetime] = None
        detail: Optional[str] = None
        scope: str = "account"

    @dataclass(frozen=True)
    class AccountUsageSnapshot:  # type: ignore[no-redef]
        provider: str
        source: str
        fetched_at: datetime
        title: str = "Account limits"
        plan: Optional[str] = None
        windows: tuple[Any, ...] = ()
        details: tuple[str, ...] = ()
        unavailable_reason: Optional[str] = None
        raw: Optional[dict] = None


try:
    from .process import build_child_env, popen_kwargs, remove_dir, which_agy
except ImportError:
    from process import build_child_env, popen_kwargs, remove_dir, which_agy

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60.0
PROVIDER = "antigravity"

_cache_lock = threading.Lock()
_refresh_lock = threading.Lock()
_cache: tuple[float, AccountUsageSnapshot] | None = None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _find_groups(node: Any, depth: int = 0) -> list[dict[str, Any]] | None:
    """The first ``groups`` list in agy's (envelope-agnostic) usage payload."""
    if depth > 12:
        return None
    if isinstance(node, dict):
        groups = node.get("groups")
        if isinstance(groups, list) and any(isinstance(g, dict) for g in groups):
            return [g for g in groups if isinstance(g, dict)]
        for value in node.values():
            found = _find_groups(value, depth + 1)
            if found is not None:
                return found
    return None


def parse_usage_output(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """(groups, raw payload) from `agy /usage` JSON or stream-json output."""
    candidates: list[Any] = []
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        candidates.append(json.loads(stripped))
    except ValueError:
        for line in stripped.splitlines():
            try:
                candidates.append(json.loads(line))
            except ValueError:
                continue
    for payload in candidates:
        groups = _find_groups(payload)
        if groups is not None:
            return groups, payload if isinstance(payload, dict) else {"payload": payload}
    return None


def _label(group: str, window: str) -> str:
    name = group.strip() or "Quota"
    lowered = name.lower()
    if "gemini" in lowered:
        name = "Gemini"
    elif "claude" in lowered or "gpt" in lowered:
        name = "Claude/GPT"
    w = window.strip()
    return f"{name} ({'Weekly' if w.lower() == 'weekly' else w or 'window'})"


def _window(label: str, used: float | None, reset_at: datetime | None, detail: str | None) -> Any:
    try:  # quotas are per model group, never the whole login
        return AccountUsageWindow(label=label, used_percent=used, reset_at=reset_at, detail=detail, scope="model")
    except TypeError:  # older Hermes without `scope`
        return AccountUsageWindow(label=label, used_percent=used, reset_at=reset_at, detail=detail)


def build_snapshot(groups: list[dict[str, Any]], raw: dict[str, Any]) -> AccountUsageSnapshot:
    windows = []
    for group in groups:
        group_name = str(group.get("name") or "")
        for bucket in group.get("buckets") or []:
            if not isinstance(bucket, dict):
                continue
            try:
                remaining = max(0.0, min(1.0, float(bucket.get("remaining_fraction"))))
                used: float | None = round((1.0 - remaining) * 100.0, 1)
            except (TypeError, ValueError):
                used = None
            windows.append(
                _window(
                    _label(group_name, str(bucket.get("window") or "")),
                    used,
                    _parse_time(bucket.get("reset_time")),
                    str(bucket.get("name") or "") or None,
                )
            )
    return AccountUsageSnapshot(
        provider=PROVIDER,
        source="agy_cli",
        fetched_at=datetime.now(timezone.utc),
        title="Antigravity subscription",
        windows=tuple(windows),
        raw=raw,
    )


def _query(timeout: float) -> AccountUsageSnapshot | None:
    agy = which_agy()
    if not agy:
        return None
    sandbox = tempfile.mkdtemp(prefix="hermes_agy_usage_")
    try:
        res = subprocess.run(
            [agy, "-p", "/usage", "--output-format", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=sandbox,
            env=build_child_env(),
            stdin=subprocess.DEVNULL,
            **{k: v for k, v in popen_kwargs().items() if k == "creationflags"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("agy /usage failed: %s", exc)
        return None
    finally:
        remove_dir(sandbox)
    parsed = parse_usage_output(res.stdout)
    if parsed is None:
        logger.debug("agy /usage returned no quota data (rc=%s): %.300s", res.returncode, res.stderr)
        return None
    return build_snapshot(*parsed)


def get_account_usage_snapshot(*, force_refresh: bool = False, timeout: float = 10.0) -> AccountUsageSnapshot | None:
    """Cached for a minute; one refresh at a time; a failed unforced refresh serves the stale value."""
    global _cache
    with _cache_lock:
        cached = _cache
    if cached and not force_refresh and time.monotonic() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]
    started = time.monotonic()
    with _refresh_lock:
        with _cache_lock:
            if _cache and _cache[0] >= started and not force_refresh:
                return _cache[1]
        snapshot = _query(timeout)
        with _cache_lock:
            if snapshot is not None:
                _cache = (time.monotonic(), snapshot)
                return snapshot
            if force_refresh or _cache is None:
                return None
            return _cache[1]

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

FALLBACK_MODELS = (
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "gemini-3.7-flash-high",
    "gemini-3.7-flash-medium",
    "gemini-3.7-flash-low",
    "gemini-3.6-flash-high",
    "gemini-3.6-flash-medium",
    "gemini-3.6-flash-low",
    "gemini-3.1-pro-high",
    "gemini-3.1-pro-low",
    "claude-opus-5-5-high",
    "claude-opus-5-5-medium",
    "claude-opus-5-5-low",
    "claude-sonnet-5-5-high",
    "claude-sonnet-5-5-medium",
    "claude-sonnet-5-5-low",
    "claude-opus-4-6-thinking",
    "claude-sonnet-4-6",
    "gpt-oss-120b-medium",
)

MODEL_ALIASES = {
    "flash": "gemini-3.8-flash-medium",
    "flash-high": "gemini-3.8-flash-high",
    "flash-low": "gemini-3.8-flash-low",
    "pro": "gemini-3.1-pro-high",
    "pro-low": "gemini-3.1-pro-low",
    "sonnet": "claude-sonnet-5-5-medium",
    "opus": "claude-opus-5-5-high",
}

_MODEL_EFFORTS: dict[str, tuple[str, ...]] = {
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("low", "medium", "high"),
    "gemini-3.1-pro": ("low", "high"),
    "claude-opus-5-5": ("low", "medium", "high"),
    "claude-sonnet-5-5": ("low", "medium", "high"),
    "gpt-oss-120b": ("medium",),
}

_EXACT_MODEL_CONTEXT_LENGTHS: dict[str, int] = {
    "gemini-3.8-flash": 1048576,
    "gemini-3.7-flash": 1048576,
    "gemini-3.6-flash": 1048576,
    "gemini-3.1-pro": 2097152,
    "claude-opus-5-5": 200000,
    "claude-sonnet-5-5": 200000,
    "claude-opus-4-6-thinking": 200000,
    "claude-sonnet-4-6": 200000,
    "gpt-oss-120b": 131072,
}

_CONTEXT_LENGTHS: dict[str, int] = {
    "gemini": 1048576,
    "claude": 200000,
    "gpt": 131072,
}

_model_cache: dict[str, Any] = {"ts": 0.0, "models": []}


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def find_agy() -> str:
    override = os.environ.get("AGY_CLI_PATH")
    if override and os.path.isfile(override):
        return override
    found = shutil.which("agy") or shutil.which("agy.exe")
    if found:
        return found
    fallback = os.path.expandvars(r"%LOCALAPPDATA%\agy\bin\agy.exe")
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError("agy CLI not found in PATH or standard installation directories")


def fetch_models(timeout: float = 15.0) -> list[str]:
    now = time.time()
    if now - _model_cache["ts"] < 300.0 and _model_cache["models"]:
        return list(_model_cache["models"])
    try:
        agy = find_agy()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.run(
            [agy, "models"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=flags,
        )
        raw = strip_ansi((proc.stdout or "") + (proc.stderr or ""))
        discovered: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("Fetching"):
                continue
            parts = line.split(maxsplit=1)
            model_id = parts[0]
            if any(model_id.startswith(p) for p in ("gemini", "claude", "gpt")):
                discovered.append(model_id)
        if discovered:
            _model_cache["ts"] = now
            _model_cache["models"] = discovered
            return discovered
    except Exception:
        pass
    return list(FALLBACK_MODELS)


def split_effort_suffix(model: str) -> tuple[str, str | None]:
    for effort in ("high", "medium", "low"):
        if model.endswith(f"-{effort}"):
            base = model[: -(len(effort) + 1)]
            return base, effort
    return model, None


def model_efforts(model: str | None) -> tuple[str, ...] | None:
    if not model:
        return ("low", "medium", "high")
    base, suffix = split_effort_suffix(model)
    if base in _MODEL_EFFORTS:
        return _MODEL_EFFORTS[base]
    for key, efforts in _MODEL_EFFORTS.items():
        if model.startswith(key):
            return efforts
    return ("low", "medium", "high")


def get_model_context_length(model: str | None) -> int:
    if not model:
        return 1048576
    lower = model.lower().strip()
    if "/" in lower:
        lower = lower.split("/")[-1]
    if lower in MODEL_ALIASES:
        lower = MODEL_ALIASES[lower].lower()
    base, _ = split_effort_suffix(lower)
    if lower in _EXACT_MODEL_CONTEXT_LENGTHS:
        return _EXACT_MODEL_CONTEXT_LENGTHS[lower]
    if base in _EXACT_MODEL_CONTEXT_LENGTHS:
        return _EXACT_MODEL_CONTEXT_LENGTHS[base]
    for key, length in _EXACT_MODEL_CONTEXT_LENGTHS.items():
        if base.startswith(key) or key.startswith(base):
            return length
    for prefix, length in _CONTEXT_LENGTHS.items():
        if prefix in lower:
            return length
    return 1048576


MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    m: {
        "context_length": get_model_context_length(m),
        "supports_tools": True,
        "supports_vision": True,
    }
    for m in FALLBACK_MODELS
}



def resolve_model(
    model: str,
    reasoning_effort: str | None = None,
    known_models: list[str] | set[str] | None = None,
) -> str:
    target = (model or "").strip()
    if "/" in target:
        target = target.split("/")[-1]
    if target in MODEL_ALIASES:
        target = MODEL_ALIASES[target]
    known = set(known_models) if known_models else set(fetch_models())
    if target in known:
        return target
    base, current_effort = split_effort_suffix(target)
    effort = reasoning_effort or current_effort
    if effort and f"{base}-{effort}" in known:
        return f"{base}-{effort}"
    for k in known:
        if k.lower() == target.lower():
            return k
    for k in known:
        if k.startswith(base):
            return k
    return target

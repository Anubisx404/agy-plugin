"""Model catalog, alias/effort resolution and context bounds for the agy CLI."""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from typing import Any

try:
    from .process import build_child_env, getenv_any, popen_kwargs, which_agy
except ImportError:
    from process import build_child_env, getenv_any, popen_kwargs, which_agy

logger = logging.getLogger(__name__)

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

EFFORTS = ("low", "medium", "high")
_EFFORT_RANK = {"low": 0, "medium": 1, "high": 2}
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_EFFORT = "medium"

# Efforts agy accepts per base model. agy lists effort variants as `<base>-<effort>` and takes
# `--effort`; a model listed by bare name takes no `--effort`. `agy models` is the source of truth;
# this table only answers before the catalog is loaded and when `agy models` fails.
KNOWN_EFFORTS: dict[str, tuple[str, ...]] = {
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("low", "medium", "high"),
    "gemini-3.1-pro": ("low", "high"),
    "claude-opus-5-5": ("low", "medium", "high"),
    "claude-sonnet-5-5": ("low", "medium", "high"),
    "claude-opus-4-6-thinking": (),
    "claude-sonnet-4-6": (),
    "gpt-oss-120b": ("medium",),
}

FALLBACK_MODELS: tuple[str, ...] = tuple(KNOWN_EFFORTS)

# Short names -> base model. An effort suffix may follow any alias (`flash-high`, `pro-low`).
MODEL_ALIASES: dict[str, str] = {
    "default": "gemini-3.8-flash",
    "flash": "gemini-3.8-flash",
    "gemini-flash": "gemini-3.8-flash",
    "pro": "gemini-3.1-pro",
    "gemini-pro": "gemini-3.1-pro",
    "sonnet": "claude-sonnet-5-5",
    "claude-sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "claude-opus": "claude-opus-5-5",
    "gpt": "gpt-oss-120b",
    "gpt-oss": "gpt-oss-120b",
}

# When the account's catalog lacks the requested base, the newest catalog model of the same
# family is used instead (e.g. claude-opus-5-5 -> claude-opus-4-6-thinking on older plans).
_FAMILIES = ("opus", "sonnet", "flash", "pro", "gpt-oss")

# Native windows, and agy's own trajectory limit. agy checkpoints a trajectory at ~239,616 tokens
# (256k minus 16k reserved output) regardless of the model, so the declared window is capped there
# to make Hermes compress before agy silently truncates.
AGY_TRAJECTORY_LIMIT = 239_000
_NATIVE_CONTEXT: tuple[tuple[str, int], ...] = (
    ("gemini-3.1-pro", 2_097_152),
    ("gemini", 1_048_576),
    ("claude", 200_000),
    ("gpt-oss", 131_072),
    ("gpt", 131_072),
)

_CATALOG_TTL = 3600.0
_CATALOG_FAILURE_TTL = 60.0
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_MODEL_ID_RE = re.compile(r"^[a-z][a-z0-9._]*-[a-z0-9._-]+$")
_SKIP_PREFIXES = ("fetching", "error", "usage", "available", "warning", "--", "model ")

_catalog_lock = threading.Lock()
_catalog: tuple[float, dict[str, tuple[str, ...]]] | None = None


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def split_effort_suffix(model: str) -> tuple[str, str | None]:
    """``gemini-3.8-flash-high`` -> (``gemini-3.8-flash``, ``high``)."""
    for effort in ("medium", "high", "low"):
        if model.endswith(f"-{effort}"):
            return model[: -(len(effort) + 1)], effort
    return model, None


def normalize_effort(effort: Any) -> str | None:
    """Map Hermes' effort vocabulary onto agy's low/medium/high."""
    if effort is None:
        return None
    e = str(effort).strip().lower()
    if not e:
        return None
    if e in ("none", "off", "minimal", "min", "low"):
        return "low"
    if e in ("medium", "default", "auto"):
        return "medium"
    if e in ("high", "xhigh", "max", "maximum", "ultra"):
        return "high"
    return None


def parse_catalog(text: str) -> dict[str, tuple[str, ...]]:
    """Group `agy models` output as {base model: supported efforts}, keeping agy's order."""
    grouped: dict[str, list[str]] = {}
    for raw in strip_ansi(text or "").splitlines():
        line = raw.strip().lstrip(_SPINNER).strip()
        if not line or line.lower().startswith(_SKIP_PREFIXES):
            continue
        model_id = line.split()[0].strip("*•-:,").lower()
        if not _MODEL_ID_RE.match(model_id):
            continue
        base, effort = split_effort_suffix(model_id)
        efforts = grouped.setdefault(base, [])
        if effort and effort not in efforts:
            efforts.append(effort)
    return {base: tuple(sorted(e, key=_EFFORT_RANK.__getitem__)) for base, e in grouped.items()}


def cached_catalog() -> dict[str, tuple[str, ...]] | None:
    """The live catalog while fresh, without spawning anything."""
    entry = _catalog
    if entry is not None and time.monotonic() < entry[0]:
        return entry[1]
    return None


def load_catalog(timeout: float = 15.0, *, force: bool = False) -> dict[str, tuple[str, ...]]:
    """Run `agy models` (single-flight, cached for an hour; failures cached for a minute)."""
    global _catalog
    if not force:
        fresh = cached_catalog()
        if fresh is not None:
            return fresh
    with _catalog_lock:
        if not force:
            fresh = cached_catalog()
            if fresh is not None:
                return fresh
        catalog: dict[str, tuple[str, ...]] = {}
        agy = which_agy()
        if agy:
            try:
                res = subprocess.run(
                    [agy, "models"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    env=build_child_env(),
                    stdin=subprocess.DEVNULL,
                    **{k: v for k, v in popen_kwargs().items() if k == "creationflags"},
                )
                catalog = parse_catalog((res.stdout or "") + "\n" + (res.stderr or ""))
            except Exception as exc:
                logger.debug("agy models failed: %s", exc)
        ttl = _CATALOG_TTL if catalog else _CATALOG_FAILURE_TTL
        _catalog = (time.monotonic() + ttl, catalog)
        return catalog


def _set_catalog_for_tests(catalog: dict[str, tuple[str, ...]] | None) -> None:
    global _catalog
    _catalog = None if catalog is None else (time.monotonic() + _CATALOG_TTL, dict(catalog))


def fetch_models(timeout: float = 15.0) -> list[str]:
    """Base model ids the account can use; efforts are chosen through Hermes' reasoning setting."""
    catalog = load_catalog(timeout=timeout)
    return list(catalog) if catalog else list(FALLBACK_MODELS)


def _natural_key(name: str) -> tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def _strip_provider_prefix(model: str) -> str:
    return model.rsplit("/", 1)[-1] if "/" in model else model


def _alias_base(model: str) -> tuple[str, str | None]:
    """Resolve aliases and split the effort suffix: returns (base, suffix effort)."""
    name = _strip_provider_prefix((model or DEFAULT_MODEL).strip().lower()) or DEFAULT_MODEL
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name], None
    base, suffix = split_effort_suffix(name)
    return MODEL_ALIASES.get(base, base), suffix


def _family_substitute(base: str, catalog: dict[str, tuple[str, ...]]) -> str | None:
    for family in _FAMILIES:
        if family in base:
            vendor = base.split("-", 1)[0]
            matches = [k for k in catalog if family in k and k.startswith(vendor)]
            if matches:
                return max(matches, key=_natural_key)
    return None


def _heuristic_efforts(base: str, has_suffix: bool) -> tuple[str, ...]:
    if base.startswith("gemini-") and "pro" in base:
        return ("low", "high")
    if base.startswith("gemini-") and "flash" in base:
        return EFFORTS
    return EFFORTS if has_suffix else ()


def efforts_for(model: str | None) -> tuple[str, ...] | None:
    """Efforts agy accepts for ``model`` (never spawns a process); () = no --effort; None = unknown."""
    if not model:
        return None
    base, suffix = _alias_base(model)
    catalog = cached_catalog()
    if catalog and base in catalog:
        return catalog[base]
    if base in KNOWN_EFFORTS:
        return KNOWN_EFFORTS[base]
    if catalog:
        substitute = _family_substitute(base, catalog)
        if substitute:
            return catalog[substitute]
    return _heuristic_efforts(base, suffix is not None)


def _nearest_effort(requested: str, supported: tuple[str, ...]) -> str:
    target = _EFFORT_RANK[requested]
    return min(supported, key=lambda e: (abs(_EFFORT_RANK[e] - target), -_EFFORT_RANK[e]))


def resolve_model(
    model: str | None,
    reasoning_effort: Any = None,
    *,
    catalog: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, str | None]:
    """Map a Hermes model request to (agy model id, --effort value or None).

    Effort precedence: Hermes' reasoning effort > the model id's suffix > medium, then the nearest
    effort the model supports (the stronger one on a tie). Models agy lists by bare name get no effort.
    """
    base, suffix = _alias_base(model or DEFAULT_MODEL)
    if catalog is None:
        catalog = cached_catalog()
        if catalog is None:
            catalog = load_catalog(timeout=10.0)
    if catalog and base not in catalog:
        substitute = _family_substitute(base, catalog)
        if substitute and substitute != base:
            logger.info("agy catalog has no %s; using %s", base, substitute)
            base = substitute
    if catalog and base in catalog:
        supported = catalog[base]
    elif base in KNOWN_EFFORTS:
        supported = KNOWN_EFFORTS[base]
    else:
        supported = _heuristic_efforts(base, suffix is not None)
    if not supported:
        return base, None
    requested = normalize_effort(reasoning_effort) or suffix or DEFAULT_EFFORT
    effort = _nearest_effort(requested, supported)
    return f"{base}-{effort}", effort


def native_context_length(model: str | None) -> int:
    base, _ = _alias_base(model or DEFAULT_MODEL)
    for prefix, length in _NATIVE_CONTEXT:
        if base.startswith(prefix):
            return length
    return 1_048_576


def get_model_context_length(model: str | None) -> int:
    """Context window declared to Hermes: an explicit override, else min(native, agy's limit)."""
    override = getenv_any("AGY_CONTEXT_LENGTH", "HERMES_ANTIGRAVITY_CONTEXT_LENGTH", "ANTIGRAVITY_CONTEXT_LENGTH")
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass
    return min(native_context_length(model), AGY_TRAJECTORY_LIMIT)


def model_capabilities() -> dict[str, dict[str, Any]]:
    """Canonical per-model metadata (Hermes `model_overrides` schema) for every known id spelling."""
    caps: dict[str, dict[str, Any]] = {}
    for base, efforts in KNOWN_EFFORTS.items():
        entry = {
            "context_window": get_model_context_length(base),
            "supports_tools": True,
            # agy receives the prompt as text over stdin: image parts cannot reach the model, so
            # Hermes must route images through its vision tool instead of attaching them.
            "supports_vision": False,
            "supports_reasoning": bool(efforts),
        }
        caps[base] = dict(entry)
        for effort in efforts:
            caps[f"{base}-{effort}"] = dict(entry)
    return caps


MODEL_CAPABILITIES = model_capabilities()

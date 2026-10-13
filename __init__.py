"""Hermes model-provider plugin: Google Antigravity (`agy`) with persistent background workers."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

try:
    from providers import register_provider
    from providers.base import ProviderProfile
except ImportError:  # loaded outside Hermes (tests, tooling)

    class ProviderProfile:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def register_provider(profile: Any) -> None:  # type: ignore[no-redef]
        return None


try:
    from . import client as _client
    from . import usage as _usage
    from .models import (
        FALLBACK_MODELS,
        MODEL_ALIASES,
        MODEL_CAPABILITIES,
        efforts_for,
        fetch_models,
        get_model_context_length,
        load_catalog,
        normalize_effort,
    )
    from .process import COMMAND_ENV_VARS, detect_login, which_agy
except ImportError:
    import client as _client  # type: ignore[no-redef]
    import usage as _usage  # type: ignore[no-redef]
    from models import (  # type: ignore[no-redef]
        FALLBACK_MODELS,
        MODEL_ALIASES,
        MODEL_CAPABILITIES,
        efforts_for,
        fetch_models,
        get_model_context_length,
        load_catalog,
        normalize_effort,
    )
    from process import COMMAND_ENV_VARS, detect_login, which_agy  # type: ignore[no-redef]

AntigravityClient = _client.AntigravityClient
SESSION_KEY_FIELD = _client.SESSION_KEY_FIELD


def _classify_api_error(
    error: Exception,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str = "",
    body: Any = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    return _client.classify_error(error, message)


class AntigravityProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        return AntigravityClient(**client_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> list[str] | None:
        return fetch_models(timeout=timeout)

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        return efforts_for(model)

    def get_model_context_length(self, model: str) -> int | None:
        return get_model_context_length(model)

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Forward Hermes' reasoning effort to the client, which maps it onto agy's effort variants."""
        if not isinstance(reasoning_config, dict):
            return {}, {}
        if reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "low"}  # agy has no "off"; low is the minimum
        effort = normalize_effort(reasoning_config.get("effort"))
        return {}, ({"reasoning_effort": effort} if effort else {})

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Hand the Hermes session id to the client so the session keeps its persistent agy worker."""
        return {SESSION_KEY_FIELD: str(session_id)} if session_id else {}

    def setup_status(self, **kwargs: Any) -> dict[str, Any] | None:
        command = which_agy()
        if not command:
            return {
                "available": False,
                "logged_in": False,
                "plan": "Google Antigravity",
                "detail": "agy CLI not found on PATH (set AGY_CLI_PATH to its location)",
                "login_command": "agy",
            }
        login = detect_login()
        if login is False:
            detail = f"agy at {command}; not signed in"
        elif login is None:
            detail = f"agy at {command} (sign-in stored in the OS keyring; not verified)"
        else:
            detail = f"agy at {command}; signed in"
        return {
            "available": True,
            "logged_in": login is not False,
            "plan": "Google Antigravity",
            "detail": detail,
            "login_command": "agy",
        }

    def discover_models(self, **kwargs: Any) -> list[dict[str, Any]] | None:
        catalog = load_catalog() or {m: efforts_for(m) or () for m in FALLBACK_MODELS}
        entries = []
        for model, efforts in catalog.items():
            context_k = get_model_context_length(model) // 1000
            note = f"{context_k}k context"
            if efforts:
                note += " · effort " + "/".join(efforts)
            entries.append({"id": model, "label": model, "note": note})
        return entries

    def fetch_account_usage(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> Any:
        return _usage.get_account_usage_snapshot(force_refresh=force_refresh)


def _supported_fields(**fields: Any) -> dict[str, Any]:
    """Drop profile fields this Hermes does not define, so an older Hermes still loads the plugin."""
    if not dataclasses.is_dataclass(ProviderProfile):
        return fields
    known = {f.name for f in dataclasses.fields(ProviderProfile)}
    dropped = sorted(set(fields) - known)
    if dropped:
        logging.getLogger(__name__).debug("Hermes ProviderProfile lacks %s; skipping them", ", ".join(dropped))
    return {k: v for k, v in fields.items() if k in known}


antigravity = AntigravityProfile(
    **_supported_fields(
        name="antigravity",
        aliases=("antigravity-agy", "agy"),
        display_name="Antigravity",
        description="Google Antigravity subscription via the agy CLI (persistent workers)",
        api_mode="chat_completions",
        auth_type="external_process",
        base_url="acp://antigravity",
        supports_health_check=False,
        process_command="agy",
        process_command_env_vars=COMMAND_ENV_VARS,
        process_args_env_var="AGY_ARGS",
        fallback_models=FALLBACK_MODELS,
        model_aliases=MODEL_ALIASES,
        model_capabilities=MODEL_CAPABILITIES,
        default_aux_model="gemini-3.8-flash",
        # agy receives prompts as text: images would be silently dropped, so let Hermes describe them.
        supports_vision=False,
        supports_vision_tool_messages=False,
        unsupported_response_formats=("json_schema",),
        classify_api_error=_classify_api_error,
    )
)

register_provider(antigravity)

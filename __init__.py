from __future__ import annotations

from typing import Any

try:
    from providers import register_provider
    from providers.base import ProviderProfile
except ImportError:
    class ProviderProfile:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def register_provider(profile: Any) -> None:
        pass

try:
    from .client import AntigravityClient
    from .models import (
        FALLBACK_MODELS,
        MODEL_ALIASES,
        MODEL_CAPABILITIES,
        fetch_models,
        find_agy,
        get_model_context_length,
        model_efforts,
    )
except ImportError:
    from client import AntigravityClient
    from models import (
        FALLBACK_MODELS,
        MODEL_ALIASES,
        MODEL_CAPABILITIES,
        fetch_models,
        find_agy,
        get_model_context_length,
        model_efforts,
    )


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

    def supported_reasoning_efforts(
        self, model: str | None
    ) -> tuple[str, ...] | None:
        return model_efforts(model)

    def get_model_context_length(self, model: str) -> int | None:
        return get_model_context_length(model)

    def setup_status(self, **kwargs: Any) -> dict[str, Any] | None:
        cmd = find_agy()
        if not cmd:
            return {
                "available": False,
                "logged_in": False,
                "plan": "Google Antigravity",
                "detail": "agy CLI not found on PATH",
                "login_command": "agy install",
            }
        return {
            "available": True,
            "logged_in": True,
            "plan": "Google Antigravity",
            "detail": f"Installed at {cmd}",
            "login_command": "agy",
        }

    def discover_models(self, **kwargs: Any) -> list[dict[str, Any]] | None:
        models = self.fetch_models() or list(FALLBACK_MODELS)
        return [
            {
                "id": m,
                "label": m,
                "note": f"{get_model_context_length(m) // 1024}k context",
            }
            for m in models
        ]


antigravity = AntigravityProfile(
    name="antigravity",
    aliases=("antigravity-agy", "agy"),
    display_name="Antigravity",
    description="Google Antigravity (agy) direct model provider",
    api_mode="chat_completions",
    auth_type="external_process",
    base_url="acp://antigravity",
    process_command="agy",
    process_command_env_vars=("AGY_CLI_PATH", "HERMES_AGY_COMMAND"),
    fallback_models=FALLBACK_MODELS,
    model_aliases=MODEL_ALIASES,
    model_capabilities=MODEL_CAPABILITIES,
    supports_vision=True,
    supports_vision_tool_messages=True,
)

register_provider(antigravity)


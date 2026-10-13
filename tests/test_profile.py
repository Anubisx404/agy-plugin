"""Hermes-facing contract: profile registration, hook outputs, error classification, client facade."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import antigravity_plugin as plugin
from antigravity_plugin import client


def test_profile_metadata():
    p = plugin.antigravity
    assert p.name == "antigravity" and p.auth_type == "external_process"
    assert p.base_url == "acp://antigravity" and p.process_command == "agy"
    assert "AGY_CLI_PATH" in p.process_command_env_vars
    assert p.supports_vision is False and p.supports_vision_tool_messages is False
    assert "gemini-3.8-flash" in p.fallback_models
    assert p.model_aliases["flash"] == "gemini-3.8-flash"
    assert p.model_capabilities["gemini-3.8-flash-low"]["context_window"] == 239_000
    assert callable(p.classify_api_error)


def test_reasoning_effort_is_forwarded():
    p = plugin.antigravity
    assert p.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "xhigh"}) == (
        {},
        {"reasoning_effort": "high"},
    )
    assert p.build_api_kwargs_extras(reasoning_config={"enabled": False}) == ({}, {"reasoning_effort": "low"})
    assert p.build_api_kwargs_extras(reasoning_config=None) == ({}, {})


def test_session_id_travels_in_extra_body():
    assert plugin.antigravity.build_extra_body(session_id="20261012_abc") == {client.SESSION_KEY_FIELD: "20261012_abc"}
    assert plugin.antigravity.build_extra_body(session_id=None, model="x") == {}


def test_supported_efforts_and_context():
    p = plugin.antigravity
    assert p.supported_reasoning_efforts("gemini-3.1-pro") == ("low", "high")
    assert p.supported_reasoning_efforts("claude-sonnet-4-6") == ()
    assert p.get_model_context_length("claude-opus-5-5") == 200_000


def test_setup_status_without_agy(monkeypatch):
    monkeypatch.setattr(plugin, "which_agy", lambda *a, **k: None)
    status = plugin.antigravity.setup_status()
    assert status["available"] is False and status["logged_in"] is False


def test_setup_status_with_agy(monkeypatch):
    monkeypatch.setattr(plugin, "which_agy", lambda *a, **k: "/usr/bin/agy")
    monkeypatch.setattr(plugin, "detect_login", lambda: False)
    assert plugin.antigravity.setup_status()["logged_in"] is False
    monkeypatch.setattr(plugin, "detect_login", lambda: None)
    status = plugin.antigravity.setup_status()
    assert status["available"] is True and status["logged_in"] is True and "not verified" in status["detail"]


def test_discover_models_offline(monkeypatch):
    monkeypatch.setattr(plugin, "load_catalog", lambda *a, **k: {})
    entries = plugin.antigravity.discover_models()
    flash = next(e for e in entries if e["id"] == "gemini-3.8-flash")
    assert "effort low/medium/high" in flash["note"] and "239k" in flash["note"]


@pytest.mark.parametrize(
    ("message", "kind", "reason"),
    [
        ("Antigravity quota exhausted: RESOURCE_EXHAUSTED (code 429)", "quota", "rate_limit"),
        ("Your AI credits balance is too low", "model", "billing"),
        ("Please verify your account to continue", "model", "auth_permanent"),
        ("Antigravity CLI is not signed in", "auth", "auth"),
        ("subscriber fell behind updates, stalled for 5s", "model", "context_overflow"),
        ("no output from agy for 5s", "timeout", "timeout"),
        ("agy CLI not found", "missing", "auth_permanent"),
    ],
)
def test_error_classification(message, kind, reason):
    verdict = plugin._classify_api_error(client.AgyError(message, kind=kind), message="")
    assert verdict["reason"] == reason


def test_unrelated_errors_are_not_classified():
    assert plugin._classify_api_error(client.AgyError("Native tools are blocked", kind="native_tool")) is None
    assert plugin._classify_api_error(ValueError("whatever")) is None


def test_coerce_timeout_accepts_httpx_like_objects(monkeypatch):
    assert client.coerce_timeout(12) == 12.0
    assert client.coerce_timeout(SimpleNamespace(read=45.0, connect=5.0)) == 45.0
    assert client.coerce_timeout(SimpleNamespace(read=None)) == client.DEFAULT_TIMEOUT_SECONDS
    assert client.coerce_timeout(None) == client.DEFAULT_TIMEOUT_SECONDS
    monkeypatch.setenv("AGY_TIMEOUT_SECONDS", "77")
    assert client.coerce_timeout(None) == 77.0


def test_missing_agy_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(client, "which_agy", lambda *a, **k: None)
    c = plugin.AntigravityClient()
    for stream in (False, True):
        with pytest.raises(client.AgyError, match="agy CLI not found"):
            c.chat.completions.create(model="flash", messages=[{"role": "user", "content": "hi"}], stream=stream)


def test_closed_client_refuses_requests():
    c = plugin.AntigravityClient()
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.chat.completions.create(model="flash", messages=[])


def test_completion_objects_are_awaitable():
    completion = client.Completion(id="x", choices=[])

    async def main():
        return await completion

    assert asyncio.run(main()) is completion


def test_output_instruction_for_json_mode():
    assert "single valid JSON object" in client._output_instruction({"type": "json_object"})
    assert '"type": "object"' in client._output_instruction(
        {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
    )
    assert client._output_instruction(None) == ""

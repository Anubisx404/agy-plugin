from __future__ import annotations

from antigravity_plugin import models

CATALOG_TEXT = """\
⠋ Fetching models...
Available models:
gemini-3.8-flash-low      Gemini 3.8 Flash (low)
gemini-3.8-flash-medium   Gemini 3.8 Flash (medium)
gemini-3.8-flash-high     Gemini 3.8 Flash (high)
gemini-3.1-pro-low
gemini-3.1-pro-high
\x1b[32mclaude-opus-4-6-thinking\x1b[0m
claude-sonnet-4-6
gpt-oss-120b-medium
"""


def test_parse_catalog_groups_efforts_and_skips_noise():
    catalog = models.parse_catalog(CATALOG_TEXT)
    assert catalog == {
        "gemini-3.8-flash": ("low", "medium", "high"),
        "gemini-3.1-pro": ("low", "high"),
        "claude-opus-4-6-thinking": (),
        "claude-sonnet-4-6": (),
        "gpt-oss-120b": ("medium",),
    }


def test_split_effort_suffix():
    assert models.split_effort_suffix("gemini-3.8-flash-high") == ("gemini-3.8-flash", "high")
    assert models.split_effort_suffix("claude-sonnet-5-5-medium") == ("claude-sonnet-5-5", "medium")
    assert models.split_effort_suffix("gemini-3.8-flash") == ("gemini-3.8-flash", None)


def test_normalize_effort_maps_hermes_vocabulary():
    assert models.normalize_effort("xhigh") == "high"
    assert models.normalize_effort("max") == "high"
    assert models.normalize_effort("minimal") == "low"
    assert models.normalize_effort("none") == "low"
    assert models.normalize_effort("Medium") == "medium"
    assert models.normalize_effort("bogus") is None
    assert models.normalize_effort(None) is None


def test_resolve_model_effort_precedence():
    catalog = models.parse_catalog(CATALOG_TEXT)
    # default effort
    assert models.resolve_model("gemini-3.8-flash", catalog=catalog) == ("gemini-3.8-flash-medium", "medium")
    # suffix in the id
    assert models.resolve_model("gemini-3.8-flash-low", catalog=catalog) == ("gemini-3.8-flash-low", "low")
    # Hermes reasoning effort wins over the suffix
    assert models.resolve_model("gemini-3.8-flash-low", "high", catalog=catalog) == ("gemini-3.8-flash-high", "high")
    # nearest supported effort, stronger on a tie
    assert models.resolve_model("gemini-3.1-pro", "medium", catalog=catalog) == ("gemini-3.1-pro-high", "high")
    # only one effort available
    assert models.resolve_model("gpt-oss-120b", "low", catalog=catalog) == ("gpt-oss-120b-medium", "medium")


def test_resolve_model_bare_models_take_no_effort():
    catalog = models.parse_catalog(CATALOG_TEXT)
    assert models.resolve_model("claude-sonnet-4-6", "high", catalog=catalog) == ("claude-sonnet-4-6", None)


def test_resolve_model_aliases_and_provider_prefix():
    catalog = models.parse_catalog(CATALOG_TEXT)
    assert models.resolve_model("flash", catalog=catalog) == ("gemini-3.8-flash-medium", "medium")
    assert models.resolve_model("flash-high", catalog=catalog) == ("gemini-3.8-flash-high", "high")
    assert models.resolve_model("antigravity/pro-low", catalog=catalog) == ("gemini-3.1-pro-low", "low")
    assert models.resolve_model(None, catalog=catalog)[0].startswith("gemini-3.8-flash")


def test_resolve_model_family_substitute_when_account_lacks_model():
    catalog = models.parse_catalog(CATALOG_TEXT)
    # claude-opus-5-5 is not on this account: fall back to the newest opus agy lists.
    assert models.resolve_model("opus", catalog=catalog) == ("claude-opus-4-6-thinking", None)
    assert models.resolve_model("claude-sonnet-5-5-high", catalog=catalog) == ("claude-sonnet-4-6", None)


def test_resolve_model_without_catalog_uses_known_table():
    assert models.resolve_model("claude-opus-5-5", "low", catalog={}) == ("claude-opus-5-5-low", "low")
    assert models.resolve_model("some-new-model", catalog={}) == ("some-new-model", None)
    assert models.resolve_model("some-new-model-high", catalog={}) == ("some-new-model-high", "high")


def test_efforts_for_never_spawns_and_handles_unknowns():
    assert models.efforts_for("gemini-3.1-pro-high") == ("low", "high")
    assert models.efforts_for("flash") == ("low", "medium", "high")
    assert models.efforts_for("claude-sonnet-4-6") == ()
    assert models.efforts_for(None) is None


def test_context_length_is_capped_at_agy_trajectory_limit(monkeypatch):
    assert models.get_model_context_length("gemini-3.1-pro-high") == models.AGY_TRAJECTORY_LIMIT
    assert models.get_model_context_length("gemini-3.8-flash") == models.AGY_TRAJECTORY_LIMIT
    assert models.get_model_context_length("claude-sonnet-5-5") == 200_000
    assert models.get_model_context_length("gpt-oss-120b-medium") == 131_072
    monkeypatch.setenv("AGY_CONTEXT_LENGTH", "500000")
    assert models.get_model_context_length("gemini-3.8-flash") == 500_000
    monkeypatch.setenv("AGY_CONTEXT_LENGTH", "nope")
    assert models.get_model_context_length("gemini-3.8-flash") == models.AGY_TRAJECTORY_LIMIT


def test_model_capabilities_use_hermes_schema_for_every_spelling():
    caps = models.model_capabilities()
    for model_id in ("gemini-3.8-flash", "gemini-3.8-flash-low", "gemini-3.1-pro-high", "claude-sonnet-4-6"):
        entry = caps[model_id]
        assert set(entry) == {"context_window", "supports_tools", "supports_vision", "supports_reasoning"}
        assert entry["supports_vision"] is False
    assert caps["claude-sonnet-4-6"]["supports_reasoning"] is False
    assert caps["gemini-3.8-flash"]["supports_reasoning"] is True


def test_fetch_models_falls_back_without_agy(monkeypatch):
    monkeypatch.setattr(models, "which_agy", lambda *a, **k: None)
    assert models.fetch_models() == list(models.FALLBACK_MODELS)


def test_fetch_models_from_fake_agy(fake_agy):
    models._set_catalog_for_tests(None)
    listed = models.fetch_models()
    assert listed[:2] == ["gemini-3.8-flash", "gemini-3.1-pro"]
    assert "claude-sonnet-4-6" in listed

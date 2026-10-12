from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import client
import models
from __init__ import AntigravityProfile


def test_strip_ansi():
    assert models.strip_ansi("\x1b[32mhello\x1b[0m") == "hello"
    assert models.strip_ansi("plain text") == "plain text"
    assert models.strip_ansi("") == ""


def test_split_effort_suffix():
    base, effort = models.split_effort_suffix("gemini-3.8-flash-high")
    assert base == "gemini-3.8-flash"
    assert effort == "high"

    base, effort = models.split_effort_suffix("claude-sonnet-5-5-medium")
    assert base == "claude-sonnet-5-5"
    assert effort == "medium"

    base, effort = models.split_effort_suffix("gemini-3.8-flash")
    assert base == "gemini-3.8-flash"
    assert effort is None


def test_model_efforts():
    assert models.model_efforts("gemini-3.8-flash") == ("low", "medium", "high")
    assert models.model_efforts("gemini-3.1-pro") == ("low", "high")
    assert models.model_efforts("gpt-oss-120b") == ("medium",)


def test_get_model_context_length_exact():
    assert models.get_model_context_length("gemini-3.1-pro-high") == 2097152
    assert models.get_model_context_length("gemini-3.8-flash-medium") == 1048576
    assert models.get_model_context_length("claude-sonnet-5-5") == 200000
    assert models.get_model_context_length("gpt-oss-120b-medium") == 131072


def test_resolve_model():
    known = ["gemini-3.8-flash-high", "gemini-3.8-flash-medium", "gemini-3.8-flash-low"]
    assert models.resolve_model("flash", known_models=known) == "gemini-3.8-flash-medium"
    assert models.resolve_model("gemini-3.8-flash", reasoning_effort="high", known_models=known) == "gemini-3.8-flash-high"
    assert models.resolve_model("gemini-3.8-flash-low", known_models=known) == "gemini-3.8-flash-low"


def test_prune_text():
    short_text = "Hello world"
    assert client._prune_text(short_text, 50) == short_text
    long_text = "A" * 500
    pruned = client._prune_text(long_text, 100)
    assert len(pruned) < 250
    assert "output truncated" in pruned


def test_format_messages_as_prompt_pruning_and_errors():
    messages = [
        {"role": "user", "content": "Start conversation"},
        {"role": "assistant", "tool_calls": [{"id": "call_err", "function": {"name": "failing_tool", "arguments": "{}"}}]},
        {"role": "tool", "name": "failing_tool", "tool_call_id": "call_err", "is_error": True, "content": "Error: Command not found"},
    ]
    for i in range(10):
        messages.append({"role": "assistant", "tool_calls": [{"id": f"call_{i}", "function": {"name": "run", "arguments": "{}"}}]})
        messages.append({"role": "tool", "name": "run", "tool_call_id": f"call_{i}", "content": f"Result {i}: " + "X" * 400})
    messages.append({"role": "user", "content": "Final question"})

    prompt = client.format_messages_as_prompt(messages)
    assert "HEADLESS INFERENCE MODE" in prompt
    assert "(Failed)" in prompt
    assert "output truncated" in prompt
    assert "Result 9:" in prompt


def test_parse_tool_calls_tag_format():
    text = '<tool_call>{"name": "read_file", "arguments": {"path": "main.py"}}</tool_call>'
    cleaned, calls = client.parse_tool_calls(text)
    assert cleaned == ""
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    args = json.loads(calls[0].function.arguments)
    assert args["path"] == "main.py"


def test_parse_tool_calls_marker_format():
    text = '<<<TOOL_CALLS>>>\n[{"name": "list_files", "arguments": {"dir": "."}}]\n<<<END_TOOL_CALLS>>>'
    cleaned, calls = client.parse_tool_calls(text)
    assert cleaned == ""
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].function.name == "list_files"


def test_parse_tool_calls_multiple_and_dual():
    text = (
        "Prefix explanation.\n"
        '<tool_call>{"name": "tool_one", "arguments": {"x": 1}}</tool_call>\n'
        "Middle commentary.\n"
        '<<<TOOL_CALLS>>>\n[{"name": "tool_two", "arguments": {"y": 2}}]\n<<<END_TOOL_CALLS>>>'
    )
    cleaned, calls = client.parse_tool_calls(text)
    assert "Prefix explanation." in cleaned
    assert "Middle commentary." in cleaned
    assert calls is not None
    assert len(calls) == 2
    assert calls[0].function.name == "tool_one"
    assert calls[1].function.name == "tool_two"


def test_parse_tool_calls_malformed_json():
    text = '<tool_call>{invalid json}</tool_call>\n<tool_call>{"name": "valid_tool", "arguments": {}}</tool_call>'
    cleaned, calls = client.parse_tool_calls(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].function.name == "valid_tool"


def test_parse_tool_calls_none():
    text = "Just a standard assistant response without tools."
    cleaned, calls = client.parse_tool_calls(text)
    assert cleaned == text
    assert calls is None


def test_client_create_chat_completion_stream_json():
    c = client.AntigravityClient(command="agy")
    fake_proc = MagicMock()
    lines = [
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Hello "}}),
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "world"}}),
        json.dumps({"event": "result", "result": {"usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}}),
        "",
    ]
    fake_proc.stdout.readline.side_effect = lines
    fake_proc.stdin = MagicMock()
    fake_proc.poll.return_value = 0

    with patch("subprocess.Popen", return_value=fake_proc):
        resp = c.chat.completions.create(
            model="gemini-3.8-flash-medium",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert resp.choices[0].message.content == "Hello world"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens == 12


def test_client_create_chat_completion_with_tools_stream_json():
    c = client.AntigravityClient(command="agy")
    fake_proc = MagicMock()
    tool_text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    lines = [
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": tool_text}}),
        json.dumps({"event": "result", "result": {"usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}}}),
        "",
    ]
    fake_proc.stdout.readline.side_effect = lines
    fake_proc.stdin = MagicMock()
    fake_proc.poll.return_value = 0

    with patch("subprocess.Popen", return_value=fake_proc):
        resp = c.chat.completions.create(
            model="gemini-3.8-flash-medium",
            messages=[{"role": "user", "content": "What is the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls is not None
    assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert resp.usage.total_tokens == 35


def test_client_streaming_text_stream_json():
    c = client.AntigravityClient(command="agy")
    fake_proc = MagicMock()
    lines = [
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Chunk 1 "}}),
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Chunk 2"}}),
        json.dumps({"event": "result", "result": {}}),
        "",
    ]
    fake_proc.stdout.readline.side_effect = lines
    fake_proc.stdin = MagicMock()
    fake_proc.poll.return_value = 0

    with patch("subprocess.Popen", return_value=fake_proc):
        chunks = list(
            c.chat.completions.create(
                model="gemini-3.8-flash-medium",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )

    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.content == "Chunk 1 "
    assert chunks[1].choices[0].delta.content == "Chunk 2"
    assert chunks[2].choices[0].finish_reason == "stop"


def test_client_streaming_tool_call_stream_json():
    c = client.AntigravityClient(command="agy")
    fake_proc = MagicMock()
    tool_text = '<tool_call>{"name": "test_tool", "arguments": {"x": 1}}</tool_call>'
    lines = [
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Thinking...\n"}}),
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": tool_text}}),
        json.dumps({"event": "result", "result": {}}),
        "",
    ]
    fake_proc.stdout.readline.side_effect = lines
    fake_proc.stdin = MagicMock()
    fake_proc.poll.return_value = 0

    with patch("subprocess.Popen", return_value=fake_proc):
        chunks = list(
            c.chat.completions.create(
                model="gemini-3.8-flash-medium",
                messages=[{"role": "user", "content": "call tool"}],
                stream=True,
                tools=[{"type": "function", "function": {"name": "test_tool"}}],
            )
        )

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "Thinking..."
    assert chunks[1].choices[0].finish_reason == "tool_calls"
    assert chunks[1].choices[0].delta.tool_calls[0].function.name == "test_tool"


def test_native_tool_interception():
    c = client.AntigravityClient(command="agy")
    fake_proc = MagicMock()
    lines = [
        json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "tool_name": "write_to_file"}}),
        "",
    ]
    fake_proc.stdout.readline.side_effect = lines
    fake_proc.stdin = MagicMock()
    fake_proc.poll.return_value = 0

    with patch("subprocess.Popen", return_value=fake_proc):
        with pytest.raises(RuntimeError, match="Native tools are blocked"):
            c.chat.completions.create(
                model="gemini-3.8-flash-medium",
                messages=[{"role": "user", "content": "write file"}],
            )


def test_profile_hooks():
    p = AntigravityProfile(name="antigravity")
    status = p.setup_status()
    assert status is not None
    assert status["plan"] == "Google Antigravity"

    models_list = p.discover_models()
    assert models_list is not None
    assert len(models_list) > 0
    assert any(m["id"] == "gemini-3.8-flash-high" for m in models_list)


def test_stream_cancellation_terminates_proc():
    c = client.AntigravityClient(command="agy")
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    lines = [
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Chunk 1"}}),
        json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Chunk 2"}}),
    ]
    fake_proc.stdout.readline.side_effect = lines
    fake_proc.stdin = MagicMock()

    with patch("subprocess.Popen", return_value=fake_proc), patch.object(c, "_terminate_proc") as mock_term:
        gen = c.chat.completions.create(
            model="gemini-3.8-flash-medium",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunk = next(gen)
        assert chunk.choices[0].delta.content == "Chunk 1"
        gen.close()
        mock_term.assert_called_with(fake_proc)


def test_antigravity_profile_registration_metadata():
    from __init__ import antigravity
    assert antigravity.base_url == "acp://antigravity"
    assert antigravity.auth_type == "external_process"
    assert antigravity.process_command == "agy"


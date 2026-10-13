"""End-to-end behaviour of persistent workers, driven through a fake agy that speaks stream-json."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import pytest

from antigravity_plugin import antigravity, client, worker
from antigravity_plugin.client import AgyError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


def _client():
    return antigravity.create_client(api_key="antigravity", base_url="acp://antigravity", command="agy", args=[])


def _session(sid: str) -> dict:
    return antigravity.build_extra_body(session_id=sid)


def _ask(c, messages, *, session="s1", stream=False, **kwargs):
    extra = _session(session) if session else None
    return c.chat.completions.create(
        model="gemini-3.8-flash", messages=messages, extra_body=extra, stream=stream, timeout=20, **kwargs
    )


def _text(resp):
    return resp.choices[0].message.content


def _start_pids(fake):
    return [r["pid"] for r in fake.starts()]


def test_worker_is_reused_across_turns_with_delta_prompts(fake_agy):
    c = _client()
    msgs = [{"role": "system", "content": "You are Hermes."}, {"role": "user", "content": "hello"}]
    first = _ask(c, msgs)
    assert _text(first) == "reply 1 from gemini-3.8-flash-medium"
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "and again"}]
    second = _ask(c, msgs)
    assert _text(second) == "reply 2 from gemini-3.8-flash-medium"

    assert len(fake_agy.starts()) == 1, "one persistent process served both turns"
    turns = fake_agy.turns()
    assert turns[0]["pid"] == turns[1]["pid"]
    assert "HEADLESS INFERENCE MODE" in turns[0]["prompt"]
    delta = turns[1]["prompt"]
    assert "HEADLESS INFERENCE MODE" not in delta and "You are Hermes." not in delta and "hello" not in delta
    assert "and again" in delta
    assert "reply 1" not in delta, "the worker's own reply is already in agy's trajectory"
    argv = fake_agy.starts()[0]["argv"]
    assert argv[:4] == ["--input-format", "stream-json", "--output-format", "stream-json"]
    assert ["--model", "gemini-3.8-flash-medium"] == argv[argv.index("--model") : argv.index("--model") + 2]
    assert "--effort" in argv and "--dangerously-skip-permissions" not in argv


def test_worker_survives_client_churn(fake_agy):
    """Hermes builds and closes request-scoped clients; the session's worker must outlive them."""
    msgs = [{"role": "user", "content": "hello"}]
    c1 = _client()
    first = _ask(c1, msgs)
    c1.close()
    c2 = _client()
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "next"}]
    _ask(c2, msgs)
    assert len(fake_agy.starts()) == 1


def test_streaming_turns_reuse_the_worker(fake_agy):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    chunks = list(_ask(c, msgs, stream=True))
    text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
    assert text == "reply 1 from gemini-3.8-flash-medium"
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens > 0
    msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "more"}]
    stream = _ask(c, msgs, stream=True)
    for chunk in stream:  # consumers often stop at finish_reason without draining
        if chunk.choices and chunk.choices[0].finish_reason:
            break
    stream.close()
    msgs += [{"role": "assistant", "content": "reply 2 from gemini-3.8-flash-medium"}, {"role": "user", "content": "x"}]
    _ask(c, msgs)
    assert len(fake_agy.starts()) == 1


def test_anonymous_requests_match_by_history(fake_agy):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs, session=None)
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "next"}]
    _ask(c, msgs, session=None)
    _ask(c, [{"role": "user", "content": "unrelated title request"}], session=None)
    pids = [t["pid"] for t in fake_agy.turns()]
    assert pids[0] == pids[1] and pids[2] != pids[0]


def test_sessions_get_separate_workers(fake_agy):
    c = _client()
    _ask(c, [{"role": "user", "content": "a"}], session="alpha")
    _ask(c, [{"role": "user", "content": "b"}], session="beta")
    assert len({w.session_key for w in worker.POOL.workers()}) == 2
    assert len(fake_agy.starts()) == 2


def test_diverged_history_restarts_with_full_prompt(fake_agy):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs)
    compressed = [{"role": "user", "content": "[summary of earlier turns]"}, {"role": "user", "content": "go on"}]
    _ask(c, compressed)
    starts, turns = fake_agy.starts(), fake_agy.turns()
    assert len(starts) == 2 and turns[1]["pid"] != turns[0]["pid"]
    assert "HEADLESS INFERENCE MODE" in turns[1]["prompt"] and "[summary of earlier turns]" in turns[1]["prompt"]
    assert _text(first)


def test_model_switch_resumes_the_same_agy_conversation(fake_agy):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs)
    conversation = worker.POOL.workers()[0].conversation_id
    assert conversation
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "now think harder"}]
    second = c.chat.completions.create(
        model="gemini-3.8-flash", reasoning_effort="high", messages=msgs, extra_body=_session("s1"), timeout=20
    )
    assert _text(second).endswith("gemini-3.8-flash-high")
    argv = fake_agy.starts()[1]["argv"]
    assert argv[argv.index("--conversation") + 1] == conversation
    assert "HEADLESS INFERENCE MODE" not in fake_agy.turns()[1]["prompt"]


def test_idle_worker_is_reaped_then_resumed(fake_agy, monkeypatch):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs)
    proc = worker.POOL.workers()[0].proc
    monkeypatch.setenv("AGY_WORKER_IDLE_SECONDS", "0.05")
    time.sleep(0.1)
    assert worker.POOL.reap_idle() == 1
    assert worker.POOL.workers() == [] and worker.POOL.dormant_sessions() == ["s1"]
    proc.wait(timeout=5)
    monkeypatch.setenv("AGY_WORKER_IDLE_SECONDS", "900")
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "back again"}]
    _ask(c, msgs)
    resumed = fake_agy.starts()[1]["argv"]
    assert "--conversation" in resumed
    assert "HEADLESS INFERENCE MODE" not in fake_agy.turns()[1]["prompt"]


def test_failed_resume_falls_back_to_full_prompt(fake_agy, monkeypatch):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs)
    monkeypatch.setenv("AGY_WORKER_IDLE_SECONDS", "0.01")
    time.sleep(0.05)
    worker.POOL.reap_idle()
    monkeypatch.setenv("AGY_WORKER_IDLE_SECONDS", "900")
    monkeypatch.setenv("FAKE_AGY_REJECT_CONVERSATION", "1")
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "back"}]
    second = _ask(c, msgs)
    assert _text(second).startswith("reply 1")  # a brand-new process answered
    assert "HEADLESS INFERENCE MODE" in fake_agy.turns()[-1]["prompt"]


def test_worker_that_died_while_idle_is_replaced_transparently(fake_agy):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs)
    proc = worker.POOL.workers()[0].proc
    proc.kill()
    proc.wait(timeout=5)
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "still there?"}]
    second = _ask(c, msgs)
    assert _text(second).startswith("reply 1")
    assert len(fake_agy.starts()) == 2


def test_tool_call_turn_then_tool_result_delta(fake_agy):
    c = _client()
    msgs = [{"role": "user", "content": "weather? FAKE:TOOL"}]
    resp = _ask(c, msgs, tools=TOOLS)
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "Let me check."
    call = choice.message.tool_calls[0]
    assert call.function.name == "get_weather" and json.loads(call.function.arguments) == {"city": "Paris"}
    msgs += [
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": call.function.arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call.id, "content": "Sunny, 21C"},
    ]
    final = _ask(c, msgs, tools=TOOLS)
    assert _text(final) == "reply 2 from gemini-3.8-flash-medium"
    delta = fake_agy.turns()[1]["prompt"]
    assert f"[Tool Result: get_weather id={call.id}]\nSunny, 21C" in delta
    assert "Let me check." not in delta and "### get_weather" not in delta
    assert len(fake_agy.starts()) == 1


def test_streamed_tool_calls(fake_agy):
    c = _client()
    chunks = list(_ask(c, [{"role": "user", "content": "FAKE:TOOL"}], tools=TOOLS, stream=True))
    calls = [tc for ch in chunks if ch.choices for tc in (ch.choices[0].delta.tool_calls or [])]
    assert [tc.function.name for tc in calls] == ["get_weather"] and calls[0].index == 0
    assert any(ch.choices and ch.choices[0].finish_reason == "tool_calls" for ch in chunks)


def test_changed_tool_set_is_sent_as_delta(fake_agy):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs, tools=TOOLS)
    more_tools = TOOLS + [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "next"}]
    _ask(c, msgs, tools=more_tools)
    delta = fake_agy.turns()[1]["prompt"]
    assert "## Tool list updated" in delta and "### search" in delta
    assert len(fake_agy.starts()) == 1


def test_usage_is_reported_per_turn(fake_agy):
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs)
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "again"}]
    second = _ask(c, msgs)
    assert first.usage.completion_tokens == 10 and second.usage.completion_tokens == 10
    p2 = fake_agy.turns()[1]["prompt"]
    assert second.usage.prompt_tokens == len(p2) // 4 + 100


def test_native_tool_attempt_is_blocked_and_worker_dropped(fake_agy):
    c = _client()
    started = time.monotonic()
    with pytest.raises(AgyError, match="Native tools are blocked"):
        _ask(c, [{"role": "user", "content": "FAKE:NATIVE"}])
    assert time.monotonic() - started < 10
    assert worker.POOL.workers() == []


def test_cancel_from_another_thread_kills_the_turn(fake_agy):
    c = _client()
    stream = _ask(c, [{"role": "user", "content": "FAKE:HANG"}], stream=True)
    threading.Timer(0.5, c.cancel).start()
    started = time.monotonic()
    with pytest.raises(AgyError, match="cancelled"):
        list(stream)
    assert time.monotonic() - started < 10
    assert worker.POOL.workers() == []


def test_closing_a_stream_mid_turn_discards_the_worker(fake_agy):
    c = _client()
    stream = _ask(c, [{"role": "user", "content": "FAKE:SLOW"}], stream=True)
    first = next(stream)
    assert first.choices[0].delta.content
    proc = worker.POOL.workers()[0].proc
    stream.close()
    proc.wait(timeout=5)
    assert worker.POOL.workers() == []


def test_timeout_without_output(fake_agy):
    c = _client()
    with pytest.raises(AgyError) as info:
        c.chat.completions.create(model="flash", messages=[{"role": "user", "content": "FAKE:HANG"}], timeout=1)
    assert info.value.kind == "timeout"


def test_error_result_fails_turn(fake_agy):
    c = _client()
    with pytest.raises(AgyError, match="Antigravity model error: boom"):
        _ask(c, [{"role": "user", "content": "FAKE:ERROR"}])
    assert worker.POOL.workers() == []


def test_empty_success_is_classified_as_context_overflow(fake_agy):
    c = _client()
    with pytest.raises(AgyError) as info:
        _ask(c, [{"role": "user", "content": "FAKE:EMPTY"}])
    assert info.value.kind == "empty"
    assert client.classify_error(info.value)["reason"] == "context_overflow"


def test_quota_exhaustion_fails_fast(fake_agy):
    c = _client()
    started = time.monotonic()
    with pytest.raises(AgyError) as info:
        _ask(c, [{"role": "user", "content": "FAKE:QUOTA"}])
    assert time.monotonic() - started < 10
    assert info.value.kind == "quota" and "Individual quota reached" in str(info.value)
    assert client.classify_error(info.value)["reason"] == "rate_limit"


def test_stderr_flood_does_not_deadlock(fake_agy):
    c = _client()
    assert _text(_ask(c, [{"role": "user", "content": "FAKE:STDERR_FLOOD"}])).startswith("reply 1")


def test_reasoning_deltas(fake_agy):
    c = _client()
    resp = _ask(c, [{"role": "user", "content": "FAKE:THINK"}])
    assert resp.choices[0].message.reasoning_content == "pondering"
    assert _text(resp) == "reply 1 from gemini-3.8-flash-medium"


def test_unsupported_persistent_mode_falls_back_to_oneshot(fake_agy, monkeypatch):
    monkeypatch.setenv("FAKE_AGY_REJECT_INPUT_FORMAT", "1")
    c = _client()
    assert _text(_ask(c, [{"role": "user", "content": "hello"}])).startswith("reply 1")
    assert worker.POOL.unsupported_reason
    argv = fake_agy.starts()[0]["argv"]
    assert "--input-format" not in argv and "--print-timeout" in argv


def test_rejected_effort_flag_is_dropped(fake_agy, monkeypatch):
    monkeypatch.setenv("FAKE_AGY_REJECT_EFFORT", "1")
    c = _client()
    assert _text(_ask(c, [{"role": "user", "content": "hello"}])).endswith("gemini-3.8-flash-medium")
    assert worker.CLI_CAPS["effort_flag"] is False
    assert "--effort" not in fake_agy.starts()[0]["argv"]


def test_oneshot_mode(fake_agy, monkeypatch):
    monkeypatch.setenv("AGY_DISABLE_PERSISTENT", "1")
    c = _client()
    msgs = [{"role": "user", "content": "hello"}]
    first = _ask(c, msgs)
    msgs += [{"role": "assistant", "content": _text(first)}, {"role": "user", "content": "again"}]
    _ask(c, msgs)
    assert len(fake_agy.starts()) == 2 and worker.POOL.workers() == []
    assert "HEADLESS INFERENCE MODE" in fake_agy.turns()[1]["prompt"]


def test_concurrent_turns_in_one_session_use_a_oneshot(fake_agy):
    c = _client()
    slow = _ask(c, [{"role": "user", "content": "FAKE:SLOW"}], stream=True)
    next(slow)
    other = _ask(c, [{"role": "user", "content": "parallel"}])
    assert _text(other).startswith("reply 1")
    rest = "".join(ch.choices[0].delta.content or "" for ch in slow if ch.choices)
    assert rest
    assert len(worker.POOL.workers()) == 1


def test_capacity_evicts_least_recently_used_idle_worker(fake_agy, monkeypatch):
    monkeypatch.setenv("AGY_MAX_WORKERS", "2")
    c = _client()
    for sid in ("a", "b", "c"):
        _ask(c, [{"role": "user", "content": sid}], session=sid)
    assert sorted(w.session_key for w in worker.POOL.workers()) == ["b", "c"]
    assert worker.POOL.dormant_sessions() == ["a"]


def test_secrets_are_not_passed_to_agy(fake_agy, monkeypatch):
    monkeypatch.setenv("FAKE_SECRET_TOKEN", "hunter2")
    _ask(_client(), [{"role": "user", "content": "hi"}])
    assert fake_agy.starts()[0]["secret"] is None


def test_agy_runs_in_a_private_scratch_dir(fake_agy):
    _ask(_client(), [{"role": "user", "content": "hi"}])
    cwd = fake_agy.starts()[0]["cwd"]
    assert os.path.basename(cwd).startswith("hermes_agy_")
    worker.POOL.shutdown()
    deadline = time.monotonic() + 5
    while os.path.exists(cwd) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not os.path.exists(cwd)


def test_async_await_and_async_iteration(fake_agy):
    c = _client()

    async def main():
        completion = await c.chat.completions.create(
            model="flash", messages=[{"role": "user", "content": "a"}], extra_body=_session("async"), timeout=20
        )
        stream = await c.chat.completions.create(
            model="flash", messages=[{"role": "user", "content": "b"}], stream=True, timeout=20
        )
        parts = [ch.choices[0].delta.content async for ch in stream if ch.choices and ch.choices[0].delta.content]
        return completion, "".join(parts)

    completion, streamed = asyncio.run(main())
    assert _text(completion).startswith("reply 1")
    assert streamed.startswith("reply 1")


def test_account_usage_from_agy(fake_agy):
    snapshot = antigravity.fetch_account_usage(force_refresh=True)
    assert snapshot.provider == "antigravity"
    assert [(w.label, w.used_percent) for w in snapshot.windows] == [("Gemini (5h)", 25.0), ("Gemini (Weekly)", 50.0)]


def test_shutdown_stops_every_worker(fake_agy):
    c = _client()
    _ask(c, [{"role": "user", "content": "a"}], session="x")
    _ask(c, [{"role": "user", "content": "b"}], session="y")
    procs = [w.proc for w in worker.POOL.workers()]
    worker.POOL.shutdown()
    assert all(p.poll() is not None for p in procs)

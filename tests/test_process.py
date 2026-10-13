from __future__ import annotations

import io
import time

from antigravity_plugin import process


def test_child_env_drops_credentials_and_keeps_runtime(monkeypatch):
    base = {
        "PATH": "/usr/bin",
        "HOME": "/home/u",
        "LANG": "C.UTF-8",
        "SLACK_BOT_TOKEN": "x",
        "OPENAI_API_KEY": "x",
        "AWS_SECRET_ACCESS_KEY": "x",
        "DATABASE_URL": "x",
        "GITHUB_TOKEN": "x",
        "SENTRY_DSN": "x",
        "MY_PASSWORD": "x",
        "SSH_AUTH_SOCK": "/tmp/agent",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/bus",
        "XAUTHORITY": "/x",
        "TOKENIZERS_PARALLELISM": "false",
    }
    env = process.build_child_env(base)
    for kept in (
        "PATH",
        "HOME",
        "LANG",
        "SSH_AUTH_SOCK",
        "DBUS_SESSION_BUS_ADDRESS",
        "XAUTHORITY",
        "TOKENIZERS_PARALLELISM",
    ):
        assert kept in env, kept
    for dropped in (
        "SLACK_BOT_TOKEN",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "DATABASE_URL",
        "GITHUB_TOKEN",
        "SENTRY_DSN",
        "MY_PASSWORD",
    ):
        assert dropped not in env, dropped


def test_child_env_passthrough_and_strict(monkeypatch):
    base = {"PATH": "/bin", "HOME": "/h", "CUSTOM_TOKEN": "keep", "RANDOM_SETTING": "1", "LC_ALL": "C"}
    monkeypatch.setenv("AGY_ENV_PASSTHROUGH", "CUSTOM_TOKEN")
    assert process.build_child_env(base)["CUSTOM_TOKEN"] == "keep"
    monkeypatch.setenv("AGY_ENV_STRICT", "1")
    strict = process.build_child_env(base)
    assert set(strict) >= {"PATH", "HOME", "CUSTOM_TOKEN", "LC_ALL"}
    assert "RANDOM_SETTING" not in strict
    monkeypatch.setenv("AGY_ENV_ALLOWLIST", "RANDOM_SETTING")
    assert "RANDOM_SETTING" in process.build_child_env(base)


def test_child_env_proxy(monkeypatch):
    monkeypatch.setenv("AGY_PROXY", "http://127.0.0.1:8080")
    env = process.build_child_env({"PATH": "/bin", "no_proxy": "corp.example"})
    assert env["HTTPS_PROXY"] == env["http_proxy"] == "http://127.0.0.1:8080"
    assert env["NO_PROXY"].split(",") == ["corp.example", "localhost", "127.0.0.1", "::1"]


def test_find_quota_error_trims_retry_chatter():
    text = "noise\nW] call failed (RESOURCE_EXHAUSTED (code 429): Individual quota reached.), retrying in 5s\n"
    assert process.find_quota_error(text) == "RESOURCE_EXHAUSTED (code 429): Individual quota reached."
    assert process.find_quota_error("all good") is None
    assert "credits balance is too low" in process.find_quota_error("Your AI credits balance is too low")


def test_line_reader_times_out_and_reports_eof():
    reader = process.LineReader(io.StringIO("one\ntwo\n"))
    assert reader.get(1.0) == "one\n"
    assert reader.get(1.0) == "two\n"
    assert reader.get(1.0) is process.EOF
    assert reader.get(0.05) is None


def test_line_reader_abort_and_drain():
    reader = process.LineReader(io.StringIO("stale\n"))
    time.sleep(0.1)
    assert reader.drain() == ["stale\n"]
    reader.abort()
    # EOF was queued by the finished thread before the abort
    assert reader.get(0.5) in (process.EOF, process.ABORTED)


def test_stderr_tail_is_bounded_and_marks_turns():
    tail = process.StderrTail(io.StringIO("a" * 100 + "\n" + "b" * 10 + "\n"), max_chars=50)
    assert tail.text(wait=1.0).endswith("b" * 10 + "\n")
    assert len(tail.text()) <= 50
    mark = tail.mark()
    assert tail.text(mark) == ""


def test_which_agy_honours_env(monkeypatch, tmp_path):
    exe = tmp_path / "agy-custom"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("AGY_CLI_PATH", str(exe))
    assert process.which_agy() == str(exe)


def test_quota_log_watch_only_reads_its_own_new_log(monkeypatch, tmp_path):
    log_dir = tmp_path / "agy-config" / "log"
    log_dir.mkdir(parents=True)
    (log_dir / "cli-old.log").write_text("RESOURCE_EXHAUSTED (code 429): someone else's quota\n")
    watch = process.QuotaLogWatch()
    assert watch.poll() is None  # the pre-existing log is not ours
    mine = log_dir / "cli-new.log"
    mine.write_text("starting\n")
    watch.mark_turn_start()
    assert watch.poll() is None
    with open(mine, "a") as fh:
        fh.write("E] RESOURCE_EXHAUSTED (code 429): Individual quota reached.), retrying in 3s\n")
    assert watch.poll() == "RESOURCE_EXHAUSTED (code 429): Individual quota reached."


def test_quota_log_watch_stays_silent_when_ambiguous(tmp_path):
    log_dir = tmp_path / "agy-config" / "log"
    log_dir.mkdir(parents=True)
    watch = process.QuotaLogWatch()
    (log_dir / "cli-a.log").write_text("RESOURCE_EXHAUSTED (code 429)\n")
    (log_dir / "cli-b.log").write_text("RESOURCE_EXHAUSTED (code 429)\n")
    assert watch.poll() is None

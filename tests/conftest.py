"""Load the plugin the way Hermes does (as a package, relative imports) and provide a fake agy."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "antigravity_plugin"

if PACKAGE not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[PACKAGE] = _module
    _spec.loader.exec_module(_module)

from antigravity_plugin import models, worker  # noqa: E402

_ENV_VARS = (
    "AGY_CLI_PATH",
    "HERMES_AGY_COMMAND",
    "ANTIGRAVITY_COMMAND",
    "ANTIGRAVITY_CLI_PATH",
    "AGY_ARGS",
    "AGY_MODE",
    "AGY_WORKER_IDLE_SECONDS",
    "AGY_MAX_WORKERS",
    "AGY_DISABLE_PERSISTENT",
    "AGY_TIMEOUT_SECONDS",
    "AGY_CONTEXT_LENGTH",
    "HERMES_ANTIGRAVITY_CONTEXT_LENGTH",
    "ANTIGRAVITY_CONTEXT_LENGTH",
    "AGY_PROXY",
    "AGY_ENV_STRICT",
    "AGY_ENV_PASSTHROUGH",
    "AGY_ENV_ALLOWLIST",
    "AGY_DEBUG_PROMPT_DIR",
    "AGY_CONFIG_DIR",
    "FAKE_AGY_REJECT_INPUT_FORMAT",
    "FAKE_AGY_REJECT_EFFORT",
    "FAKE_AGY_REJECT_CONVERSATION",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Keep agy log scanning away from the real home directory.
    monkeypatch.setenv("AGY_CONFIG_DIR", str(tmp_path / "agy-config"))
    worker.POOL.reset()
    models._set_catalog_for_tests(None)
    yield
    worker.POOL.reset()
    models._set_catalog_for_tests(None)


class FakeAgy:
    def __init__(self, path: Path, log: Path) -> None:
        self.path = path
        self.log = log

    def records(self, kind: str | None = None) -> list[dict]:
        if not self.log.exists():
            return []
        rows = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in rows if kind is None or r["kind"] == kind]

    def turns(self) -> list[dict]:
        return self.records("turn")

    def starts(self) -> list[dict]:
        return self.records("start")


@pytest.fixture
def fake_agy(monkeypatch, tmp_path):
    if os.name == "nt":
        pytest.skip("the fake agy executable relies on a POSIX shebang")
    script = tmp_path / "agy"
    source = (Path(__file__).parent / "fake_agy.py").read_text(encoding="utf-8")
    script.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "fake_agy.log"
    monkeypatch.setenv("AGY_CLI_PATH", str(script))
    monkeypatch.setenv("FAKE_AGY_LOG", str(log))
    models._set_catalog_for_tests(
        models.parse_catalog(
            "gemini-3.8-flash-low\ngemini-3.8-flash-medium\ngemini-3.8-flash-high\n"
            "gemini-3.1-pro-low\ngemini-3.1-pro-high\nclaude-sonnet-4-6\n"
        )
    )
    return FakeAgy(script, log)

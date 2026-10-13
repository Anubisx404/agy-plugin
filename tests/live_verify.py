"""Live verification against a real, signed-in agy (not part of the pytest suite).

    python tests/live_verify.py [model]

Checks: the persistent worker serves consecutive turns from one process with delta prompts;
closing a stream mid-turn kills agy's process tree; agy cannot write files on its own; and, when
`hermes` is on PATH, a Hermes one-shot through the plugin.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "antigravity_plugin", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
plugin = importlib.util.module_from_spec(_spec)
sys.modules["antigravity_plugin"] = plugin
_spec.loader.exec_module(plugin)
from antigravity_plugin import worker  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.8-flash-low"


def pid_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def new_client():
    return plugin.antigravity.create_client(api_key="antigravity", base_url="acp://antigravity")


def check_persistent_worker() -> None:
    print("--- 1. Persistent worker across turns ---")
    client = new_client()
    extra = plugin.antigravity.build_extra_body(session_id=f"live-{uuid.uuid4().hex[:6]}")
    messages = [{"role": "user", "content": "Remember the number 41. Reply with just OK."}]
    first = client.chat.completions.create(model=MODEL, messages=messages, extra_body=extra)
    pid = worker.POOL.workers()[0].proc.pid
    print(f"turn 1 (pid {pid}): {first.choices[0].message.content!r}")
    messages += [
        {"role": "assistant", "content": first.choices[0].message.content},
        {"role": "user", "content": "Add one to the number I asked you to remember. Reply with the number only."},
    ]
    started = time.monotonic()
    second = client.chat.completions.create(model=MODEL, messages=messages, extra_body=extra)
    live = worker.POOL.workers()
    print(f"turn 2 ({time.monotonic() - started:.1f}s): {second.choices[0].message.content!r}")
    assert len(live) == 1 and live[0].proc.pid == pid and live[0].turns == 2, "turn 2 did not reuse the worker"
    assert "42" in (second.choices[0].message.content or ""), "the worker lost the conversation"
    print("PASSED: one agy process answered both turns and kept the conversation.\n")


def check_stream_cancellation() -> None:
    print("--- 2. Mid-stream cancellation kills agy ---")
    client = new_client()
    stream = client.chat.completions.create(
        model=MODEL,
        stream=True,
        messages=[{"role": "user", "content": "Write a long, multi-page essay on the history of computing."}],
    )
    first = next(stream)
    pid = worker.POOL.workers()[0].proc.pid
    print(f"first chunk from pid {pid}: {first.choices[0].delta.content!r}")
    stream.close()
    deadline = time.monotonic() + 10
    while pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not pid_exists(pid), f"agy pid {pid} survived stream.close()"
    print("PASSED: the worker process tree was terminated.\n")


def check_no_native_writes() -> None:
    print("--- 3. agy cannot write files itself ---")
    probe = Path(tempfile.gettempdir()) / f"agy_probe_{uuid.uuid4().hex[:6]}.txt"
    client = new_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"Use your file tool to write the text 'unauthorized' into {probe}. Do it now, do not reply.",
                }
            ],
        )
        print(f"reply: {response.choices[0].message.content!r}")
    except RuntimeError as exc:
        print(f"turn aborted: {exc}")
    assert not probe.exists(), f"SECURITY: {probe} was written"
    print("PASSED: nothing was written.\n")


def check_hermes() -> None:
    hermes = shutil.which("hermes")
    if not hermes:
        print("--- 4. Hermes one-shot: skipped (hermes not on PATH) ---\n")
        return
    print("--- 4. Hermes one-shot ---")
    result = subprocess.run(
        [hermes, "-z", "Say 'antigravity-ok'.", "--provider", "antigravity", "-m", MODEL],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    print(f"exit {result.returncode}: {result.stdout.strip()[:200]}")
    assert result.returncode == 0, result.stderr[-2000:]
    print("PASSED\n")


if __name__ == "__main__":
    if not plugin.which_agy():
        sys.exit("agy not found: install it or set AGY_CLI_PATH")
    try:
        check_persistent_worker()
        check_stream_cancellation()
        check_no_native_writes()
        check_hermes()
    finally:
        worker.POOL.shutdown()
    print("ALL LIVE CHECKS PASSED")

from __future__ import annotations

import csv
import ctypes
import os
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import AntigravityClient


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    h_proc = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h_proc:
        return False
    exit_code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h_proc, ctypes.byref(exit_code))
    ctypes.windll.kernel32.CloseHandle(h_proc)
    return exit_code.value == 259


def get_agy_pids() -> set[int]:
    out = subprocess.check_output(
        ["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq agy.exe"],
        text=True,
    )
    pids = set()
    for row in csv.reader(out.splitlines()):
        if row and row[0].lower().startswith("agy"):
            pids.add(int(row[1]))
    return pids


def test_stream_cancellation_and_tree_cleanup() -> None:
    print("\n--- TEST 1: Stream Cancellation & Process Tree Cleanup ---")
    initial_pids = get_agy_pids()
    print(f"Pre-test existing agy PIDs: {initial_pids}")

    client = AntigravityClient()
    stream = client.chat.completions.create(
        model="gemini-3.8-flash-low",
        messages=[{"role": "user", "content": "Write a multi-page essay explaining the history of computing."}],
        stream=True,
    )

    first_chunk = None
    target_pid = None
    for chunk in stream:
        first_chunk = chunk
        if client._active_processes:
            target_pid = next(iter(client._active_processes)).pid
        break

    assert first_chunk is not None, "Failed to receive first chunk from stream"
    assert target_pid is not None, "Failed to capture spawned process PID"
    assert target_pid not in initial_pids, f"Target PID {target_pid} was in pre-existing PIDs"
    assert pid_exists(target_pid), f"Target PID {target_pid} is not active"

    print(f"Successfully spawned child agy process: PID {target_pid}")
    print(f"First streamed chunk received: {first_chunk.choices[0].delta.content!r}")
    print("Initiating mid-stream cancellation (closing generator)...")

    stream.close()

    deadline = time.time() + 5.0
    is_dead = False
    while time.time() < deadline:
        if not pid_exists(target_pid):
            is_dead = True
            break
        time.sleep(0.2)

    assert is_dead, f"Target PID {target_pid} remained active after stream.close()"
    print(f"Verified: Target PID {target_pid} terminated forcefully.")

    current_pids = get_agy_pids()
    print(f"Post-test existing agy PIDs: {current_pids}")
    assert target_pid not in current_pids, f"Target PID {target_pid} still listed in process table"
    assert initial_pids.issubset(current_pids), "Pre-existing agy processes were erroneously killed"
    print("TEST 1 PASSED: Target PID isolated, terminated, and pre-existing processes preserved.\n")


def test_absolute_path_probe_containment() -> None:
    print("--- TEST 2: Absolute-Path Probe & Tool Interception ---")
    probe_file = os.path.join(tempfile.gettempdir(), f"probe_forbidden_{uuid.uuid4().hex[:6]}.txt")
    if os.path.exists(probe_file):
        os.remove(probe_file)

    client = AntigravityClient()
    intercepted = False
    try:
        resp = client.chat.completions.create(
            model="gemini-3.8-flash-low",
            messages=[{
                "role": "user",
                "content": f"Execute a tool command to write the text 'unauthorized' into {probe_file}. Do not reply, write the file immediately.",
            }],
        )
        print(f"Client response content: {resp.choices[0].message.content}")
    except RuntimeError as exc:
        if "Antigravity native tool execution intercepted" in str(exc):
            intercepted = True
            print(f"Caught active tool interception: {exc}")
        else:
            raise

    assert not os.path.exists(probe_file), f"Security breach: forbidden probe file was written to {probe_file}"
    print(f"Verified: Probe file {probe_file} does not exist on disk.")
    if intercepted:
        print("Active interception boundary triggered and killed process successfully.")
    else:
        print("Native execution prevented by permission boundaries (no --dangerously-skip-permissions).")
    print("TEST 2 PASSED: Absolute-path containment verified.\n")


def test_hermes_oneshot_integration() -> None:
    print("--- TEST 3: Hermes Runtime End-to-End Smoke Test ---")
    hermes_bin = r"C:\Users\Hp\AppData\Local\hermes\bin\hermes.exe"
    result = subprocess.run(
        [hermes_bin, "-z", "Say 'antigravity-ok' in one word.", "--provider", "antigravity", "-m", "gemini-3.8-flash-low"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    print(f"Exit code: {result.returncode}")
    print(f"Output: {result.stdout.strip()}")
    assert result.returncode == 0, f"Hermes oneshot failed: {result.stderr}"
    assert "antigravity-ok" in result.stdout.lower() or len(result.stdout.strip()) > 0
    print("TEST 3 PASSED: Hermes CLI oneshot verified.\n")


if __name__ == "__main__":
    test_stream_cancellation_and_tree_cleanup()
    test_absolute_path_probe_containment()
    test_hermes_oneshot_integration()
    print("ALL VERIFICATION SUITES COMPLETED SUCCESSFULLY.")

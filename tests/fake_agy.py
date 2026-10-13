"""A stand-in for the agy CLI that speaks its stream-json protocol, for integration tests.

Persistent mode (``--input-format stream-json``) reads ``{"event": "user", ...}`` lines and answers
each with init (first turn only) / step_update / result events, reporting cumulative usage like agy
1.2.10+. One-shot mode reads the whole prompt from stdin. Markers in the prompt pick a behaviour.
Every start and turn is appended as JSON to ``$FAKE_AGY_LOG``.
"""

import json
import os
import sys
import time
import uuid

ARGV = sys.argv[1:]
STATE = {"turn": 0, "input": 0, "output": 0}


def arg(flag):
    if flag in ARGV:
        i = ARGV.index(flag)
        return ARGV[i + 1] if i + 1 < len(ARGV) else ""
    return None


def log(record):
    path = os.environ.get("FAKE_AGY_LOG")
    if path:
        record["pid"] = os.getpid()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


CONVERSATION = arg("--conversation") or f"conv-{uuid.uuid4().hex[:8]}"
MODEL = arg("--model") or "unknown"


def run_turn(prompt):
    STATE["turn"] += 1
    turn = STATE["turn"]
    log({"kind": "turn", "turn": turn, "prompt": prompt, "model": MODEL, "effort": arg("--effort")})
    if turn == 1:
        emit({"event": "init", "init": {"conversation_id": CONVERSATION}})
    if "FAKE:DIE" in prompt:
        sys.exit(3)
    if "FAKE:HANG" in prompt:
        time.sleep(60)
    if "FAKE:QUOTA" in prompt:
        sys.stderr.write(
            "W1012 rpc.go:88] model call failed (RESOURCE_EXHAUSTED (code 429): Individual quota reached. "
            "Resets in 4h), retrying in 5s\n"
        )
        sys.stderr.flush()
        time.sleep(60)
    if "FAKE:NATIVE" in prompt:
        emit({"event": "step_update", "step_update": {"step_type": "tool", "tool_name": "run_command"}})
        time.sleep(60)
    if "FAKE:STDERR_FLOOD" in prompt:
        sys.stderr.write("x" * (2 * 1024 * 1024))
        sys.stderr.flush()
    if "FAKE:ERROR" in prompt:
        emit({"event": "result", "result": {"status": "ERROR", "error": "boom"}})
        return
    if "FAKE:EMPTY" in prompt:
        emit({"event": "result", "result": {"status": "SUCCESS"}})
        return
    if "FAKE:THINK" in prompt:
        emit({"event": "step_update", "step_update": {"step_type": "thinking", "text_delta": "pondering"}})
    if "FAKE:TOOL" in prompt:
        text = 'Let me check.\n<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    else:
        text = f"reply {turn} from {MODEL}"
    step = max(1, len(text) // 3)
    for i in range(0, len(text), step):
        emit({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": text[i : i + step]}})
        if "FAKE:SLOW" in prompt:
            time.sleep(0.3)
    STATE["input"] += len(prompt) // 4 + 100
    STATE["output"] += 10
    persistent = "--input-format" in ARGV
    usage = {
        "input_tokens": STATE["input"] if persistent else len(prompt) // 4 + 100,
        "output_tokens": STATE["output"] if persistent else 10,
        "cache_read_tokens": 0,
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    emit(
        {
            "event": "result",
            "result": {"status": "SUCCESS", "response": text, "usage": usage, "conversation_id": CONVERSATION},
        }
    )


def main():
    if ARGV[:1] == ["models"]:
        for name in (
            "gemini-3.8-flash-low",
            "gemini-3.8-flash-medium",
            "gemini-3.8-flash-high",
            "gemini-3.1-pro-low",
            "gemini-3.1-pro-high",
            "claude-opus-4-6-thinking",
            "claude-sonnet-4-6",
        ):
            print(f"{name}    description")
        return 0
    if arg("-p") == "/usage":
        print(
            json.dumps(
                {
                    "command": {
                        "name": "usage",
                        "data": {
                            "groups": [
                                {
                                    "name": "Gemini Models",
                                    "buckets": [
                                        {
                                            "id": "g5h",
                                            "window": "5h",
                                            "remaining_fraction": 0.75,
                                            "reset_time": "2026-10-13T00:00:00Z",
                                        },
                                        {
                                            "id": "gwk",
                                            "window": "weekly",
                                            "remaining_fraction": 0.5,
                                            "reset_time": "2026-10-19T00:00:00Z",
                                        },
                                    ],
                                },
                            ]
                        },
                    }
                }
            )
        )
        return 0
    if os.environ.get("FAKE_AGY_REJECT_INPUT_FORMAT") and "--input-format" in ARGV:
        sys.stderr.write("Error: unknown flag: --input-format\n")
        return 2
    if os.environ.get("FAKE_AGY_REJECT_EFFORT") and "--effort" in ARGV:
        sys.stderr.write("Error: unknown flag: --effort\n")
        return 2
    if os.environ.get("FAKE_AGY_REJECT_CONVERSATION") and "--conversation" in ARGV:
        sys.stderr.write("Error: conversation not found\n")
        return 1
    log({"kind": "start", "argv": ARGV, "cwd": os.getcwd(), "secret": os.environ.get("FAKE_SECRET_TOKEN")})
    if "--input-format" in ARGV:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("event") == "user":
                run_turn(event["message"]["content"])
        return 0
    run_turn(sys.stdin.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Changelog

## v4.0.0 — 2026-10-13

### Added
- **Persistent background agy workers.** Turns run on a long-lived `agy --input-format stream-json` process that keeps the conversation across turns; after the first turn only the new messages are sent. Workers live in a process-wide pool keyed by the Hermes session id (passed through `build_extra_body`), so they survive Hermes creating and closing request-scoped clients. Requests without a session id are matched by conversation history.
- Model or effort switches resume the same agy conversation with `--conversation <id>`; idle workers are stopped after `AGY_WORKER_IDLE_SECONDS` (default 900) and resumed the same way on the next turn. `AGY_MAX_WORKERS` (default 4) bounds concurrent workers with least-recently-used eviction; all workers stop at interpreter exit.
- Transparent recovery before any output is streamed: a worker that died while idle, a refused resume, an agy without `--input-format` (falls back to one-shot mode) or without `--effort` (flag dropped).
- `client.cancel()` for Hermes interrupts; `classify_api_error` for quota, billing, auth, context overflow, model-not-found and timeouts; `fetch_account_usage` for `/usage`; reasoning effort forwarding through `build_api_kwargs_extras`.
- Streaming with tools enabled: text streams immediately while possible `<tool_call>` openings are held back; tool calls stream as deltas; thinking output becomes `reasoning_content`.
- Tool parsing for fenced JSON, unterminated blocks and bare JSON naming a known tool; full JSON schemas in the tool catalogue; tool-set changes sent mid-session; JSON-mode instructions for `response_format`.
- Fast quota failover from agy's stderr and its own log file; credential-looking environment variables removed from agy's environment (`AGY_ENV_PASSTHROUGH`, `AGY_ENV_STRICT`, `AGY_ENV_ALLOWLIST`); `AGY_PROXY`; `AGY_DEBUG_PROMPT_DIR`.
- Awaitable completions and `async for` streams for Hermes' async auxiliary path.
- End-to-end tests against a fake agy speaking stream-json; `tests/live_verify.py` for a real agy.

### Changed
- Catalog grouped by base model with its efforts (`gemini-3.8-flash` + low/medium/high); suffixed ids still resolve. Hermes' reasoning effort maps to the nearest supported effort and is passed with `--effort`; models without effort variants get none; a model missing from the account falls back to the newest one of its family.
- Declared context window is min(native, 239,000 tokens) to stay under agy's trajectory checkpoint; `model_capabilities` now uses Hermes' `context_window` key and covers base and suffixed ids.
- Vision declared unsupported: agy receives text only, so Hermes describes images instead of their being dropped.
- Setup status reports the real sign-in state when it can be determined.

### Fixed
- stderr was never read: a verbose agy could fill the pipe and hang the turn.
- A silent agy blocked forever; reads now honour the timeout, including Hermes' `httpx.Timeout` objects (which the old `timeout > 0` comparison could not handle).
- Killing agy now terminates its whole process group on POSIX (the language-server child was left running).
- Hermes' reasoning effort never reached the client; `model_capabilities` used a key Hermes ignores.
- Empty, error or result-less turns are reported as failures instead of empty answers.

## v3.0.0 — 2026-10-12

### Added
- **Native Hermes Model-Provider Plugin Architecture:** Converted project from an external standalone HTTP proxy daemon into an in-process native Hermes model-provider plugin (`kind: model-provider`, `auth_type: external_process`, `base_url: "acp://antigravity"`).
- **Hermes Provider Router Credential Resolution:** Configured `base_url="acp://antigravity"` on `AntigravityProfile` to satisfy Hermes `_resolve_external_process_branch` credential expectations without requiring dummy API keys or remote server endpoints.
- **Stream-JSON Execution Engine:** Migrated inference runner to `agy --output-format stream-json --disable-slash-commands` for token-by-token streaming deltas, structured events, and exact usage metrics (`input_tokens`, `output_tokens`, `thinking_tokens`, `cache_read_tokens`).
- **Eliminated `--dangerously-skip-permissions`:** Completely removed dangerous auto-approval flag from command execution.
- **Isolated Temporary Working Directories:** Inference runs in dedicated temporary scratch directories (`tempfile.mkdtemp(prefix="hermes_agy_")`) wiped cleanly on turn completion or failure.
- **Active Native Tool Interception:** Subprocess output monitors step types in real time; any attempted internal AGY tool execution triggers immediate process-tree termination (`taskkill /F /T /PID`) and security exception to preserve Hermes tool orchestration authority.
- **Dual-Protocol Tool Parsing & Multi-Call Support:** Unified parser supporting multiple tool calls in a single turn, simultaneous `<tool_call>` tags and `<<<TOOL_CALLS>>>` markers, and graceful recovery from malformed JSON.
- **Enhanced Multi-Turn Conversation Pruning:** Preserves recent tool messages in full while applying head-and-tail diagnostic retention to older entries, alongside failed tool status tracking (`[Tool Result: {name} (Failed)]`).
- **Hermes External-Process Contract:** Implemented `setup_status()`, `discover_models()`, and populated canonical `model_capabilities` with exact per-model context limits (2,097,152 for Gemini Pro, 1,048,576 for Gemini Flash, 200,000 for Claude, 131,072 for GPT-OSS).
- **Verified Process Tree Cancellation:** Validated through Win32 API process handle checks that generator cancellation forcefully terminates the spawned child process and its tree without leaving zombie processes or affecting pre-existing parent sessions.
- **Comprehensive Verification Suite:** 45 passing pytest unit tests and dedicated live hardening verification (`tests/verify_cancellation_and_security.py`) covering multi-turn tool calling, early cancellation, and absolute-path probe containment.

## v2.1.0 — 2026-09-23

### Fixed
- **Subprocess Permission Prompt Blocking:** Added `--dangerously-skip-permissions` to the AGY command line so the headless background subprocess never blocks on unhandled interactive permission prompts.
- **Agent Mode Conflation:** Removed the hardcoded `--mode plan` flag from default execution to prevent AGY from hijacking conversational turns into autonomous planning artifacts. This eliminates multi-minute planning delays and restores standard, low-latency LLM responses. Custom modes remain accessible via the `AGY_MODE` environment variable.
- **Streaming Tool-Call Schema:** Fixed streaming tool-call responses to emit OpenAI-compliant `chat.completion.chunk` structures containing `delta` objects rather than non-streaming `message` objects, eliminating `ValidationError` crashes in OpenAI client libraries.
- **SSE Keepalive Heartbeats:** Added periodic `: keep-alive\n\n` comments during inference latency to maintain active TCP sockets and prevent HTTP client read and gateway timeouts.
- **Model Name & Prefix Resolution:** Added `_resolve_model()` with fuzzy and provider prefix normalization (`antigravity/`, `custom/`) so Hermes and other client model configurations resolve accurately without strict ID mismatch errors (HTTP 400).
- **Tool JSON Parsing:** Added automatic stripping of markdown code fences (```json ... ```) from model-generated `<<<TOOL_CALLS>>>` payloads to prevent `JSONDecodeError`.

## v2.0.0 — 2026-09-06

Initial open-source release.

### Features
- OpenAI-compatible `/v1/chat/completions` endpoint (streaming + non-streaming)
- `/v1/models` and `/v1/models/{id}` for model discovery
- `/health` endpoint for monitoring
- Full tool calling support via OpenAI function-calling protocol
- Prompt passed via stdin to bypass Windows 32K argument length limit
- UTF-8 encoding enforcement for cross-platform Unicode support
- ANSI escape code stripping from AGY output
- Model list caching (5 minute TTL)
- Cross-platform support (Windows, macOS, Linux)

### Tested With
- Hermes Agent (Nous Research)
- Python OpenAI SDK
- curl

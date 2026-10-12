# Changelog

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

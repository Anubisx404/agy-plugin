from __future__ import annotations

import json

from antigravity_plugin import prompt

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    },
    {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object", "properties": {}}}},
]
KNOWN = prompt.tool_names(TOOLS)


def _names(calls):
    return [c.function.name for c in calls or []]


# ── tool-call parsing ──


def test_parse_tag_format():
    cleaned, calls = prompt.parse_tool_calls(
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    )
    assert cleaned == ""
    assert _names(calls) == ["read_file"]
    assert json.loads(calls[0].function.arguments) == {"path": "a.py"}
    assert calls[0].id.startswith("call_") and len(calls[0].id) > 12


def test_parse_openai_shape_with_string_arguments():
    payload = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b"}'}}
    _, calls = prompt.parse_tool_calls(f"<tool_call>{json.dumps(payload)}</tool_call>")
    assert _names(calls) == ["read_file"]
    assert json.loads(calls[0].function.arguments) == {"path": "b"}
    assert calls[0].id != "call_1"  # model-written ids repeat across turns; always mint our own


def test_parse_marker_format_with_list_and_fences():
    text = '<<<TOOL_CALLS>>>\n```json\n[{"name": "list_files", "arguments": {}}, {"name": "read_file", "arguments": {"path": "x"}}]\n```\n<<<END_TOOL_CALLS>>>'
    cleaned, calls = prompt.parse_tool_calls(text)
    assert cleaned == ""
    assert _names(calls) == ["list_files", "read_file"]


def test_text_before_calls_kept_and_text_after_dropped():
    text = (
        "Let me look.\n"
        '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>\n'
        "Tool Result: (hallucinated)\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "y"}}</tool_call>'
    )
    cleaned, calls = prompt.parse_tool_calls(text)
    assert cleaned == "Let me look."
    assert _names(calls) == ["list_files", "read_file"]


def test_malformed_block_is_shown_as_text():
    text = "<tool_call>{invalid json}</tool_call>"
    cleaned, calls = prompt.parse_tool_calls(text)
    assert calls is None
    assert "{invalid json}" in cleaned


def test_unterminated_block_with_valid_json_is_accepted():
    _, calls = prompt.parse_tool_calls('<tool_call>{"name": "list_files", "arguments": {}}')
    assert _names(calls) == ["list_files"]


def test_bare_json_accepted_only_for_known_tools():
    _, calls = prompt.parse_tool_calls('{"name": "read_file", "arguments": {"path": "z"}}', KNOWN)
    assert _names(calls) == ["read_file"]
    cleaned, calls = prompt.parse_tool_calls('{"name": "not_a_tool", "arguments": {}}', KNOWN)
    assert calls is None and "not_a_tool" in cleaned
    cleaned, calls = prompt.parse_tool_calls('{"answer": 42}', KNOWN)
    assert calls is None and cleaned == '{"answer": 42}'


def test_plain_text_passes_through():
    cleaned, calls = prompt.parse_tool_calls("Just an answer with a < sign.")
    assert cleaned == "Just an answer with a < sign." and calls is None


def test_stream_parser_holds_back_split_tags():
    parser = prompt.ToolCallStreamParser(known_tools=KNOWN)
    events = []
    for piece in [
        "Checking ",
        "now <to",
        'ol_call>{"name": "list_',
        'files", "arguments": {}}</tool',
        "_call> trailing",
    ]:
        events += parser.feed(piece)
    events += parser.finish()
    texts = [v for k, v in events if k == "text"]
    calls = [v for k, v in events if k == "call"]
    assert "".join(texts) == "Checking now "
    assert _names(calls) == ["list_files"]
    assert parser.dropped_text.strip() == "trailing"


def test_stream_parser_releases_ordinary_text_immediately():
    parser = prompt.ToolCallStreamParser(known_tools=KNOWN)
    assert parser.feed("Hello world") == [("text", "Hello world")]
    assert parser.feed(" again <") == [("text", " again ")]
    assert parser.finish() == [("text", "<")]


def test_stream_parser_disabled_without_tools():
    parser = prompt.ToolCallStreamParser(enabled=False)
    assert parser.feed("<tool_call>{}</tool_call>") == [("text", "<tool_call>{}</tool_call>")]


def test_code_answer_is_not_held_as_bare_json():
    parser = prompt.ToolCallStreamParser(known_tools=KNOWN)
    assert parser.feed("```python\nprint(1)\n```") == [("text", "```python\nprint(1)\n```")]


# ── prompts ──


def _conversation(n_tools: int = 10):
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Start"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_err",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "missing"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_err", "content": "Error: No such file"},
    ]
    for i in range(n_tools):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": f"call_{i}", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"Result {i}: " + "X" * 900})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Final question"}]})
    return messages


def test_full_prompt_structure():
    text = prompt.format_full_prompt(_conversation(), tools=TOOLS)
    assert text.startswith("You are running in HEADLESS INFERENCE MODE")
    assert "## Available Hermes tools" in text
    assert '"required":["path"]' in text  # full JSON schema, not a lossy summary
    assert "## Hermes system instructions\nYou are Hermes." in text
    assert "[Tool Result: read_file (failed) id=call_err]" in text
    assert '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>' in text
    assert text.rstrip().split("## Latest user message\n", 1)[1].startswith("Final question")
    assert text.count("Final question") == 1


def test_full_prompt_prunes_old_tool_output_only():
    text = prompt.format_full_prompt(_conversation(), tools=TOOLS)
    assert "Result 0: " in text and "characters of tool output omitted" in text
    assert "Result 9: " + "X" * 900 in text  # recent results stay whole


def test_full_prompt_without_tools_has_no_protocol():
    text = prompt.format_full_prompt([{"role": "user", "content": "hi"}])
    assert "<tool_call>" not in text
    assert "No tools are available" in text


def test_tool_choice_hints():
    forced = {"type": "function", "function": {"name": "read_file"}}
    assert "must call the tool `read_file`" in prompt.render_tools(TOOLS, forced)
    assert "at least one tool" in prompt.render_tools(TOOLS, "required")


def test_delta_prompt_renders_only_new_messages():
    context = [
        {"role": "user", "content": "Start"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
    ]
    delta = [{"role": "tool", "tool_call_id": "call_a", "content": "file body"}]
    text = prompt.format_delta_prompt(delta, context_messages=context, tools=TOOLS)
    assert "Start" not in text and "HEADLESS INFERENCE MODE" not in text
    assert "[Tool Result: read_file id=call_a]\nfile body" in text
    assert "## Next step" in text
    assert prompt.DELTA_REMINDER in text


def test_delta_prompt_announces_tool_changes():
    text = prompt.format_delta_prompt([{"role": "user", "content": "go"}], tools=TOOLS, tools_changed=True)
    assert "## Tool list updated" in text and "### read_file" in text


def test_images_become_explicit_placeholders():
    content = [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:..."}}]
    assert "cannot view images" in prompt.render_content(content)


# ── continuity ──


def test_continuation_delta_requires_strict_prefix():
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    recorded = prompt.history_keys(history)
    incoming = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},  # list form == string form
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert prompt.continuation_delta(recorded, incoming) == [{"role": "user", "content": "c"}]
    assert prompt.continuation_delta(recorded, incoming[:2]) is None  # not an extension
    edited = [{"role": "user", "content": "a2"}, *incoming[1:]]
    assert prompt.continuation_delta(recorded, edited) is None
    assert prompt.continuation_delta((), incoming) is None


def test_message_keys_ignore_argument_formatting():
    a = {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": '{"x": 1, "y": 2}'}}]}
    b = {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": {"y": 2, "x": 1}}}]}
    assert prompt.normalize_message(a) == prompt.normalize_message(b)
    assert prompt.message_key(a) == prompt.message_key(b) and len(prompt.message_key(a)) == 16


def test_reply_matches():
    reply = prompt.Reply(content="Hello there", tool_names=())
    assert reply.matches({"role": "assistant", "content": "Hello there"})
    assert reply.matches({"role": "assistant", "content": " Hello   there "})
    assert not reply.matches({"role": "assistant", "content": "Something else"})
    assert not reply.matches({"role": "user", "content": "Hello there"})
    tool_reply = prompt.Reply(content="", tool_names=("read_file",))
    assert tool_reply.matches(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c", "function": {"name": "read_file", "arguments": "{}"}}],
        }
    )
    assert not tool_reply.matches({"role": "assistant", "content": "x"})


def test_tools_signature_tracks_changes():
    assert prompt.tools_signature(TOOLS) == prompt.tools_signature(list(TOOLS))
    assert prompt.tools_signature(TOOLS) != prompt.tools_signature(TOOLS[:1])
    assert prompt.tools_signature(TOOLS, "required") != prompt.tools_signature(TOOLS)

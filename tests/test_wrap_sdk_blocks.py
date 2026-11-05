"""A tool loop written the way the SDK's own docs write it puts the response's content blocks, SDK objects, back into
`messages`: `{"role": "assistant", "content": r.content}`. The raw SDK accepts them and dumps them as it sends them. The
wrapper must accept them on every path, send the same bytes as the raw SDK, and still carry its levers out on them.
(Before 0.3.2 the equivalence check `json.dumps`-ed the request and raised `TypeError: Object of type ToolUseBlock is not
JSON serializable` on the second call of any wrapped tool loop.)"""
import asyncio
import io
import json

import anthropic
import httpx2 as httpx
import pytest

from recant import quality, wrap
from test_wrap_paths import TENANT, apply_breakpoint, records, sse

TOOLS = [{"name": "search", "description": "search the docs", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}]

TOOL_TURN = {"id": "msg_0", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "stop_reason": "tool_use",
             "stop_sequence": None,
             "content": [{"type": "thinking", "thinking": "look it up", "signature": "sig"},
                         {"type": "text", "text": "Searching.", "citations": None},
                         {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "price"}}],
             "usage": {"input_tokens": 10, "output_tokens": 5}}
FINAL = {**TOOL_TURN, "id": "msg_1", "stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}


class API:
    """Answers the first request with a tool-use turn, every later one with a final answer; keeps every body's bytes."""

    def __init__(self):
        self.raw: list[bytes] = []

    def __call__(self, request):
        self.raw.append(request.content)
        body = json.loads(request.content)
        if not any(m["role"] == "assistant" for m in body["messages"]):
            return httpx.Response(200, json=TOOL_TURN)
        if body.get("stream"):
            return httpx.Response(200, text=sse(body["model"]), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=FINAL)


def sdk(api, aio):
    if aio:
        return anthropic.AsyncAnthropic(api_key="x", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)))
    return anthropic.Anthropic(api_key="x", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(api)))


def base():
    return dict(model="claude-sonnet-5-5", max_tokens=512, tools=TOOLS, system="Be brief. " * 10,
                messages=[{"role": "user", "content": "What does the X2 cost?"}])


async def loop(client, path, aio):
    """The textbook loop: call, append the SDK's own content blocks, answer the tool, call again on `path`."""
    req = base()
    first = (await client.messages.create(**req)) if aio else client.messages.create(**req)
    assert first.stop_reason == "tool_use" and not isinstance(first.content[0], dict)      # SDK objects, not dicts
    req["messages"] += [{"role": "assistant", "content": first.content},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": first.content[-1].id, "content": "$1,240"}]}]
    if path == "create":
        r = (await client.messages.create(**req)) if aio else client.messages.create(**req)
        return r.content[-1].text
    if path == "create_stream":
        if aio:
            s = await client.messages.create(**req, stream=True)
            async with s:
                return "".join([e.delta.text async for e in s if e.type == "content_block_delta"])
        with client.messages.create(**req, stream=True) as s:
            return "".join(e.delta.text for e in s if e.type == "content_block_delta")
    if aio:
        async with client.messages.stream(**req) as s:
            return (await s.get_final_message()).content[-1].text
    with client.messages.stream(**req) as s:
        return s.get_final_message().content[-1].text


def drive(client, path, aio, wrapped):
    async def go():
        if not wrapped:
            return await loop(client, path, aio)
        if aio:
            async with client.run("agent", session="s"):
                return await loop(client, path, aio)
        with client.run("agent", session="s"):
            return await loop(client, path, aio)
    return asyncio.run(go())


PATHS = pytest.mark.parametrize("path", ["create", "create_stream", "stream_helper"])
AIO = pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])


@AIO
@PATHS
def test_the_wrapped_tool_loop_sends_the_raw_sdks_bytes(layout, aio, path):
    raw, wrapped = API(), API()
    assert drive(sdk(raw, aio), path, aio, wrapped=False) == "ok"
    client = wrap(sdk(wrapped, aio), store=layout, tenant=TENANT)
    assert drive(client, path, aio, wrapped=True) == "ok"
    assert wrapped.raw == raw.raw                                    # byte for byte, both calls, with no lever in force
    assert client.last_equivalent is True
    recs = records(layout)
    assert len(recs) == 2 and all(r["error"] is None for r in recs)


@AIO
@PATHS
def test_the_levers_are_carried_out_on_a_request_holding_sdk_blocks(layout, aio, path):
    raw, wrapped = API(), API()
    drive(sdk(raw, aio), path, aio, wrapped=False)
    apply_breakpoint(layout)
    client = wrap(sdk(wrapped, aio), store=layout, tenant=TENANT)
    assert drive(client, path, aio, wrapped=True) == "ok"
    assert client.last_equivalent is True and client.last_sent_markers                      # markers placed
    sent, want = json.loads(wrapped.raw[-1]), json.loads(raw.raw[-1])
    assert sent != want and quality.strip_cache_markers(sent) == quality.strip_cache_markers(want)   # only the markers differ
    counts = [r for r in records(layout, "run_finished")][-1].get("counts") or {}
    assert counts == {"levers_equivalent": 2}


def test_a_value_with_no_plain_form_is_sent_as_built_untouched_and_still_metered(layout):
    """A file object as a base64 source (the SDK reads it), with the messages a generator: walked once, sent whole,
    no lever applied though one is in force, the call metered with the object written as its type."""
    def req(blob):
        msgs = ({"role": "user", "content": [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": blob}},
                                             {"type": "text", "text": "summarize"}]} for _ in range(1))
        return dict(model="claude-sonnet-5-5", max_tokens=64, system="Be brief. " * 10, messages=msgs)
    raw, wrapped = API(), API()
    sdk(raw, False).messages.create(**req(io.BytesIO(b"%PDF-1.4 tiny")))
    apply_breakpoint(layout)
    client = wrap(sdk(wrapped, False), store=layout, tenant=TENANT)
    with client.run("agent", session="s"):
        client.messages.create(**req(io.BytesIO(b"%PDF-1.4 tiny")))
    assert wrapped.raw == raw.raw and client.last_sent_markers == []
    rec, = records(layout)
    assert rec["error"] is None
    counts = [r for r in records(layout, "run_finished")][-1].get("counts") or {}
    assert counts.get("request_opaque") == 1

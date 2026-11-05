"""live: one real call per wrapper path — sync and async, `create`, `create(stream=True)`, the `stream(...)` helper — on
a model that thinks by default (Opus 5.5), with the breakpoint lever applied. Each call's record must equal the final
message's usage as the API reported it, thinking included, and the request must have carried the cache markers.
Runs only with RECANT_LIVE_TESTS=1 and a key in ANTHROPIC_API_KEY (the fence keeps that one); six short calls, about
$0.05."""
import asyncio
import json
import os

import pytest

from recant import wrap
from recant.log import read_run
from recant_fixtures import TENANT

pytestmark = pytest.mark.skipif(os.environ.get("RECANT_LIVE_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
                                reason="live: RECANT_LIVE_TESTS=1 and ANTHROPIC_API_KEY")

MODEL = "claude-opus-5-5"
SYSTEM = "You answer arithmetic puzzles with the number only. " + " ".join(f"Rule {i}: show no working in the reply." for i in range(120))
Q = "Find the smallest positive integer that leaves remainder 2 when divided by 3, 3 when divided by 5, and 4 when divided by 7."


def want(u) -> dict:
    d = u.output_tokens_details
    return {"input": u.input_tokens, "output": u.output_tokens, "cache_read": u.cache_read_input_tokens or 0,
            "cache_write": u.cache_creation_input_tokens or 0, "reasoning": (d.thinking_tokens or 0) if d else 0,
            "cache_write_long": (u.cache_creation.ephemeral_1h_input_tokens or 0) if u.cache_creation else 0}


def setup(layout, aio):
    import anthropic
    d = layout.costshape_dir(TENANT)
    d.mkdir(parents=True, exist_ok=True)
    (d / "overlay.json").write_text(json.dumps({"breakpoint.add": {"kind": "applied", "action": {}}}))
    return wrap(anthropic.AsyncAnthropic() if aio else anthropic.Anthropic(), store=layout, tenant=TENANT)


def final_usage(client, aio, path):
    req = dict(model=MODEL, max_tokens=4000, system=SYSTEM, messages=[{"role": "user", "content": Q}])
    if not aio:
        with client.run("agent", session=path):
            if path == "create":
                return client.messages.create(**req).usage
            if path == "create_stream":
                with client.messages.create(**req, stream=True) as s:
                    usage = None
                    for e in s:                                   # the host accumulates as it likes; Recant does its own
                        if e.type == "message_start":
                            usage = e.message.usage
                        elif e.type == "message_delta":
                            usage.output_tokens = e.usage.output_tokens
                            usage.output_tokens_details = e.usage.output_tokens_details
                    return usage
            with client.messages.stream(**req) as s:
                return s.get_final_message().usage

    async def go():
        async with client.run("agent", session=path):
            if path == "create":
                return (await client.messages.create(**req)).usage
            if path == "create_stream":
                s = await client.messages.create(**req, stream=True)
                usage = None
                async with s:
                    async for e in s:
                        if e.type == "message_start":
                            usage = e.message.usage
                        elif e.type == "message_delta":
                            usage.output_tokens = e.usage.output_tokens
                            usage.output_tokens_details = e.usage.output_tokens_details
                return usage
            async with client.messages.stream(**req) as s:
                return (await s.get_final_message()).usage
    return asyncio.run(go())


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("path", ["create", "create_stream", "stream_helper"])
def test_a_live_call_on_each_path_is_recorded_as_the_api_reported_it(layout, aio, path):
    client = setup(layout, aio)
    usage = final_usage(client, aio, path)
    assert client.last_sent_markers == ["system.0 (5m)", "messages.0.0 (5m)"] and client.last_equivalent is True
    rec, = [r for p in layout.log_dir(TENANT).glob("*.jsonl") for r in read_run(p, typed=False) if r["kind"] == "model_call"]
    print(path, "async" if aio else "sync", rec["usage"])
    assert rec["usage"] == want(usage) and rec["stop_reason"] == "end_turn" and rec["error"] is None
    assert rec["usage"]["reasoning"] > 0                                          # the model thought, and it was recorded
    assert rec["usage"]["cache_write"] + rec["usage"]["cache_read"] > 0           # the markers reached the API and cached

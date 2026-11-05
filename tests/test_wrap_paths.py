"""the wrapper on every path the SDK offers — sync and async, `create`, `create(stream=True)` and the `stream(...)`
helper — with the same lever application and the same record: usage from the final message (thinking tokens as
reasoning, the 1-hour cache writes as long), the run's session, equivalence counted. The real SDK runs against a
mock transport, so its own SSE parsing and stream accumulation are what the wrapper reads."""
import asyncio
import json

import anthropic
import httpx2 as httpx           # the SDK's own HTTP library (anthropic 1.x)
import pytest

from recant import costshape as cs, wrap
from recant.log import read_run
from recant_fixtures import TENANT

USAGE_START = {"input_tokens": 12, "output_tokens": 1, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 40,
               "cache_creation": {"ephemeral_5m_input_tokens": 30, "ephemeral_1h_input_tokens": 10}}
USAGE_END = {"output_tokens": 250, "output_tokens_details": {"thinking_tokens": 180}}
WANT = {"input": 12, "output": 250, "cache_read": 900, "cache_write": 40, "reasoning": 180, "cache_write_long": 10}


def message(model):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "stop_reason": "end_turn", "stop_sequence": None,
            "content": [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "ok"}],
            "usage": {**USAGE_START, **USAGE_END}}


def sse(model, *, stop=True):
    start = {**message(model), "content": [], "stop_reason": None, "usage": USAGE_START}
    evs = [("message_start", {"type": "message_start", "message": start}),
           ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
           ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
           ("content_block_stop", {"type": "content_block_stop", "index": 0})]
    if stop:
        evs += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": USAGE_END}),
                ("message_stop", {"type": "message_stop"})]
    return "".join(f"event: {k}\ndata: {json.dumps(v)}\n\n" for k, v in evs)


class API:
    """The Messages API as a transport: records each request body, answers JSON or SSE as the body asks."""

    def __init__(self, status=200, stop=True):
        self.bodies, self.status, self.stop = [], status, stop

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"type": "error", "error": {"type": "invalid_request_error", "message": "no"}})
        if body.get("stream"):
            return httpx.Response(200, text=sse(body["model"], stop=self.stop), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=message(body["model"]))


def sdk(api, *, aio=False):
    if aio:
        return anthropic.AsyncAnthropic(api_key="x", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)))
    return anthropic.Anthropic(api_key="x", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(api)))


def apply_breakpoint(layout):
    d = layout.costshape_dir(TENANT)
    d.mkdir(parents=True, exist_ok=True)
    (d / "overlay.json").write_text(json.dumps({"breakpoint.add": {"kind": "applied", "action": {}}}))


def records(layout, kind="model_call"):
    return [r for p in sorted(layout.log_dir(TENANT).glob("*.jsonl")) for r in read_run(p, typed=False) if r["kind"] == kind]


REQ = dict(model="claude-sonnet-5-5", max_tokens=1024, system="You are terse. " * 20,
           messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}])


def call_sync(client, path):
    if path == "create":
        return client.messages.create(**REQ).content[-1].text
    if path == "create_stream":
        with client.messages.create(**REQ, stream=True) as s:
            return "".join(e.delta.text for e in s if e.type == "content_block_delta")
    with client.messages.stream(**REQ) as s:                     # the helper
        return "".join(s.text_stream) + "|" + s.get_final_message().content[-1].text


async def call_async(client, path):
    if path == "create":
        return (await client.messages.create(**REQ)).content[-1].text
    if path == "create_stream":
        s = await client.messages.create(**REQ, stream=True)
        async with s:
            return "".join([e.delta.text async for e in s if e.type == "content_block_delta"])
    async with client.messages.stream(**REQ) as s:
        return "".join([t async for t in s.text_stream]) + "|" + (await s.get_final_message()).content[-1].text


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("path", ["create", "create_stream", "stream_helper"])
def test_every_path_carries_the_levers_out_and_records_the_final_usage(layout, aio, path):
    api = API()
    apply_breakpoint(layout)
    client = wrap(sdk(api, aio=aio), store=layout, tenant=TENANT)
    assert client.is_async is aio
    if aio:
        async def go():
            async with client.run("agent", session="s1"):
                return await call_async(client, path)
        text = asyncio.run(go())
    else:
        with client.run("agent", session="s1"):
            text = call_sync(client, path)
    assert text in ("ok", "ok|ok")                                                     # the host reads the SDK's own objects
    sent = api.bodies[-1]
    assert sent["system"][-1]["cache_control"] == {"type": "ephemeral"} and sent["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert client.last_equivalent is True and client.last_sent_markers == ["system.0 (5m)", "messages.2.0 (5m)"]
    rec, = records(layout)
    assert rec["usage"] == WANT and rec["stop_reason"] == "end_turn" and rec["session"] == "s1" and rec["model_id"] == "claude-sonnet-5-5"
    assert rec["error"] is None and rec["latency_ms"] >= 0
    counts = [r for r in records(layout, "run_finished")][-1].get("counts") or {}
    assert counts == {"levers_equivalent": 1}


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
def test_a_stream_closed_early_is_recorded_once_with_what_it_reported(layout, aio):
    client = wrap(sdk(API(stop=False), aio=aio), store=layout, tenant=TENANT)
    if aio:
        async def go():
            async with client.run("agent", session="s"):
                s = await client.messages.create(**REQ, stream=True)
                async for _ in s:
                    break
                await s.close()
        asyncio.run(go())
    else:
        with client.run("agent", session="s"):
            s = client.messages.create(**REQ, stream=True)
            for _ in s:
                break
            s.close()
            s.close()                                                                     # a second close records nothing more
    rec, = records(layout)
    assert rec["stop_reason"] is None and rec["usage"]["cache_read"] == 900 and rec["usage"]["output"] == 1


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("path", ["create", "create_stream", "stream_helper"])
def test_a_refused_request_is_recorded_as_an_error_and_raised(layout, aio, path):
    client = wrap(sdk(API(status=400), aio=aio), store=layout, tenant=TENANT)
    with pytest.raises(anthropic.BadRequestError):
        if aio:
            asyncio.run(call_async(client, path))
        else:
            call_sync(client, path)
    rec, = records(layout)
    assert rec["error"] == "BadRequestError" and rec["usage"]["input"] == 0


def test_concurrent_async_runs_each_keep_their_own_log_and_session(layout):
    client = wrap(sdk(API(), aio=True), store=layout, tenant=TENANT)

    async def task(i):
        async with client.run("agent", session=f"task-{i}"):
            for _ in range(3):
                await client.messages.create(**REQ)
                await asyncio.sleep(0)                                                    # interleave the tasks' calls

    async def go():
        await asyncio.gather(*(task(i) for i in range(4)))
    asyncio.run(go())
    files = sorted(layout.log_dir(TENANT).glob("*.jsonl"))
    by_file = [{r["session"] for r in read_run(p, typed=False) if r["kind"] == "model_call"} for p in files]
    assert len(files) == 4 and sorted(len(s) for s in by_file) == [1, 1, 1, 1]          # one run per task, its own calls only
    assert sorted(s.pop() for s in by_file) == [f"task-{i}" for i in range(4)]
    assert all(r["extends_previous"] in (None, False) for r in records(layout))          # REQ repeats: never an append


def test_thinking_reaches_the_cost_report(layout):
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    for i in range(3):
        with client.run("agent", session=f"s{i}"):
            client.messages.create(**REQ)
    host = cs.HostShape(agent_jobs=("agent",), agent_run_jobs=("agent",), carries_out=("breakpoint.add",))
    text = cs.report(layout, TENANT, host, since="30d").text([], [])
    assert "(reasoning 72%)" in text                                                      # 180 of 250 output tokens
    assert "output 250/call (reasoning 72%)" in text


def test_the_thinking_setting_is_recorded_in_recants_terms(layout):
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    with client.run("agent", session="s"):
        client.messages.create(**REQ, thinking={"type": "disabled"}, output_config={"effort": "low"})
        client.messages.create(**REQ, thinking={"type": "adaptive"})
        client.messages.create(**REQ)
    fps = [r["setting_fp"] for r in records(layout)]
    assert len(set(fps)) == 3                                                             # each setting its own window


def test_the_async_streaming_quickstart_meters_concurrent_tasks_and_carries_the_lever_out(layout, monkeypatch):
    import importlib.util
    from pathlib import Path
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("cost_quickstart_async", examples / "cost_quickstart_async.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    api, store = API(), str(layout.root)
    mod.main([store, "--tasks", "3", "--parallel", "3"], sdk=sdk(api, aio=True))
    assert len(api.bodies) == 18 and all(b["stream"] for b in api.bodies) and "cache_control" not in json.dumps(api.bodies)
    d = layout.costshape_dir("demo")
    (d / "overlay.json").write_text(json.dumps({"breakpoint.add": {"kind": "applied", "action": {}}}))
    mod.main([store, "--tasks", "2", "--parallel", "2"], sdk=sdk(api, aio=True))
    later = api.bodies[18:]
    assert len(later) == 12 and all(b["system"][-1]["cache_control"] and b["messages"][-1]["content"][-1]["cache_control"] for b in later)
    files = sorted(layout.log_dir("demo").glob("*.jsonl"))
    calls = [r for p in files for r in read_run(p, typed=False) if r["kind"] == "model_call"]
    assert len(files) == 5 and len(calls) == 30 and {r["session"] for r in calls} == {f"task-{i}" for i in range(5)}
    assert all(r["usage"] == WANT for r in calls)
    by_task = {}
    for r in calls:
        by_task.setdefault(r["session"], []).append(r["extends_previous"])
    assert all(v == [None, True, True, True, True, True] for v in by_task.values())      # each task's requests only append


class Dropped(API):
    """A dropped stream: the first events arrive, then the connection closes mid-body."""

    def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        head = sse(body["model"], stop=False).encode()

        class Sync(httpx.SyncByteStream):
            def __iter__(self):
                yield head
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

        class Async(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield head
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Async() if self.aio else Sync())


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("path", ["create_stream", "stream_helper"])
def test_a_stream_dropped_mid_body_is_recorded_with_its_error_and_what_it_reported(layout, aio, path):
    api = Dropped()
    api.aio = aio
    client = wrap(sdk(api, aio=aio), store=layout, tenant=TENANT)
    with pytest.raises(Exception, match="peer closed"):
        if aio:
            asyncio.run(call_async(client, path))
        else:
            call_sync(client, path)
    rec, = records(layout)
    assert rec["error"] == "RemoteProtocolError" and rec["stop_reason"] is None
    assert rec["usage"]["input"] == 12 and rec["usage"]["cache_read"] == 900                # message_start had arrived

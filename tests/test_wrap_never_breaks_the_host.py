"""Recant's own failure inside a wrapped call never reaches the host. If preparing the request, recording the call or
reading a stream event raises, the host's request goes out as built, the host gets its response and every event, and
the failure is counted in the friction log (`wrapper_errors`), which the feedback form pre-fills. (In 0.3.1 a crash
in the equivalence check raised into the host's tool loop, and no count anywhere showed it.)"""
import asyncio
import json

import pytest

from recant import feedback, friction, levers, wrap
from recant import metering
from test_wrap_paths import API, REQ, TENANT, apply_breakpoint, call_async, call_sync, records, sdk


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RECANT_HOME", str(tmp_path / "home"))


def boom(*a, **k):
    raise RuntimeError("recant's own bug")


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("path", ["create", "create_stream", "stream_helper"])
def test_a_failure_preparing_the_call_sends_the_host_s_request_as_built(layout, monkeypatch, aio, path):
    raw_api, api = API(), API()
    want = call_sync(sdk(raw_api), path) if not aio else asyncio.run(call_async(sdk(raw_api, aio=True), path))
    apply_breakpoint(layout)
    monkeypatch.setattr(levers, "apply", boom)
    client = wrap(sdk(api, aio=aio), store=layout, tenant=TENANT)
    if aio:
        async def go():
            async with client.run("agent", session="s"):
                return await call_async(client, path)
        got = asyncio.run(go())
    else:
        with client.run("agent", session="s"):
            got = call_sync(client, path)
    assert got == want and api.bodies == raw_api.bodies                  # the host's request, byte for byte as parsed
    assert records(layout) == []                                         # nothing metered for the broken call
    s = friction.summary()
    assert s["wrapper_errors"] == {"RuntimeError in recant.wrap._prepare": 1}


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("path", ["create", "create_stream", "stream_helper"])
def test_a_failure_recording_the_call_still_returns_the_response(layout, monkeypatch, aio, path):
    monkeypatch.setattr(metering.Meter, "record", boom)
    client = wrap(sdk(API(), aio=aio), store=layout, tenant=TENANT)
    if aio:
        async def go():
            async with client.run("agent", session="s"):
                return await call_async(client, path)
        got = asyncio.run(go())
    else:
        with client.run("agent", session="s"):
            got = call_sync(client, path)
    assert got in ("ok", "ok|ok")
    assert sum(friction.summary()["wrapper_errors"].values()) == 1


def test_a_failure_reading_a_stream_event_leaves_the_host_every_event(layout, monkeypatch):
    import sys
    monkeypatch.setattr(sys.modules["recant.wrap"]._Events, "see", boom)      # the module, not recant.wrap the function
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    with client.run("agent", session="s"):
        with client.messages.create(**REQ, stream=True) as s:
            kinds = [e.type for e in s]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    assert friction.summary()["wrapper_errors"] == {"RuntimeError in recant.wrap._see": 1}


def test_the_form_pre_fills_the_wrapper_s_errors_and_the_bundle_counts_them(layout, monkeypatch):
    monkeypatch.setattr(levers, "apply", boom)
    apply_breakpoint(layout)
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    client.messages.create(**REQ)
    asked = []
    answers = feedback.ask_form(ask=lambda q: (asked.append(q), "")[1], say=lambda *_: None)
    assert "RuntimeError in recant.wrap._prepare ×1 inside a wrapped call" in answers["errors_hit"]
    bundle, _ = feedback.build(bench_dirs=[], eval_dirs=[])
    assert bundle["friction"]["wrapper_errors"] == {"RuntimeError in recant.wrap._prepare": 1}


def test_an_error_inside_a_comprehension_keeps_its_class_in_the_bundle():
    assert feedback.error_name("TypeError in recant.cli.<dictcomp>") == "TypeError in recant.cli.<dictcomp>"
    assert feedback.error_name("TypeError in recant.cli.<my-thing>") == "HostError"


def test_after_the_walk_the_fallback_sends_what_the_walk_left_of_an_iterator(layout, monkeypatch):
    """Messages as an iterator, a failure after the walk consumed it. The host's message still goes."""
    raw_api, api = API(), API()
    sdk(raw_api).messages.create(**dict(REQ, messages=iter(REQ["messages"])))
    apply_breakpoint(layout)
    monkeypatch.setattr(levers, "apply", boom)
    client = wrap(sdk(api), store=layout, tenant=TENANT)
    client.messages.create(**dict(REQ, messages=iter(REQ["messages"])))
    assert api.bodies == raw_api.bodies and api.bodies[-1]["messages"][-1]["content"] == "more"


@pytest.mark.parametrize("path", ["create", "create_stream", "stream_helper"])
def test_a_failure_reading_the_response_s_usage_still_returns_it(layout, monkeypatch, path):
    import sys
    monkeypatch.setattr(sys.modules["recant.wrap"], "_usage", boom)
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    with client.run("agent", session="s"):
        assert call_sync(client, path) in ("ok", "ok|ok")
    assert sum(friction.summary()["wrapper_errors"].values()) >= 1


def test_a_run_whose_meter_cannot_open_still_sends_its_calls(layout, monkeypatch):
    import sys
    w = sys.modules["recant.wrap"]
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    monkeypatch.setattr(w, "Meter", boom)
    with client.run("agent", session="s"):
        assert client.messages.create(**REQ).content[-1].text == "ok"
    assert sum(friction.summary()["wrapper_errors"].values()) == 2           # the run's meter, then its call


def test_a_switch_that_cannot_be_read_is_a_pass_through(layout, monkeypatch):
    from recant import switch
    api = API()
    client = wrap(sdk(api), store=layout, tenant=TENANT)
    monkeypatch.setattr(switch, "state", boom)
    assert client.messages.create(**REQ).content[-1].text == "ok"
    assert records(layout) == [] and friction.summary()["wrapper_errors"] == {"RuntimeError in recant.wrap.mode": 1}


def test_a_value_the_walk_does_not_read_is_left_as_built_and_never_consumed():
    """Deciding what to read checks types only: an iterator and a value with a model_dump of its own are noted, never
    read, so the request goes as the host's own objects, with no lever."""
    import sys
    readable = sys.modules["recant.wrap"]._readable

    class BadDump:
        def model_dump(self, **kw):
            raise RuntimeError("cannot dump")
    it = iter([{"role": "user", "content": "hello"}])
    opaque: list = []
    readable({"messages": it, "system": BadDump()}, opaque)
    assert sorted(opaque) == ["BadDump", "list_iterator"]
    assert next(it) == {"role": "user", "content": "hello"}                    # never consumed

def test_an_iterator_inside_a_message_is_read_once_and_sent_whole(layout):
    raw_api, api = API(), API()
    msgs = lambda: [{"role": "user", "content": (b for b in [{"type": "text", "text": "more"}])}]       # noqa: E731
    sdk(raw_api).messages.create(**dict(REQ, messages=msgs()))
    wrap(sdk(api), store=layout, tenant=TENANT).messages.create(**dict(REQ, messages=msgs()))
    assert api.bodies == raw_api.bodies and api.bodies[-1]["messages"][0]["content"][0]["text"] == "more"


def test_a_value_whose_model_dump_property_raises_does_not_lose_its_siblings(layout):
    """A list subclass the SDK serializes as a list, whose `model_dump` attribute raises when read, beside an iterator
    of messages: the messages still go (in an earlier 0.3.2 candidate they went as [])."""
    class System(list):
        @property
        def model_dump(self):
            raise RuntimeError("unavailable")
    raw_api, api = API(), API()
    req = lambda: dict(model="claude-sonnet-5-5", max_tokens=64, messages=iter([{"role": "user", "content": "hello"}]),   # noqa: E731
                       system=System([{"type": "text", "text": "Be brief."}]))
    sdk(raw_api).messages.create(**req())
    wrap(sdk(api), store=layout, tenant=TENANT).messages.create(**req())
    assert api.bodies == raw_api.bodies and api.bodies[-1]["messages"] == [{"role": "user", "content": "hello"}]


def test_a_failure_to_report_a_failure_is_not_the_host_s_error(layout, monkeypatch):
    def unreadable(exc):
        raise PermissionError("the package's folder is gone")
    monkeypatch.setattr(friction, "_where", unreadable)
    monkeypatch.setattr(levers, "apply", boom)
    apply_breakpoint(layout)
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    assert client.messages.create(**REQ).content[-1].text == "ok"


def test_only_pydantic_models_are_dumped_as_the_sdk_dumps_them(layout):
    """A list subclass with a model_dump of its own is sent as the list it is, as the SDK sends it (an earlier 0.3.2
    candidate called its model_dump and sent `system: []`)."""
    class Blocks(list):
        def model_dump(self, **kwargs):
            return []
    raw_api, api = API(), API()
    req = lambda: dict(model="claude-sonnet-5-5", max_tokens=64, messages=[{"role": "user", "content": "hi"}],   # noqa: E731
                       system=Blocks([{"type": "text", "text": "Be brief."}]))
    sdk(raw_api).messages.create(**req())
    wrap(sdk(api), store=layout, tenant=TENANT).messages.create(**req())
    assert api.bodies == raw_api.bodies and api.bodies[-1]["system"] == [{"type": "text", "text": "Be brief."}]


def _same_bytes(layout, make):
    raw_api, api = API(), API()
    sdk(raw_api).messages.create(**make())
    wrap(sdk(api), store=layout, tenant=TENANT).messages.create(**make())
    assert api.bodies == raw_api.bodies
    return api.bodies[-1]


def test_an_iterator_with_a_read_attribute_is_still_sent_as_a_list(layout):

    class Msgs:
        read = None

        def __init__(self, xs):
            self.it = iter(xs)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.it)
    three = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    body = _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8, messages=Msgs(three)))
    assert len(body["messages"]) == 3


def test_the_host_s_iterator_raising_is_the_host_s_error_as_with_the_raw_sdk(layout):
    def broken():
        yield {"role": "user", "content": "a"}
        raise ValueError("the host's own iterator")
    with pytest.raises(ValueError, match="the host's own iterator"):
        sdk(API()).messages.create(model="claude-sonnet-5-5", max_tokens=8, messages=broken())
    api = API()
    with pytest.raises(ValueError, match="the host's own iterator"):
        wrap(sdk(api), store=layout, tenant=TENANT).messages.create(model="claude-sonnet-5-5", max_tokens=8, messages=broken())
    assert api.bodies == []                                               # nothing truncated was sent


def test_a_model_that_is_also_a_sequence_goes_as_the_sdk_sends_it(layout):
    from collections.abc import Sequence
    import pydantic

    class Both(pydantic.BaseModel, Sequence):
        text: str

        def __getitem__(self, i):
            return [("text", self.text)][i]

        def __len__(self):
            return 1
    _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": "a"}],
                                     extra_body={"x": Both(text="keep")}))


def test_a_path_that_is_also_a_sequence_is_a_file_input_as_with_the_raw_sdk(layout, tmp_path):
    import pathlib
    from collections.abc import Sequence
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 tiny")

    class SeqPath(type(pathlib.Path()), Sequence):
        def __getitem__(self, i):
            return ["sequence item"][i]

        def __len__(self):
            return 1
    body = _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": SeqPath(f)}}]}]))
    assert body["messages"][0]["content"][0]["source"]["data"] != ["sequence item"]



def test_one_iterator_shared_by_two_arguments_goes_as_the_sdk_reads_it(layout):
    """The host passes the same iterator twice; which argument gets its items is the SDK's order, not Recant's."""
    def make():
        it = iter([{"role": "user", "content": "a"}])
        return dict(model="claude-sonnet-5-5", max_tokens=8, extra_body={"x": it}, messages=it)
    raw_api, api = API(), API()
    try:
        sdk(raw_api).messages.create(**make())
    except Exception as exc:                                    # noqa: BLE001 — whatever the SDK does, the wrapper does too
        with pytest.raises(type(exc)):
            wrap(sdk(api), store=layout, tenant=TENANT).messages.create(**make())
        return
    wrap(sdk(api), store=layout, tenant=TENANT).messages.create(**make())
    assert api.bodies == raw_api.bodies


def test_an_omission_sentinel_is_left_to_the_sdk(layout):
    anthropic = pytest.importorskip("anthropic")

    class Skip(anthropic.NotGiven, list):
        pass
    _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": "a"}],
                                     extra_body={"x": Skip([1, 2])}))


def test_a_plain_tool_loop_request_still_gets_its_levers(layout):
    """The rule leaves exotic values alone; the ordinary case, plain data and the SDK's own blocks, is still walked."""
    import sys
    w = sys.modules["recant.wrap"]
    block = pytest.importorskip("anthropic").types.TextBlock(type="text", text="hi")
    req = {"messages": [{"role": "assistant", "content": [block]}], "max_tokens": 8}
    opaque: list = []
    w._readable(req, opaque)
    assert opaque == [] and w._plain(req) == {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}],
                                              "max_tokens": 8}


def test_a_list_subclass_that_drains_itself_is_never_iterated(layout):
    class Drain(list):
        def __iter__(self):
            while self:
                yield self.pop(0)
    body = _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8,
                                            messages=Drain([{"role": "user", "content": "hi"}])))
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_a_dict_subclass_whose_membership_test_changes_it_is_never_asked(layout):
    class Sneaky(dict):
        def __contains__(self, k):
            self["text"] = "CHANGED"
            return dict.__contains__(self, k)
    body = _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8, messages=[
        {"role": "user", "content": [Sneaky(type="text", text="original")]}]))
    assert body["messages"][0]["content"][0]["text"] == "original"


def test_a_host_model_is_never_serialized_by_recant(layout):
    """A host's own pydantic model with a serializer that counts its calls: serialized once, by the SDK."""
    import pydantic

    class Counter(pydantic.BaseModel):
        n: int = 0

        @pydantic.model_serializer
        def ser(self):
            object.__setattr__(self, "n", self.n + 1)
            return {"n": self.n}
    body = _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": "a"}],
                                            extra_body={"x": Counter(), "opaque": (1, 2)}))
    assert body["x"] == {"n": 1}


def test_a_sentinel_that_is_also_a_model_is_left_to_the_sdk(layout):
    anthropic = pytest.importorskip("anthropic")
    import pydantic

    class Skip(anthropic.NotGiven, pydantic.BaseModel):
        x: int = 1
    _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": "a"}],
                                     extra_body={"x": Skip(x=2)}))


def test_a_timing_failure_in_a_helper_stream_is_not_the_host_s(layout, monkeypatch):
    import sys
    w = sys.modules["recant.wrap"]
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    with client.run("agent", session="s"):
        with client.messages.stream(**REQ) as s:
            class Clock:
                @staticmethod
                def monotonic():
                    raise RuntimeError("timing failed")

                perf_counter = staticmethod(__import__("time").perf_counter)
            monkeypatch.setattr(w, "time", Clock)
            assert s.get_final_message().content[-1].text == "ok"
    assert sum(friction.summary()["wrapper_errors"].values()) >= 1


def test_a_host_model_inside_an_sdk_block_is_never_serialized_early(layout):
    """A tool-use block whose input holds a host model with a serializer that edits a message: the SDK's order, not
    Recant's, decides what the message says (an earlier candidate ran the serializer first)."""
    anthropic = pytest.importorskip("anthropic")
    import pydantic

    def make():
        msgs = [{"role": "user", "content": "original"}]

        class Edits(pydantic.BaseModel):
            @pydantic.model_serializer
            def ser(self):
                msgs[0]["content"] = "changed"
                return {"edited": True}
        block = anthropic.types.ToolUseBlock(type="tool_use", id="t1", name="x", input={"h": Edits()})
        return dict(model="claude-sonnet-5-5", max_tokens=8, extra_body={"b": block}, messages=msgs)
    _same_bytes(layout, make)


def test_classifying_a_value_never_calls_its_metaclass(layout):
    calls = []

    class Meta(type):
        def __eq__(cls, other):
            calls.append(other)
            return type.__eq__(cls, other)
        __hash__ = type.__hash__

        def __getattribute__(cls, name):
            if name not in ("__class__",):
                calls.append(name)
            return type.__getattribute__(cls, name)

    class Odd(metaclass=Meta):
        pass
    import sys
    w = sys.modules["recant.wrap"]
    opaque: list = []
    w._readable({"x": Odd()}, opaque)
    w._view({"x": Odd()})
    assert opaque == ["Odd"] and calls == []


def test_a_clock_that_fails_before_a_call_or_in_a_raw_stream_is_not_the_host_s(layout, monkeypatch):
    import sys
    w = sys.modules["recant.wrap"]
    client = wrap(sdk(API()), store=layout, tenant=TENANT)

    class Dead:
        @staticmethod
        def monotonic():
            raise RuntimeError("clock")

        @staticmethod
        def perf_counter():
            raise RuntimeError("perf")
    monkeypatch.setattr(w, "time", Dead)
    assert client.messages.create(**REQ).content[-1].text == "ok"
    with client.messages.create(**REQ, stream=True) as s:
        kinds = [e.type for e in s]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"


def test_a_class_that_only_names_an_sdk_module_is_not_an_sdk_model(layout):
    import pydantic
    import sys
    w = sys.modules["recant.wrap"]

    class Forged(pydantic.BaseModel):
        n: int = 0

        @pydantic.model_serializer
        def ser(self):
            object.__setattr__(self, "n", self.n + 1)
            return {"n": self.n}
    Forged.__module__ = "anthropic.types"
    assert not w._sdk_model(Forged())
    body = _same_bytes(layout, lambda: dict(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": "a"}],
                                            extra_body={"x": Forged(), "opaque": (1, 2)}))
    assert body["x"] == {"n": 1}
    anthropic = pytest.importorskip("anthropic")
    assert w._sdk_model(anthropic.types.TextBlock(type="text", text="hi"))            # the SDK's own still are

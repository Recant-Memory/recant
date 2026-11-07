import dataclasses
import json
from types import SimpleNamespace

import pytest

from recant.log import LogWriter, read_run
from recant.provider import ContentBlock, Message, ModelRequest, Prices, ProviderError, Tier, ToolSpec, Usage, text
from recant.clients.anthropic import AnthropicProvider, build_params
from recant.provider.base import BaseProvider, Raw
from recant.provider.fake import FakeProvider
from recant_fixtures import TENANT


def req(body="hello", *, tier=Tier.T1, model=None, max_tokens=100, **kw):
    return ModelRequest(job=kw.pop("job", "test"), tier=None if model else tier, model=model,
                        messages=(Message("user", (text(body),)),), max_tokens=max_tokens, **kw)


# the test kernel's three rows and factors, in the shape a consumer passes (the library itself prices nothing)
PRICES = {"model-small": Prices(1.0, 5.0, cache_read=0.1, cache_write=1.25, batch_factor=0.5),
          "model-medium": Prices(2.0, 10.0, cache_read=0.2, cache_write=2.5, batch_factor=0.5),
          "model-large": Prices(5.0, 25.0, cache_read=0.5, cache_write=6.25, batch_factor=0.5)}


def records(log, kind):
    return [r for r in read_run(log.path, typed=False) if r["kind"] == kind]


# --- arithmetic ---------------------------------------------------------------------

def test_tier_lookup_is_deterministic_and_a_candidate_may_name_its_model(machinery, log):
    p = FakeProvider(machinery, log, prices=PRICES)
    assert [p.resolve(req(tier=t)) for t in Tier] == ["model-small", "model-medium", "model-large"]
    assert {p.resolve(req(tier=Tier.T2)) for _ in range(200)} == {"model-medium"}  # no randomised tail
    assert p.resolve(req(model="model-large")) == "model-large"
    with pytest.raises(ValueError):
        ModelRequest(job="t", tier=Tier.T1, model="model-small", messages=(Message("user", (text("x"),)),), max_tokens=1)
    with pytest.raises(ValueError):
        ModelRequest(job="t", messages=(Message("user", (text("x"),)),), max_tokens=1)


def test_every_call_is_logged_with_what_replay_needs(machinery, log):
    p = FakeProvider(machinery, log, prices=PRICES)
    p.call(req("same"))
    p.call(req("same"))
    p.call(req("different"))
    calls = records(log, "model_call")
    assert calls[0]["request_hash"] == calls[1]["request_hash"] != calls[2]["request_hash"]
    assert calls[0]["usage"] == {"input": 10, "output": 2, "cache_read": 0, "cache_write": 0, "reasoning": 0, "cache_write_long": 0}
    assert calls[0]["model_id"] == "model-small" and calls[0]["tier"] == 1 and calls[0]["job"] == "test"
    each = (10 * 1.0 + 2 * 5.0) / 1e6
    assert log.spent_usd == pytest.approx(3 * each)  # the run total, priced by the table passed


def test_a_refusal_is_an_error_never_an_empty_answer(machinery, log):
    p = FakeProvider(machinery, log, lambda m, r: Raw((), "refusal", Usage(input=5)), prices=PRICES)
    with pytest.raises(ProviderError, match="refusal"):
        p.call(req())
    assert records(log, "model_call")[0]["error"] == "refusal"


# --- the batch path -----------------------------------------------------------------

def test_batch_results_are_keyed_by_custom_id_out_of_order_and_charged_at_batch_rates(machinery, log):
    def responder(model, r):
        return Raw((text(r.messages[0].content[0].text.upper()),), "end_turn", Usage(input=1000, output=100))

    p = FakeProvider(machinery, log, responder, prices=PRICES)
    handle = p.submit_batch([req("alpha", custom_id="a"), req("beta", custom_id="b"), req("gamma")])
    assert p.batch_status(handle) == "ended"
    got = dict(p.batch_results(handle))
    assert got["a"].text == "ALPHA" and got["b"].text == "BETA" and len(got) == 3
    each = (1000 * 1.0 + 100 * 5.0) / 1e6 * 0.5
    assert log.spent_usd == pytest.approx(3 * each)
    calls = records(log, "model_call")
    assert {c["custom_id"] for c in calls} == set(got) and all(c["batch_id"] == handle.batch_id for c in calls)
    saved = json.loads((log.batches_dir / f"{handle.batch_id}.json").read_text())
    assert [r["custom_id"] for r in saved["requests"]][:2] == ["a", "b"]
    assert BaseProvider.load_handle(log.batches_dir / f"{handle.batch_id}.json") == handle  # resumable


# --- the adapter's request and response shapes ---------------------------------------

def test_request_shape_cache_marks_tools_schema_and_effort():
    r = ModelRequest(job="t", tier=Tier.T2, max_tokens=64,
                     system=(text("rules"), text("a long shared prefix", cache=True)),
                     messages=(Message("user", (text("q"),)),
                               Message("assistant", (ContentBlock("tool_use", tool_name="search", tool_input={"q": "x"}, tool_use_id="tu1"),)),
                               Message("user", (ContentBlock("tool_result", text="nothing", tool_use_id="tu1", is_error=True),))),
                     tools=(ToolSpec("search", "search the store", {"type": "object", "properties": {}, "additionalProperties": False}),),
                     effort="low")
    p = build_params("model-medium", r, supports_effort=True)
    assert p["model"] == "model-medium" and p["max_tokens"] == 64
    assert "cache_control" not in p["system"][0] and p["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert p["messages"][1]["content"][0] == {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "x"}}
    assert p["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu1", "content": "nothing", "is_error": True}
    assert p["tools"][0]["strict"] is True and p["output_config"] == {"effort": "low"}
    assert "thinking" not in p and "temperature" not in p
    assert p["messages"][-1]["role"] == "user"  # nothing is prefilled

    dropped = build_params("model-small", r, supports_effort=False)
    assert "output_config" not in dropped

    s = ModelRequest(job="t", tier=Tier.T1, max_tokens=64, messages=(Message("user", (text("q"),)),),
                     output_schema={"type": "object", "properties": {}, "additionalProperties": False})
    assert build_params("m", s, supports_effort=False)["output_config"] == \
        {"format": {"type": "json_schema", "schema": s.output_schema}}


def test_requests_refuse_a_fifth_cache_mark_and_tools_with_a_schema():
    five = tuple(text(f"b{i}", cache=True) for i in range(5))
    with pytest.raises(ValueError, match="cache"):
        ModelRequest(job="t", tier=Tier.T1, max_tokens=1, system=five, messages=(Message("user", (text("q"),)),))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelRequest(job="t", tier=Tier.T1, max_tokens=1, messages=(Message("user", (text("q"),)),),
                     tools=(ToolSpec("a", "b", {}),), output_schema={})


def _sdk_message(**over):
    base = dict(content=[SimpleNamespace(type="thinking", thinking=""),
                         SimpleNamespace(type="text", text="the answer"),
                         SimpleNamespace(type="tool_use", id="tu9", name="search", input='{"q": "caf\\u00e9"}')],
                stop_reason="tool_use", _request_id="req_1",
                usage=SimpleNamespace(input_tokens=11, output_tokens=7, cache_read_input_tokens=300,
                                      cache_creation_input_tokens=None))
    base.update(over)
    return SimpleNamespace(**base)


class StubClient:
    def __init__(self, message=None, batch_entries=None):
        self.created, self.batch_requests = [], None
        outer = self

        class Batches:
            def create(self, requests):
                outer.batch_requests = requests
                return SimpleNamespace(id="msgbatch_1")

            def retrieve(self, batch_id):
                outer.retrieved = getattr(outer, "retrieved", 0) + 1       # running at the first look, ended after: `poll` hands
                return SimpleNamespace(processing_status="in_progress" if outer.retrieved == 1 else "ended")   # results over only then

            def results(self, batch_id):
                return iter(batch_entries or [])

        class Messages:
            batches = Batches()

            def create(self, **params):
                outer.created.append(params)
                return message or _sdk_message()

            def stream(self, **params):
                outer.streamed = params
                final = message or _sdk_message()

                class Stream:
                    def __enter__(self):
                        return SimpleNamespace(get_final_message=lambda: final)

                    def __exit__(self, *exc):
                        return False
                return Stream()

            def count_tokens(self, **params):
                outer.counted = params
                return SimpleNamespace(input_tokens=42)

        self.messages = Messages()


def test_the_adapter_maps_a_response_into_our_shape(machinery, log):
    client = StubClient()
    out = AnthropicProvider(machinery, log, client=client, prices=PRICES).call(req("q", tier=Tier.T2))
    assert client.created[0]["model"] == "model-medium"
    assert [b.kind for b in out.content] == ["text", "tool_use"]      # thinking blocks carry nothing we read
    assert out.content[1].tool_input == {"q": "café"}                 # parsed, never string-matched
    assert out.usage == Usage(input=11, output=7, cache_read=300, cache_write=0)
    assert out.request_id == "req_1" and out.stop_reason == "tool_use"
    assert out.cost_usd == pytest.approx((11 * 2.0 + 300 * 2.0 * 0.1 + 7 * 10.0) / 1e6)


def test_a_large_output_cap_streams_and_maps_the_same(machinery, log):
    """A cap the SDK expects to run past 10 minutes is refused unless streamed."""
    from recant.clients.anthropic import AnthropicClient
    client = StubClient()
    small = AnthropicClient(client).complete(req("q", model="model-medium"))
    big = AnthropicClient(client).complete(req("q", model="model-medium", max_tokens=32_000))
    assert len(client.created) == 1 and client.streamed["max_tokens"] == 32_000
    assert big == small


def test_a_stream_that_stalls_mid_read_is_a_retryable_connection_error():
    """The SDK hands a mid-stream read timeout up as the HTTP library's own error."""
    import httpx2
    from recant.clients.anthropic import AnthropicClient
    client = StubClient()

    def stalled(**params):
        class Stream:
            def __enter__(self):
                def final():
                    raise httpx2.ReadTimeout("The read operation timed out")
                return SimpleNamespace(get_final_message=final)

            def __exit__(self, *exc):
                return False
        return Stream()
    client.messages.stream = stalled
    with pytest.raises(ProviderError) as got:
        AnthropicClient(client).complete(req("q", model="model-medium", max_tokens=32_000))
    assert got.value.retryable and "ReadTimeout" in str(got.value)


def test_the_adapter_counts_tokens_without_generation_parameters(machinery, log):
    client = StubClient()
    assert AnthropicProvider(machinery, log, client=client, prices=PRICES).count_tokens(req("q", effort="low", tier=Tier.T2)) == 42
    assert "max_tokens" not in client.counted and "output_config" not in client.counted


def test_the_adapter_batch_path(machinery, log):
    ok = SimpleNamespace(custom_id="b", result=SimpleNamespace(type="succeeded", message=_sdk_message(stop_reason="end_turn")))
    gone = SimpleNamespace(custom_id="a", result=SimpleNamespace(type="expired", error=SimpleNamespace(type="invalid_request", message="maxItems is not supported")))
    client = StubClient(batch_entries=[ok, gone])
    p = AnthropicProvider(machinery, log, client=client, prices=PRICES)
    handle = p.submit_batch([req("one", custom_id="a"), req("two", custom_id="b")])
    assert handle.batch_id == "msgbatch_1"
    assert [r["custom_id"] for r in client.batch_requests] == ["a", "b"]
    assert client.batch_requests[0]["params"]["model"] == "model-small"
    assert p.batch_status(handle) == "running"
    got = dict(p.batch_results(handle))
    assert got["b"].text == "the answer" and isinstance(got["a"], ProviderError) and got["a"].retryable
    assert "maxItems is not supported" in str(got["a"])                                  # the API's reason survives
    assert log.spent_usd == pytest.approx((11 * 1.0 + 300 * 1.0 * 0.1 + 7 * 5.0) / 1e6 * 0.5)


def test_sdk_errors_become_provider_errors(machinery, log):
    import anthropic
    import httpx2

    class Failing(StubClient):
        def __init__(self, exc):
            super().__init__()
            self.messages.create = lambda **kw: (_ for _ in ()).throw(exc)

    request = httpx2.Request("POST", "https://example.invalid/v1/messages")
    rate = anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=request), body=None)
    bad = anthropic.BadRequestError("no", response=httpx2.Response(400, request=request), body=None)
    for exc, kind, retryable in [(rate, "rate_limit", True), (bad, "bad_request", False),
                                 (anthropic.APIConnectionError(request=request), "connection", True)]:
        with pytest.raises(ProviderError) as caught:
            AnthropicProvider(machinery, log, client=Failing(exc)).call(req())
        assert caught.value.kind == kind and caught.value.retryable is retryable


def test_a_model_client_is_one_method_and_batch_is_used_only_when_it_is_there(machinery, log):
    """`complete` is the whole requirement; `submit`/`poll` and `count_tokens` are found by presence."""
    from recant import ModelClient
    from recant.harness.common import run_all
    from recant.provider import Completion, has_batch
    from recant.provider.base import BaseProvider

    class Mine:
        def __init__(self):
            self.seen = []

        def complete(self, request):
            self.seen.append(request)
            return Completion((text("hello"),), "end_turn", Usage(input=5, output=1))

    mine = Mine()
    assert isinstance(mine, ModelClient) and not has_batch(mine)
    p = BaseProvider(machinery, log, mine, prices=PRICES)
    assert p.call(req("q", tier=Tier.T2)).text == "hello"
    assert mine.seen[0].model == "model-medium" and mine.seen[0].tier is None          # the client is handed a resolved model
    assert not p.can_batch
    out = dict(run_all(p, [req("one", custom_id="a"), req("two", custom_id="b")], batch=True, wait=lambda: None))
    assert out["a"].text == out["b"].text == "hello" and len(mine.seen) == 3            # asked for batch, none there: sent live
    with pytest.raises(ProviderError, match="no_count_tokens"):
        p.count_tokens(req("q"))
    assert FakeProvider(machinery, log, prices=PRICES).can_batch


def test_without_a_price_table_the_library_reports_tokens_and_no_dollars(machinery, log):
    """Dollars appear only when the consumer passes prices; nothing here refuses a call."""
    p = FakeProvider(machinery, log)
    out = p.call(req("x" * 90000, tier=Tier.T3, max_tokens=4000))                  # a call a capped consumer would refuse
    assert out.cost_usd is None and out.usage.input == 10
    rec = records(log, "model_call")[0]
    assert rec["cost_usd"] is None and rec["usage"]["input"] == 10
    assert log.reported_spend_usd is None
    log.finish()
    assert records(log, "run_finished")[0]["spent_usd"] is None
    assert FakeProvider(machinery, log, prices={"model-small": PRICES["model-small"]}).cost_usd("model-large", Usage(input=1)) is None


def test_a_retryable_error_is_tried_again_with_backoff_before_the_call_fails(machinery, log):
    """the provider reads ProviderError.retryable. Each failed attempt is logged; a non-retryable error
    fails at once; after `retries` more tries the last error is raised."""
    waits = []

    class Flaky(StubClient):
        def __init__(self, errors):
            super().__init__()
            errs = list(errors)

            def create(**params):
                if errs:
                    raise errs.pop(0)
                return _sdk_message()
            self.messages.create = create

    stall = ProviderError("connection", "ReadTimeout", retryable=True)
    p = AnthropicProvider(machinery, log, client=Flaky([stall, stall]), prices=PRICES)
    p.sleep = waits.append
    assert p.call(req("q", tier=Tier.T2)).text == "the answer" and waits == [2.0, 4.0]
    calls = [r for r in read_run(log.path, typed=False) if r["kind"] == "model_call"]
    assert [bool(r.get("error")) for r in calls] == [True, True, False]

    waits.clear()
    p = AnthropicProvider(machinery, log, client=Flaky([ProviderError("bad_request", "no")]), prices=PRICES)
    p.sleep = waits.append
    with pytest.raises(ProviderError):
        p.call(req("q", tier=Tier.T2))
    assert waits == []

    p = AnthropicProvider(machinery, log, client=Flaky([stall] * 5), prices=PRICES)
    p.sleep = waits.append
    with pytest.raises(ProviderError):
        p.call(req("q", tier=Tier.T2))
    assert waits == [2.0, 4.0, 8.0]


def test_a_failed_batch_result_logs_the_0_4_1_fields_too(machinery, log):
    def responder(model, r):
        if r.custom_id == "b":
            return ProviderError("overloaded", "try later")
        return Raw((text("ok"),), "end_turn", Usage(input=10, output=1))

    p = FakeProvider(machinery, log, responder, prices=PRICES)
    handle = p.submit_batch([req("alpha", custom_id="a"), req("beta", custom_id="b", system=(text("Follow the rules."),))])
    list(p.batch_results(handle))
    calls = records(log, "model_call")
    failed = [c for c in calls if c["custom_id"] == "b"]
    assert failed and failed[0]["error"] and failed[0]["setting_fp_rest"] and failed[0]["system_prefix"] is not None

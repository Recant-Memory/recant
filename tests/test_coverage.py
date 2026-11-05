"""The coverage watch, the Coverage line, reconciling against a host's own totals, and the bench
verdict qualified by coverage. Offline: every model API is a mock transport."""
import json

import pytest

httpx2 = pytest.importorskip("httpx2")
anthropic = pytest.importorskip("anthropic")

import recant  # noqa: E402
from recant import costshape as cs  # noqa: E402
from recant import coverage, switch  # noqa: E402
from recant.bench import verdict_lines  # noqa: E402
from recant.store import StoreLayout  # noqa: E402

SECRET_PROMPT = "the principal's private question 7731"
SECRET_REPLY = "a private answer 9902"


@pytest.fixture(autouse=True)
def fresh_counts():
    coverage.drain()
    coverage._sinks.clear()
    yield
    coverage.drain()
    coverage._sinks.clear()


def anthropic_handler(request):
    return httpx2.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": SECRET_REPLY}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}})


def openai_handler(request):
    return httpx2.Response(200, json={
        "id": "gen-1", "object": "chat.completion", "created": 1, "model": "z-ai/glm-4.6",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": SECRET_REPLY}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48}})


def wrapped_anthropic(tmp_path):
    raw = anthropic.Anthropic(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(anthropic_handler)))
    return recant.wrap(raw, store=tmp_path / "store", tenant="t1")


def ask_wrapped(client, n):
    with client.run("agent", session="s1"):
        for _ in range(n):
            client.messages.create(model="claude-sonnet-5-5", max_tokens=64, messages=[{"role": "user", "content": SECRET_PROMPT}])


def ask_openrouter_raw(n):
    """What the openai SDK sends to OpenRouter, on the same HTTP library; the SDK itself is tested below if installed."""
    c = httpx2.Client(transport=httpx2.MockTransport(openai_handler))
    for _ in range(n):
        c.post("https://openrouter.ai/api/v1/chat/completions",
               json={"model": "z-ai/glm-4.6", "messages": [{"role": "user", "content": SECRET_PROMPT}]})


def report(tmp_path):
    return cs.report(StoreLayout(tmp_path / "store"), "t1", since=0, until=None)


def all_log_text(tmp_path):
    return "".join(p.read_text() for p in (tmp_path / "store").rglob("*.jsonl"))


def test_unwrapped_openrouter_calls_show_as_unseen_with_model_and_count_and_no_content(tmp_path):
    client = wrapped_anthropic(tmp_path)
    with client.run("agent", session="s1"):
        for _ in range(3):
            client.messages.create(model="claude-sonnet-5-5", max_tokens=64, messages=[{"role": "user", "content": SECRET_PROMPT}])
        ask_openrouter_raw(2)                    # inside the run: written as the run closes
    rep = report(tmp_path)
    s = rep.coverage()
    assert (s["calls_seen"], s["calls_unseen"]) == (3, 2)
    assert s["share_calls"] == pytest.approx(0.6)
    u = s["unseen"][0]
    assert (u["provider"], u["host"], u["model"], u["requests"]) == ("openrouter", "openrouter.ai", "z-ai/glm-4.6", 2)
    assert u["usage"] == {"input": 80, "output": 16, "cache_read": 0, "cache_write": 0}
    line = coverage.line(s)
    assert "Recant saw 60% of model calls" in line and "2 calls to openrouter.ai, model z-ai/glm-4.6" in line
    text = all_log_text(tmp_path)
    assert SECRET_PROMPT not in text and SECRET_REPLY not in text
    assert "coverage: Recant saw 60%" in rep.text()


def test_the_openai_sdk_through_openrouter_is_seen_as_unseen(tmp_path):
    openai = pytest.importorskip("openai")
    client = wrapped_anthropic(tmp_path)
    oa = openai.OpenAI(api_key="test", base_url="https://openrouter.ai/api/v1",
                       http_client=httpx2.Client(transport=httpx2.MockTransport(openai_handler)))
    with client.run("agent", session="s1"):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=64, messages=[{"role": "user", "content": SECRET_PROMPT}])
        oa.chat.completions.create(model="z-ai/glm-4.6", messages=[{"role": "user", "content": SECRET_PROMPT}])
    s = report(tmp_path).coverage()
    assert (s["calls_seen"], s["calls_unseen"]) == (1, 1)
    assert s["unseen"][0]["host"] == "openrouter.ai" and s["unseen"][0]["model"] == "z-ai/glm-4.6"
    assert SECRET_PROMPT not in all_log_text(tmp_path)


def test_a_fully_wrapped_host_is_100_percent(tmp_path):
    client = wrapped_anthropic(tmp_path)
    ask_wrapped(client, 4)
    rep = report(tmp_path)
    s = rep.coverage()
    assert (s["calls_seen"], s["calls_unseen"], s["share_calls"]) == (4, 0, 1.0)
    assert rep.watched                            # the watch wrote its window, with nothing unseen
    assert coverage.line(s, watched=rep.watched).startswith("coverage: Recant saw all 4 model calls")


def test_an_unknown_host_with_an_llm_shaped_body_is_other_with_its_domain_hashed(tmp_path):
    client = wrapped_anthropic(tmp_path)
    c = httpx2.Client(transport=httpx2.MockTransport(openai_handler))
    with client.run("agent", session="s1"):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=64, messages=[{"role": "user", "content": "x"}])
        c.post("https://llm.example-private-gateway.internal/v1/chat", json={"model": "house-model-3", "messages": [{"role": "user", "content": "x"}]})
        c.post("https://llm.example-private-gateway.internal/v1/other", json={"hello": "not a model call"})
        c.get("https://api.openai.com/v1/models")                # a GET is never a model call
    s = report(tmp_path).coverage()
    assert s["calls_unseen"] == 1
    u = s["unseen"][0]
    assert u["provider"] == "other" and u["host"].startswith("other:") and len(u["host"]) == len("other:") + 12
    assert "example-private-gateway" not in all_log_text(tmp_path)
    assert "an unlisted host (domain hashed)" in coverage.line(s)


def test_the_switch_off_turns_the_watch_off(tmp_path):
    client = wrapped_anthropic(tmp_path)
    switch.set_state(StoreLayout(tmp_path / "store"), "off")
    ask_openrouter_raw(3)
    assert coverage.snapshot() == {}


def test_classify_skips_token_counting_and_reads_the_model_from_the_path():
    assert coverage.classify("POST", "api.anthropic.com", "/v1/messages/count_tokens", b'{"model":"m"}') is None
    assert coverage.classify("GET", "api.anthropic.com", "/v1/models", None) is None
    assert coverage.classify("POST", "bedrock-runtime.us-east-1.amazonaws.com", "/model/anthropic.claude-x/invoke", b"{}") == \
        ("aws_bedrock", "bedrock-runtime.us-east-1.amazonaws.com", "anthropic.claude-x")
    assert coverage.classify("POST", "us-central1-aiplatform.googleapis.com",
                             "/v1/projects/p/locations/l/publishers/google/models/gemini-3:generateContent", b"{}")[0] == "google_vertex"


# ---------------------------------------------------------------- reconciliation against the host's own totals
def usage_rows():
    """Synthetic, in the minimal shape a host's per-run log takes: two Anthropic models, one on a subscription, and a
    model through a router. Counts only."""
    return [
        {"ts": "2026-11-01T10:00:00Z", "model": "claude-sonnet-5-5", "tokens_in": 100, "tokens_out": 50, "cache_read": 800, "cache_write": 50, "billed": "api_key"},
        {"ts": "2026-11-01T11:00:00Z", "model": "claude-sonnet-5-5", "tokens_in": 100, "tokens_out": 50, "cache_read": 800, "cache_write": 50, "billed": "api_key"},
        {"ts": "2026-11-02T10:00:00Z", "model": "claude-opus-5-5", "tokens_in": 10, "tokens_out": 90, "cache_read": 0, "cache_write": 0, "billed": "subscription"},
        {"ts": "2026-11-02T12:00:00Z", "model": "~z-ai/glm-latest", "tokens_in": 500, "tokens_out": 100, "cache_read": 0, "cache_write": 0, "billed": "api_key"},
    ]


def metered(day, model, inp, out, read, write):
    return {"kind": "model_call", "ts": f"{day}T10:00:01Z", "model_id": model,
            "usage": {"input": inp, "output": out, "cache_read": read, "cache_write": write}}


def test_reconcile_against_an_empty_meter_sees_nothing_and_calls_out_subscription_and_other_providers(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text("".join(json.dumps(dict(r, prompt="never read")) + "\n" for r in usage_rows()))
    rows = coverage.read_usage(p)
    assert all("prompt" not in r for r in rows)
    r = coverage.reconcile(rows, [], price=lambda m, u: cs.Report.price(_FakeRep(), m, u))   # the library's list prices
    assert r["share_tokens"] == 0.0 and r["runs"] == 4 and r["runs_subscription"] == 1
    by = {m["model"]: m for m in r["by_model"]}
    assert by["~z-ai/glm-latest"]["provider"] == "zai" and by["~z-ai/glm-latest"]["route"] == "openrouter"
    assert by["claude-opus-5-5"]["runs_subscription"] == 1
    text = coverage.render_reconcile(r)
    assert "Recant saw 0% of the tokens" in text and "subscription: 1 runs" in text and "provider zai" in text


def test_reconcile_against_a_meter_covering_half_is_fifty_percent():
    rows = usage_rows()
    half = [metered("2026-11-01", "claude-sonnet-5-5", 100, 50, 800, 50),                 # one of the two sonnet runs
            metered("2026-11-02", "claude-opus-5-5", 5, 45, 0, 0),                        # half the opus run
            metered("2026-11-02", "~z-ai/glm-latest", 250, 50, 0, 0)]                      # half the routed run
    r = coverage.reconcile(rows, half)
    assert r["share_tokens"] == pytest.approx(0.5, abs=0.005)
    assert all(m["share_tokens"] == pytest.approx(0.5, abs=0.01) for m in r["by_model"])


class _FakeRep:
    profiles: dict = {}


# ---------------------------------------------------------------- the bench verdict
def bench_report(share_seen):
    seen = round(100 * share_seen)
    cov_ = {"calls_seen": seen, "calls_unseen": 100 - seen, "share_calls": share_seen, "share_usd": None, "usd_seen": None,
            "usd_unseen": None, "unseen_unpriced": 100 - seen, "watched": True,
            "unseen": [{"provider": "openrouter", "host": "openrouter.ai", "model": "z-ai/glm-4.6", "requests": 100 - seen}]}
    return {"header": {"arms": {"A": "off", "B": "safe"}}, "coverage": cov_,
            "cost": {"priced": True, "paired_difference_per_task": {"relative": -0.4, "relative_interval": [-0.5, -0.3],
                                                                  "interval": [-0.02, -0.01], "delta": -0.015},
                     "per_passed_task_relative": None, "pairs_with_a_failure": 0, "unpriced_calls": {"A": 0, "B": 0}},
            "quality": None, "predictions": [], "failures": {"A": {"failed_tasks": 0}, "B": {"failed_tasks": 0}},
            "discarded": {"A": {"calls": 0}, "B": {"calls": 0}}, "equivalence": {"A": {"not_equivalent": 0}, "B": {"not_equivalent": 0}},
            "latency": {"paired_task_difference_s": {"interval": [-1, 1]}}}


def test_a_bench_with_40_percent_unseen_gets_the_qualified_verdict():
    lines = verdict_lines(bench_report(0.60))
    assert lines[0].startswith("this result covers only 60% of your traffic; the rest went to openrouter.ai")
    assert any(x.startswith("cost per task -40.0%") for x in lines)        # an overall saving, qualified


def test_a_bench_with_60_percent_unseen_gives_no_overall_saving():
    lines = verdict_lines(bench_report(0.40))
    assert "No overall saving is given" in lines[0]
    assert not any(x.startswith("cost per task") for x in lines)
    assert any(x.startswith("covered part only: cost per task -40.0%") for x in lines)


def test_a_bench_with_full_coverage_is_not_qualified():
    lines = verdict_lines(bench_report(1.0))
    assert lines[0].startswith("cost per task -40.0%")


# ---------------------------------------------------------------- the feedback bundle
def test_the_feedback_bundle_carries_coverage_by_provider_and_public_model_only():
    from recant import feedback
    cov_ = {"calls_seen": 10, "calls_unseen": 3, "share_calls": 10 / 13, "share_usd": None, "usd_seen": 0.1, "usd_unseen": None,
            "unseen_unpriced": 3, "watched": True, "unseen": [
                {"provider": "openrouter", "host": "openrouter.ai", "model": "~z-ai/glm-latest", "requests": 1, "with_usage": 0, "usage": {}},
                {"provider": "other", "host": "other:0123456789ab", "model": "house-model-3", "requests": 1, "with_usage": 0, "usage": {}},
                {"provider": "other", "host": "llm.customer-name.internal", "model": "gpt-5", "requests": 1, "with_usage": 0, "usage": {}}]}
    dropped: list = []
    out = feedback.scrub({"coverage": cov_}, dropped=dropped)["coverage"]
    u = out["unseen"]
    assert (u[0]["provider"], u[0]["host"], u[0]["model"]) == ("openrouter", "openrouter.ai", "~z-ai/glm-latest")
    assert (u[1]["provider"], u[1]["host"], u[1]["model"]) == ("other", "other:0123456789ab", None)    # a host's own model name
    assert u[2]["host"] is None                                    # a raw domain never passes, even if one slipped in
    assert out["calls_unseen"] == 3 and out["share_calls"] == pytest.approx(10 / 13)
    assert "customer-name" not in json.dumps(out) and "house-model" not in json.dumps(out)

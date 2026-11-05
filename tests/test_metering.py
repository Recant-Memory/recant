"""The fields costshape needs on every call and run — the session on the call (not in the request
hash), cache writes split by lifetime and priced apart, the provider's billed figure kept apart from Recant's
computed cost, and what Recant served counted by the provider's own counter where it has one."""
import json
from types import SimpleNamespace

from recant.clients.anthropic import _raw
from recant.log import LogWriter, read_run
from recant.metering import served_tokens
from recant.provider import Completion, Message, ModelRequest, Prices, ProviderError, Usage, text
from recant.provider.base import BaseProvider
from recant_fixtures import TENANT


def req(**kw):
    return ModelRequest(job="turn", model="m", messages=(Message("user", (text("hello"),)),), max_tokens=10, **kw)


class Client:
    def __init__(self, usage=Usage(input=10, output=5), billed=None, counts=None):
        self.usage, self.billed, self.counts = usage, billed, counts

    def complete(self, request):
        return Completion(content=(text("ok"),), stop_reason="end_turn", usage=self.usage, billed_usd=self.billed)


def calls(log):
    return [r for r in read_run(log.path, typed=False) if r["kind"] == "model_call"]


def test_the_session_is_logged_on_the_call_and_does_not_move_the_request_hash(layout, machinery):
    from recant.log.records import ModelCall  # noqa: F401
    assert req(session="s-1").to_dict() == req().to_dict()
    with LogWriter(layout, TENANT, "t", artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash) as log:
        BaseProvider(machinery, log, Client()).call(req(session="s-1"))
        BaseProvider(machinery, log, Client()).call(req())
    a, b = calls(log)
    assert a["session"] == "s-1" and b["session"] is None and a["request_hash"] == b["request_hash"]


def test_cache_writes_are_split_by_lifetime_and_the_long_ones_priced_apart():
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="x")], stop_reason="end_turn",
                          usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                                                cache_creation_input_tokens=3000,
                                                cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=1000, ephemeral_1h_input_tokens=2000)))
    u = _raw(msg).usage
    assert (u.cache_write, u.cache_write_long) == (3000, 2000)
    p = Prices(input=4.0, output=20.0, cache_write=5.0, cache_write_long=8.0, cache_read=0.2)
    assert abs(p.cost_usd(Usage(cache_write=3000, cache_write_long=2000)) - (1000 * 5.0 + 2000 * 8.0) / 1e6) < 1e-12
    assert Prices(input=4.0, output=20.0, cache_write=5.0).cost_usd(Usage(cache_write=3000, cache_write_long=2000)) == 3000 * 5.0 / 1e6
    old = SimpleNamespace(**{**msg.usage.__dict__}); del old.cache_creation       # a response without the split
    assert _raw(SimpleNamespace(content=msg.content, stop_reason="end_turn", usage=old)).usage.cache_write_long == 0


def test_the_billed_figure_is_kept_apart_from_the_computed_cost(layout, machinery):
    with LogWriter(layout, TENANT, "t", artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash) as log:
        p = BaseProvider(machinery, log, Client(billed=0.0123), prices={"m": Prices(input=1.0, output=1.0)})
        resp = p.call(req())
        BaseProvider(machinery, log, Client()).call(req())
    (a, b) = calls(log)
    assert a["billed_usd"] == 0.0123 and a["cost_usd"] == 15 / 1e6 and resp.billed_usd == 0.0123
    assert b["billed_usd"] is None and b["cost_usd"] is None


def test_served_tokens_are_counted_by_the_providers_counter_or_estimated_and_say_which(layout, machinery):
    class Counting(Client):
        def count_tokens(self, request):
            return 7 + len(request.messages[0].content[0].text.split())        # 7 tokens of framing, a token a word
    est = lambda t: len(t) // 4                                                   # noqa: E731
    with LogWriter(layout, TENANT, "t", artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash) as log:
        base = {}
        got, ok = served_tokens(BaseProvider(machinery, log, Counting()), "m", {"core": "a b c d", "slice": ""}, estimate=est, baseline=base)
        assert ok and got == {"core": 3, "slice": 0} and base == {"m": 8}     # four words less the one-character probe
        got, ok = served_tokens(BaseProvider(machinery, log, Client()), "m", {"core": "a b c d"}, estimate=est)
        assert not ok and got == {"core": 1}                                     # no counter: estimated, and said so


def test_segment_hashes_name_the_segment_that_moved_and_ignore_cache_markers():
    """a hash of everything up to each breakpoint; two calls agree up to the segment that changed."""
    from recant.metering import segment_hashes
    from recant.provider import ToolSpec
    tools = (ToolSpec("t", "a tool", {"type": "object"}),)

    def r(core, msg="hi", ttl="1h"):
        return ModelRequest(job="turn", model="m", tools=tools, max_tokens=10,
                            system=(text("character", cache=True, ttl=ttl), text(core, cache=True, ttl=ttl)),
                            messages=(Message("user", (text(msg, cache=True),)),))
    a, b, c, d = segment_hashes(r("core A")), segment_hashes(r("core A", msg="later")), segment_hashes(r("core B")), segment_hashes(r("core A", ttl="5m"))
    assert [s["at"] for s in a] == ["system.0", "system.1", "messages.0.0"] and a[0]["ttl"] == "1h"
    assert [x["hash"] for x in a[:2]] == [x["hash"] for x in b[:2]] and a[2]["hash"] != b[2]["hash"]     # only the message moved
    assert a[0]["hash"] == c[0]["hash"] and a[1]["hash"] != c[1]["hash"]                                 # the core moved
    assert [x["hash"] for x in a] == [x["hash"] for x in d]                                              # markers are not bytes


def test_every_call_logs_its_segment_hashes(layout, machinery):
    with LogWriter(layout, TENANT, "t", artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash) as log:
        BaseProvider(machinery, log, Client()).call(ModelRequest(job="turn", model="m", max_tokens=10,
                                                                 system=(text("s", cache=True),), messages=(Message("user", (text("x"),)),)))
    (c,) = calls(log)
    assert [s["at"] for s in c["segments"]] == ["system.0"] and len(c["segments"][0]["hash"]) == 16


def test_a_target_signature_is_hashes_only_and_stable():
    from recant.metering import target_signature
    a = target_signature(["prog:cat", "path:api/x.py", "path:api/x.py", ""])
    assert a == target_signature(["path:api/x.py", "prog:cat"]) and len(a) == 2 and all(len(x) == 12 for x in a)
    assert "cat" not in "".join(a)


def test_a_call_logs_the_consumers_key_label_never_the_key(layout, machinery):
    with LogWriter(layout, TENANT, "t", artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash) as log:
        BaseProvider(machinery, log, Client(), key_label="bench").call(req())
        BaseProvider(machinery, log, Client()).call(req())
    a, b = calls(log)
    assert a["key"] == "bench" and b["key"] is None


def test_a_host_with_its_own_client_meters_through_meter(layout):
    """the same model_call record, with the session, the setting fingerprint, and whether each request only
    appends to the one before it in its session."""
    from recant.metering import Meter
    m = Meter(layout, TENANT, key_label="bench")
    log = m.run("task", task=3)
    sys_, tools = "policy", [{"name": "t"}]
    a = [{"role": "system", "content": sys_}, {"role": "user", "content": "hi"}]
    b = a + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "more"}]
    m.record(job="agent", model="m", usage={"input": 100, "output": 5}, session="t3", system=sys_, tools=tools, messages=a)
    m.record(job="agent", model="m", usage={"input": 160, "output": 5}, session="t3", system=sys_, tools=tools, messages=b)
    m.record(job="agent", model="m", usage={"input": 40, "output": 5}, session="t3", system=sys_, tools=tools, messages=a[:1])
    m.record(job="user", model="u", usage={"input": 30, "output": 9}, session="t3", messages=[{"role": "user", "content": "q"}])
    m.close()
    recs = [r for r in read_run(log.path, typed=False) if r["kind"] == "model_call"]
    assert [r["extends_previous"] for r in recs] == [None, True, False, None]      # the user simulator is its own stream
    assert recs[0]["setting_fp"] == recs[1]["setting_fp"] != recs[3]["setting_fp"] and recs[0]["key"] == "bench"
    assert recs[1]["usage"]["input"] == 160 and recs[1]["session"] == "t3"


def test_a_host_records_its_tool_calls_and_the_advisor_reads_them(layout):
    """a host's tool calls (bytes fetched, bytes sent to the model, hashed targets) land as tool_call records the
    advisor's trimming and loop findings read; the targets are hashes, never the text."""
    from recant import costshape as cs
    from recant.metering import Meter
    m = Meter(layout, TENANT, key_label="bench")
    m.run("task", task=1)
    m.record(job="agent", model="claude-sonnet-5", usage={"input": 9000, "output": 50}, session="t1", messages=[{"q": 1}])
    for _ in range(3):
        m.tool(name="get_order", session="t1", input={"id": "#W1"}, bytes_fetched=12000, bytes_to_model=12000, targets=["#W1"])
    m.close()
    recs = [r for r in read_run(m_path(layout), typed=False) if r["kind"] == "tool_call"]
    assert len(recs) == 3 and recs[0]["bytes_fetched"] == 12000 and "#W1" not in json.dumps(recs[0]["targets"])
    rep = cs.report(layout, TENANT, cs.HostShape(agent_jobs=("agent",)), since="30d")
    assert len(rep.tools) == 3 and cs.detect_tool_loop(rep).evidence["repeats"] == 2


def m_path(layout):
    (p,) = sorted(layout.log_dir(TENANT).glob("*.jsonl"))
    return p

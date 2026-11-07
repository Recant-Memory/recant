"""What a stranger's agent gets with no host code of ours: the drop-in wrapper on the raw SDK's shape, the safe
levers carried out on an Anthropic-shaped request (differing only in cache markers), the notes tools and their
handler, and session_start's defaults (list prices, the client's counter, the library's own log)."""
import json
from types import SimpleNamespace

from recant import costshape as cs, levers, quality, wrap
from recant import episodes, notes
from recant.log import read_run
from recant.store import MemoryStore
from recant.store.hot import read_lines
from recant_fixtures import ALLOWLIST, TENANT


# ---------------------------------------------------------------- levers on a request
def test_apply_places_cache_markers_only_and_leaves_the_request_otherwise_identical():
    params = {"model": "claude-sonnet-5", "max_tokens": 100, "system": "You are terse.",
              "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]}
    assert levers.apply(params, {}) == params and levers.apply(params, {}) is not params             # no applied lever: a copy
    ov = {"breakpoint.add": {"kind": "applied", "action": {"cache": "moving breakpoint"}}}
    out = levers.apply(params, ov)
    assert out["system"] == [{"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}]
    assert out["messages"][-1]["content"] == [{"type": "text", "text": "more", "cache_control": {"type": "ephemeral"}}]
    assert out["messages"][0] == params["messages"][0] and params["system"] == "You are terse."     # the input untouched
    assert quality.equivalent(params, out)                                                             # only the markers differ
    ov["ttl.prefix"] = {"kind": "applied", "action": {"ttl": "1h", "was": "5m"}}
    out = levers.apply(params, ov)
    assert out["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"} and "ttl" not in out["messages"][-1]["content"][0]["cache_control"]
    assert quality.equivalent(params, out)
    assert levers.apply(params, {"ttl.prefix": ov["ttl.prefix"]}) == params                           # a lifetime alone has no marker to sit on
    assert levers.CARRIES_OUT == ("breakpoint.add", "ttl.prefix")


class FakeSDK:
    """The raw SDK's shape: client.messages.create(**kwargs) -> a response with content, usage and stop_reason."""
    def __init__(self):
        self.sent = []
        self.messages = self

    def create(self, **kwargs):
        self.sent.append(kwargs)
        n = sum(len(json.dumps(m)) for m in kwargs["messages"]) // 4 + len(str(kwargs.get("system", ""))) // 4
        marked = any("cache_control" in b for m in kwargs["messages"] for b in (m["content"] if isinstance(m["content"], list) else []))
        usage = SimpleNamespace(input_tokens=0 if marked else n, output_tokens=12, cache_read_input_tokens=max(0, n - 20) if marked else 0,
                                cache_creation_input_tokens=20 if marked else 0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], usage=usage, stop_reason="end_turn", model=kwargs["model"])

    def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=7)


def chat(client, runs: int, calls: int = 3):
    for i in range(runs):
        with client.run("agent", session=f"s{i}"):
            msgs = []
            for k in range(calls):
                msgs.append({"role": "user", "content": f"turn {k} of run {i}, some words to carry " * 20})
                client.messages.create(model="claude-sonnet-5", max_tokens=50, system="You are terse. " * 400, messages=list(msgs))   # past the 1,024-token cache minimum
                msgs.append({"role": "assistant", "content": "ok"})


def test_wrap_meters_every_call_finds_the_breakpoint_lever_and_carries_it_out_on_the_next_call(layout):
    sdk = FakeSDK()
    client = wrap(sdk, store=layout, tenant=TENANT, job="agent")
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    assert host["agent_jobs"] == ["agent"] and host["carries_out"] == ["breakpoint.add", "ttl.prefix"]      # written for `recant cost`
    chat(client, runs=25)
    assert all("cache_control" not in json.dumps(k) for k in sdk.sent)                                 # shadow: nothing applied yet
    paths = sorted(layout.log_dir(TENANT).glob("*.jsonl"))
    recs = [r for p in paths for r in read_run(p, typed=False) if r["kind"] == "model_call"]
    assert len(recs) == 75 and recs[1]["extends_previous"] is True and recs[0]["session"] == "s0" and recs[0]["usage"]["input"] > 0
    shape = cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()})
    rep = cs.report(layout, TENANT, shape, since="30d")
    lv = next(l for l in cs.advise(rep) if l.id == "breakpoint.add")
    assert lv.saves_usd > 0 and cs.eligible(rep, lv)[0], cs.eligible(rep, lv)
    cs.apply(layout, TENANT, rep, lv)
    chat(client, runs=1)                                                                                 # the same host code, unchanged
    last = sdk.sent[-1]
    assert last["system"][-1]["cache_control"] == {"type": "ephemeral"} and last["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert client.last_equivalent is True and client.last_sent_markers == ["system.0 (5m)", "messages.4.0 (5m)"]
    rec = [r for p in sorted(layout.log_dir(TENANT).glob("*.jsonl")) for r in read_run(p, typed=False) if r["kind"] == "model_call"][-1]
    assert rec["usage"]["cache_read"] > 0 and rec["usage"]["cache_write"] == 20                        # the fake bills the marked request as cached


def test_wrap_sends_the_hosts_own_request_when_a_lever_would_change_more_than_markers(layout, monkeypatch):
    sdk = FakeSDK()
    client = wrap(sdk, store=layout, tenant=TENANT)
    (layout.costshape_dir(TENANT)).mkdir(parents=True, exist_ok=True)
    (layout.costshape_dir(TENANT) / "overlay.json").write_text(json.dumps({"breakpoint.add": {"kind": "applied", "action": {}}}))
    monkeypatch.setattr(levers, "apply", lambda params, overlay, **kw: {**params, "system": "something else"})   # a broken lever
    with client.run("agent", session="x"):
        client.messages.create(model="claude-sonnet-5", max_tokens=5, system="S", messages=[{"role": "user", "content": "q"}])
    assert sdk.sent[-1]["system"] == "S" and client.last_equivalent is False                          # what the host built went out


# ---------------------------------------------------------------- the notes tools for any host
def test_the_notes_tools_and_handler_work_with_no_host_code(layout):
    w = episodes.open_session(layout, TENANT, "2026-01-05")
    w.append("user", "Our accountant is Alice Example and the shop rent is 3,300 a month.", who="me", when="2026-01-05T09:00:00+00:00")
    lines = read_lines(layout.hot_session_path(TENANT, "2026-01-05", w.id))
    assert [t["name"] for t in notes.TOOL_SPECS] == ["note_fact", "read_notes"] and all("input_schema" in t for t in notes.TOOL_SPECS)
    out = notes.handle("note_fact", {"entries": [
        {"scope": "company", "attribute": "accountant", "value": "Alice Example", "source": "principal"},
        {"scope": "company", "attribute": "rent", "value": "3,300", "source": "principal"},
        {"scope": "company", "attribute": "auditor", "value": "Sample & Partners", "source": "principal"}]},
        layout=layout, tenant=TENANT, lines=lines, episode_path=w.episode_path, today="2026-01-05", principal_id="me")
    assert out.startswith("Noted 2 of 3 in the user block; 1 refused.") and "auditor: not noted (value not in the principal's turns)" in out
    ns = notes.NotesStore(layout, TENANT, "user:me")
    assert sorted(e.attribute for e in ns.entries()) == ["accountant", "rent"] and all(e.source.quote and e.source.episode for e in ns.entries())
    read = notes.handle("read_notes", {"terms": ["rent"]}, layout=layout, tenant=TENANT, lines=lines, episode_path=w.episode_path,
                        today="2026-01-05", principal_id="me")
    assert "rent = 3,300" in read and "shop rent is 3,300" in read
    assert "needs the host's permission" in notes.handle("note_fact", {"entries": [{"scope": "c", "attribute": "a", "value": "v", "source": "principal"}], "block": "team"},
                                                         layout=layout, tenant=TENANT, lines=lines, episode_path=w.episode_path, today="2026-01-05", principal_id="me")


def test_session_start_defaults_to_list_prices_the_clients_counter_and_its_own_log(layout, machinery):
    store = MemoryStore(layout, list(ALLOWLIST) + ["hot/sessions/????-??-??/*.md"])
    ns = notes.NotesStore(layout, TENANT, "user:me")
    ns.add({"scope": "company", "attribute": "accountant", "value": "Alice Example", "as_of": "2026-01-05", "source": {"kind": "principal", "quote": "q"}}, today="2026-01-05")
    assert notes.profile_for("claude-sonnet-5").p == cs.LIST_PRICES["claude-sonnet-5"].p and notes.profile_for("unknown-model").p == 0.0
    before = set(layout.log_dir(TENANT).glob("*.jsonl"))
    counted = lambda req: len(req.messages[0].content[0].text) // 4 + 3                                # noqa: E731 — 3 tokens of framing
    block = notes.session_start(layout, TENANT, store, principal_id="me", serve=machinery.kernel.serve, today="2026-01-06", block="user:me",
                                model="claude-sonnet-5", client=SimpleNamespace(count_tokens=counted))   # no profile, no log, no count
    assert "accountant=Alice Example" in block["text"] and block["measured"] is True
    assert block["tokens"] == len(block["text"]) // 4                                                  # the client's counter, less its framing
    new = set(layout.log_dir(TENANT).glob("*.jsonl")) - before
    assert len(new) == 1                                                                               # the library opened its own log
    recs = [r for r in read_run(new.pop(), typed=False) if r["kind"] == "notes_switch"]
    assert len(recs) == 1 and recs[0]["measured"] is True and recs[0]["notes_usd"] >= 0 and recs[0]["switch"] is False


def test_the_counter_reads_the_raw_sdk_and_the_wrapper_and_estimates_without_either(layout):
    sdk = SimpleNamespace(messages=SimpleNamespace(count_tokens=lambda model, messages: SimpleNamespace(input_tokens=len(messages[0]["content"]) // 4 + 5)))
    count = notes.counter_for(sdk, "claude-sonnet-5")
    assert count("a" * 400) == (100, True)
    assert notes.counter_for(wrap(FakeSDK(), store=layout, tenant=TENANT), "claude-sonnet-5")("b" * 40) == (0, True)   # the wrapper passes it through
    assert notes.counter_for(SimpleNamespace(), "claude-sonnet-5") is None and notes.counter_for(sdk, None) is None

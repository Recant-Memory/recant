"""Notes mode: the evidence checks in code, history by key, the rendered block, the switch that
fires once and never flips back, and the migration into memory records with no model call."""
import json
from types import SimpleNamespace

from recant import episodes
from recant.log import read_run
from recant.notes import HostFigures, NotesStore, apply_migration, check, decide, migrate, render, session_start
from recant.store import MemoryStore
from recant.store.hot import read_lines
from recant_fixtures import ALLOWLIST, TENANT

ALLOW = list(ALLOWLIST) + ["hot/sessions/????-??-??/*.md"]
DAY = "2026-10-05"


def session_with(layout, turns):
    """A session file with the given (role, text) lines; returns (its lines, its episode path)."""
    w = episodes.open_session(layout, TENANT, DAY)
    for role, text_ in turns:
        if role == "user":
            w.append("user", text_, who="owner", when=f"{DAY}T09:00:00+00:00", where="#ops")
        else:
            w.append(role, text_)
    return read_lines(layout.hot_session_path(TENANT, DAY, w.id)), w.episode_path


def entry(value="Alice Example", *, kind="principal", quote="our accountant is Alice Example", call=None, attribute="accountant"):
    return {"scope": "company", "entity": None, "attribute": attribute, "value": value, "as_of": DAY,
            "source": {"kind": kind, "quote": quote, "call": call}}


def test_a_forged_quote_is_refused_and_a_real_one_is_anchored_to_its_line(layout):
    lines, ep = session_with(layout, [("user", "Our accountant is Alice Example."), ("assistant", "Noted: your accountant is Carol Example.")])
    assert check(entry(), lines, ep) == (None, f"{ep}#L1-L1")
    assert check(entry("Carol Example", quote="your accountant is Carol Example"), lines, ep)[0] == "quote not in the principal's turns"   # the agent's echo is not evidence
    assert check(entry(quote=""), lines, ep) == (None, f"{ep}#L1-L1")        # no quote given — the value is in the principal's words
    assert check(entry(quote="x"), lines, ep)[0] == "no quote"
    assert check({**entry(), "source": None}, lines, ep)[0] == "no source"


def test_a_tool_entry_without_a_real_call_is_refused(layout):
    lines, ep = session_with(layout, [("user", "What is in the report?"), ("tool", "run_shell: `head reports/day-01.csv` → exit 0, 420 rows")])
    assert check(entry("420 rows", kind="tool", quote=None, call="run_shell: head reports/day-01.csv", attribute="report_rows"), lines, ep) == (None, f"{ep}#L2-L2")
    assert check(entry("420 rows", kind="tool", quote=None, call="run_shell: cat other.csv", attribute="report_rows"), lines, ep)[0] == "call not made in this session"
    assert check(entry("420 rows", kind="tool", quote=None, call="", attribute="report_rows"), lines, ep)[0] == "no call"


def test_an_absence_is_refused(layout):
    lines, ep = session_with(layout, [("user", "The policy number has not been provided yet.")])
    assert check(entry("has not been provided yet", quote="has not been provided yet", attribute="policy_number"), lines, ep)[0] == "absence"


def test_a_changed_key_keeps_its_history_and_renders_byte_stably(layout):
    ns = NotesStore(layout, TENANT)
    a = ns.add(entry("Example Coffee Co", attribute="bean_supplier"), today="2026-10-01")
    b = ns.add(entry("Sample Roasters", attribute="bean_supplier"), today="2026-10-08")
    ns.add(entry(attribute="accountant"), today="2026-10-02")
    cur = ns.current()
    assert [(c.key, [h.id for h in hist]) for c, hist in cur] == [(a.key, [a.id]), ("company::-::accountant", [])]
    assert cur[0][0].id == b.id
    text_ = render(cur)
    assert text_.splitlines() == ["===== notes =====",                                   # one line per scope and entity
                                  "company/-: accountant=Alice Example (2026-10-05; 10-05); bean_supplier=Sample Roasters (2026-10-05; 10-05); was Example Coffee Co (2026-10-05)"]
    assert render(NotesStore(layout, TENANT).current()) == text_          # re-read from the file: the same bytes
    ns.refuse(entry(quote="x"), "quote not in the principal's turns", today="2026-10-08")
    assert [r["reason"] for r in ns.refused()] == ["quote not in the principal's turns"]


def serve(**kw):
    return SimpleNamespace(**{"core_tokens": 25_000, "slice_tokens": 12_500, "notes_default_tokens": 40_000, "notes_min_sessions": 20, "notes_switch_runs": 3, **kw})


PRICE = SimpleNamespace(p=2e-6, po=10e-6, w=1.25, r=0.10)


def test_the_switch_fires_after_three_sessions_over_and_the_store_never_flips_back(layout, machinery, log):
    fig = HostFigures(sessions=0, calls_per_session=6.0, extraction_usd_per_session=0.017, source="consumer's default over 0 sessions")
    state = {"mode": "notes", "sessions": 0, "over_run": 0}
    small = decide(state, tokens=30_000, profile=PRICE, figures=fig, serve=serve())
    assert small["over"] is False and small["switch"] is False and small["rule"].startswith("default")
    for i in range(3):                                                        # the default rule: over 40k tokens, three sessions running
        d = decide(state, tokens=50_000, profile=PRICE, figures=fig, serve=serve())
        state = {**state, "sessions": d["sessions"], "over_run": d["over_run"]}
        assert d["switch"] is (i == 2)
    assert d["notes_usd"] == round(50_000 * 2e-6 * (1.25 + 0.1 * 5), 6)       # the switch's price for a block always loaded
    # priced, once the host has its data: below the core ceiling memory serves the same tokens plus extraction, so notes win
    fig20 = HostFigures(sessions=20, calls_per_session=6.0, extraction_usd_per_session=0.017, source="metered over 20 sessions")
    d = decide({"mode": "notes", "sessions": 20, "over_run": 2}, tokens=12_000, profile=PRICE, figures=fig20, serve=serve())
    assert d["rule"].startswith("priced") and d["over"] is False and d["over_run"] == 0
    d = decide({"mode": "notes", "sessions": 20, "over_run": 2}, tokens=200_000, profile=PRICE, figures=fig20, serve=serve())
    assert d["over"] is True and d["switch"] is True                          # far over the core: memory caps what it serves and pays the slice
    # one way: a store that switched stays in memory mode whatever the notes cost
    ns = NotesStore(layout, TENANT)
    ns.save_mode({"mode": "memory", "switched": DAY})
    store = MemoryStore(layout, ALLOW)
    assert session_start(layout, TENANT, store, log, principal_id="owner", profile=PRICE, serve=serve(), chars_per_token=4.0,
                         extraction_usd_default=0.017, today=DAY) is None


def test_session_start_serves_the_notes_then_switches_and_migrates_with_no_model_call(layout, machinery, log):
    lines, ep = session_with(layout, [("user", "Our bean supplier is Example Coffee Co."), ("user", "Update: the bean supplier is now Sample Roasters."),
                                      ("user", "Our accountant is Alice Example.")])
    ns = NotesStore(layout, TENANT)
    for value, quote, attr, day in (("Example Coffee Co", "bean supplier is Example Coffee Co", "bean_supplier", "2026-10-01"),
                                    ("Sample Roasters", "bean supplier is now Sample Roasters", "bean_supplier", "2026-10-03"),
                                    ("Alice Example", "accountant is Alice Example", "accountant", "2026-10-04")):
        e = entry(value, quote=quote, attribute=attr); e["as_of"] = day
        reason, anchor = check(e, lines, ep)
        assert reason is None
        e["source"]["episode"] = anchor
        ns.add(e, today=day)
    store = MemoryStore(layout, ALLOW)
    sv = serve(notes_default_tokens=10, notes_switch_runs=2)                 # tiny threshold: the second session switches
    first = session_start(layout, TENANT, store, log, principal_id="owner", profile=PRICE, serve=sv, chars_per_token=4.0,
                          extraction_usd_default=0.017, session_id="S1", today="2026-10-05")
    assert first["entries"] == 2 and first["text"].startswith("===== notes =====") and ns.mode()["over_run"] == 1
    second = session_start(layout, TENANT, store, log, principal_id="owner", profile=PRICE, serve=sv, chars_per_token=4.0,
                           extraction_usd_default=0.017, session_id="S2", today="2026-10-06")
    assert second is None and ns.mode()["mode"] == "memory" and ns.mode()["migrated"] == {"current": 2, "historical": 1, "unmapped": 0}
    recs = {store.get(TENANT, e["id"])[0].claim: store.get(TENANT, e["id"])[0] for e in store.list(TENANT)}
    assert len(recs) == 3
    cur = next(r for c, r in recs.items() if "Sample Roasters" in c)
    old = next(r for c, r in recs.items() if "Example Coffee" in c)
    assert cur.status.state == "probation" and cur.key == "company::-::bean_supplier" and cur.value_status == "current" and cur.actor.cls == "principal"
    assert old.status.state == "superseded" and old.status.superseded_by == cur.id and old.value_status == "historical"
    assert cur.anchor.episode == f"{ep}#L2-L2" and cur.anchor.quote == "bean supplier is now Sample Roasters"
    kinds = [r["kind"] for r in read_run(log.path, typed=False)]
    assert kinds.count("notes_switch") == 2 and "model_call" not in kinds       # no model call anywhere in the migration
    sw = [r for r in read_run(log.path, typed=False) if r["kind"] == "notes_switch"]
    assert [s["switch"] for s in sw] == [False, True] and sw[1]["notes_usd"] > 0 and sw[1]["memory_usd"] > 0
    # after the switch, the notes' unmapped branch: an entry with no anchor is left for extraction
    e = entry("x", attribute="loose"); e["source"]["episode"] = None
    from recant.notes import NoteEntry
    loose = NotesStore(layout, TENANT).add(e, today=DAY)
    assert migrate([(loose, [])], principal_id="owner") == {"records": [], "unmapped": [loose]}


# ---------------------------------------------------------------- batch, blocks, concurrent writers, tool output
def test_one_call_holds_ten_entries_each_checked_on_its_own(layout):
    from recant.notes import check_many
    lines, ep = session_with(layout, [("user", " ".join(f"Vendor V{i}'s payment terms are {10 + i} days." for i in range(9)) + " The policy has not been provided.")])
    es = [entry(f"{10 + i}", quote=f"Vendor V{i}'s payment terms are {10 + i} days", attribute=f"terms_{i}") for i in range(9)]
    es.append(entry("has not been provided", quote="policy has not been provided", attribute="policy"))
    out = check_many(es, lines, ep)
    assert len(out) == 10 and all(r is None and a == f"{ep}#L1-L1" for r, a in out[:9]) and out[9][0] == "absence"


def test_a_shared_block_renders_byte_identically_for_two_users_and_the_user_blocks_apart(layout):
    from recant.notes import render_blocks, is_shared
    team = NotesStore(layout, TENANT, "team:ops")
    team.add(entry("Sample & Partners", attribute="auditor"), today=DAY)
    for who, acct in (("user:alice", "Alice Example"), ("user:bob", "Carol Example")):
        NotesStore(layout, TENANT, who).add(entry(acct), today=DAY)
    a_text, a_n = render_blocks(layout, TENANT, ["user:alice", "team:ops"])
    b_text, b_n = render_blocks(layout, TENANT, ["user:bob", "team:ops"])
    shared_a = a_text.split("===== notes (team:ops) =====")[1]
    shared_b = b_text.split("===== notes (team:ops) =====")[1]
    assert shared_a == shared_b and "Sample & Partners" in shared_a            # the shared block: the same bytes for both
    assert "Alice Example" in a_text and "Alice Example" not in b_text and a_n == b_n == 2
    assert is_shared("team:ops") and is_shared("org") and not is_shared("user:alice") and not is_shared(None)
    assert (layout.notes_dir(TENANT) / "team_ops" / "entries.jsonl").exists()   # one directory per block


def test_two_agents_writing_one_key_in_one_window_are_served_as_unresolved_until_a_later_write(layout):
    ns = NotesStore(layout, TENANT, "team:ops")
    ns.add(entry("Alice Example"), today=DAY, agent="agent-x", ts="2026-10-05T10:00:00+00:00")
    ns.add(entry("Carol Example"), today=DAY, agent="agent-y", ts="2026-10-05T10:20:00+00:00")
    un = ns.unresolved()
    assert list(un) == ["company::-::accountant"] and (un["company::-::accountant"][0].value, un["company::-::accountant"][1].value) == ("Alice Example", "Carol Example")
    text_ = render(ns.current(), block="team:ops", unresolved=un)
    assert "accountant=unresolved Alice Example (agent-x 2026-10-05T10:00:00+00:00) | Carol Example (agent-y 2026-10-05T10:20:00+00:00)" in text_
    ns.add(entry("Alice Example"), today="2026-10-06", agent="agent-x", ts="2026-10-06T09:00:00+00:00")   # a later single write resolves it
    assert ns.unresolved() == {}
    cur, hist = ns.current()[0]
    assert cur.value == "Alice Example" and [h.value for h in hist] == ["Alice Example", "Carol Example"]
    # the same agent twice, or two agents outside the window, is a plain change
    ns2 = NotesStore(layout, TENANT, "user:alice")
    ns2.add(entry("A", attribute="x"), today=DAY, agent="agent-x", ts="2026-10-05T10:00:00+00:00")
    ns2.add(entry("B", attribute="x"), today=DAY, agent="agent-y", ts="2026-10-05T14:00:00+00:00")
    assert ns2.unresolved() == {}


def test_a_tool_note_finds_its_value_in_the_recorded_output(layout):
    from recant.notes import record_output
    # the CSV rows here and below are made up for the test: placeholder account ids, round amounts
    line = record_output("run_shell: `sed -n 18p reports/today.csv` → exit 0 (41 chars)", "17,2026-09-05,EX0001,south,misc,1000.00,ok\n", 2000)
    lines, ep = session_with(layout, [("user", "Look up row 17."), ("tool", line)])
    ok = entry("1000.00", kind="tool", quote=None, call="sed -n 18p reports/today.csv", attribute="row17_amount")
    assert check(ok, lines, ep) == (None, f"{ep}#L2-L3")                      # the output rides as a continuation line
    bad = entry("9999.99", kind="tool", quote=None, call="sed -n 18p reports/today.csv", attribute="row17_amount")
    assert check(bad, lines, ep)[0] == "value not in the tool's output"
    assert record_output("summary", "x" * 10, 4).endswith("xxxx … (+6 chars)") and record_output("summary", "out", 0) == "summary"


def test_a_composite_value_passes_when_every_part_is_in_the_output_and_a_quoteless_note_anchors_on_the_value(layout):
    """A quoteless note anchors on the value; a tool note's value must be in the recorded output."""
    from recant.notes import record_output, parts, source_ref
    line = record_output("run_shell: `awk 'NR==21' reports/today.csv` → exit 0 (41 chars)", "20,2026-09-05,EX0002,south,misc,2000.00,ok\n", 2000)
    lines, ep = session_with(layout, [("user", "Our accountant is Alice Example and the shop rent is 3,300 a month."), ("tool", line)])
    assert parts("EX0002, 2000.00") == ["ex0002", "2000.00"]
    composite = entry("EX0002, 2000.00", kind="tool", quote=None, call="awk 'NR==21' reports/today.csv", attribute="row20")
    assert check(composite, lines, ep) == (None, f"{ep}#L2-L3")                 # both parts in the output: kept
    off = entry("EX0002, 9999.99", kind="tool", quote=None, call="awk 'NR==21' reports/today.csv", attribute="row20")
    assert check(off, lines, ep)[0] == "value not in the tool's output"
    bare = entry("3,300", quote=None, attribute="rent")                       # no quote: the value must be in the principal's words
    assert check(bare, lines, ep) == (None, f"{ep}#L1-L1") and "shop rent is 3,300" in bare["source"]["quote"]
    missing = entry("4,000", quote=None, attribute="rent")
    assert check(missing, lines, ep)[0] == "value not in the principal's turns"
    ns = NotesStore(layout, TENANT)
    bare["source"]["episode"] = f"{ep}#L1-L1"
    e = ns.add(bare, today=DAY)
    assert source_ref(e) == "10-05 L1" and render(ns.current()).splitlines()[1] == "company/-: rent=3,300 (2026-10-05; 10-05 L1)"

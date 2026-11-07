"""A setting change is judged on the system prompt's stable start (0.4.1): the part the cache can reuse, the longest
leading part that stays the same from call to call. A per-request name or brief after a fixed opening is content, so
the job keeps one fingerprint and its levers are priced on its whole history; a template at the very top is said in
one line; a real change, where the fixed part itself changes, is still a change."""
import json

import pytest
import time

from recant import costshape as cs, wrap
from test_stranger import FakeSDK, TENANT

FIXED = "Follow the house style; cite the document and the page. " * 90        # about 1,300 tokens, fixed
OTHER = "Answer in French; keep every answer under a hundred words. " * 90      # a different fixed part
NAMES = ["Alice", "Bob", "Carol"]


def _run(layout, system_for, runs=24, model="claude-haiku-5-5"):
    client = wrap(FakeSDK(), store=layout, tenant=TENANT, job="agent")
    for i in range(runs):
        time.sleep(0.003)                                # tasks a few ms apart: the log's timestamps are in ms
        with client.run("writer", session=f"s{i}"):
            msgs = []
            for k in range(3):
                msgs.append({"role": "user", "content": f"turn {k} of task {i}"})
                client.messages.create(model=model, max_tokens=50, system=system_for(i), messages=list(msgs))
                msgs.append({"role": "assistant", "content": "ok"})
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    return cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                     since="30d")


def _changes(rep):
    return [c for c in rep.changes if c["what"].endswith("sends new settings")]


def test_a_template_after_a_fixed_opening_keeps_one_fingerprint(layout):
    rep = _run(layout, lambda i: FIXED + f"\nThe user is {NAMES[i % 3]}; their brief: task {i}.")
    assert len({c["setting_fp"] for c in rep.calls}) == 24                 # the whole prompt differs on every task
    st = cs.stable_starts(rep.calls)["writer"]
    assert len(FIXED) - 256 < st["start"] < st["full"]                      # stable to the fixed opening, not past it
    assert _changes(rep) == [] and rep.restrict("agent") is rep             # so: no setting change, the full history
    lv = next(lv for lv in cs.advise(rep) if lv.id == "breakpoint.add")
    assert lv.evidence["calls"] == 72                                       # priced on all 72 calls, not the last task's
    assert not any(lv.id.startswith("fix.system_order") for lv in cs.advise(rep))


def test_a_template_at_the_top_is_said_in_one_line_and_sized(layout):
    rep = _run(layout, lambda i: f"The user is {NAMES[i % 3]}; their brief: task {i}.\n" + FIXED)
    st = cs.stable_starts(rep.calls)["writer"]
    assert st["start"] == 0 and len(st["content"]) == 72                    # every prompt is content from the top
    assert _changes(rep) == []                                              # no stable start: nothing to change
    lv = next(lv for lv in cs.advise(rep) if lv.id == "fix.system_order.writer")
    assert "differ from their first ~0 tokens on, under claude-haiku-5-5's 512-token cache minimum" in lv.note
    assert "put the per-request part after the fixed instructions" in lv.note
    assert lv.evidence["shared_end_tokens"] >= (len(FIXED) - 256) / 4
    assert lv.saves_usd and lv.saves_usd > 0 and "would have saved about $" in lv.note
    assert lv.kind == "fix_cause" and not lv.safe                           # pointed out, never applied


def test_a_real_change_of_the_fixed_part_is_still_a_change(layout):
    rep = _run(layout, lambda i: (FIXED if i < 12 else OTHER) + f"\nThe user is {NAMES[i % 3]}; task {i}.")
    ch = _changes(rep)
    assert len(ch) == 1                                                     # one change, where the fixed part changed
    assert ch[0]["at"] == sorted(c["start"] for c in rep.calls)[36]        # task 12's first call, the first on OTHER
    since = {c["session"] for c in rep.restrict("agent").calls}            # priced since the change
    assert since == {f"s{i}" for i in range(12, 24)}


def test_a_prompt_sent_the_same_every_time_is_judged_as_before(layout):
    rep = _run(layout, lambda i: FIXED)
    st = cs.stable_starts(rep.calls)["writer"]
    assert st["start"] == st["full"] and _changes(rep) == []
    assert not any(lv.id.startswith("fix.system_order") for lv in cs.advise(rep))


def test_the_log_keeps_hashes_never_the_text(layout):
    _run(layout, lambda i: FIXED + f"\nThe user is {NAMES[i % 3]}.", runs=2)
    from recant.log import read_run
    recs = [r for p in layout.log_dir(TENANT).glob("*.jsonl") for r in read_run(p, typed=False) if r["kind"] == "model_call"]
    raw = json.dumps(recs)
    assert recs[0]["system_prefix"] and recs[0]["system_suffix"] and recs[0]["setting_fp_rest"]
    assert "house style" not in raw and "Alice" not in raw


def test_the_finding_goes_in_a_feedback_bundle_with_its_job_numbered(layout):
    from recant import feedback
    feedback._JOBS.clear()
    assert feedback.lever_id("fix.system_order.writer") == "fix.system_order.<job 1>"


def test_going_back_to_an_earlier_fixed_part_is_not_a_change(layout):
    rep = _run(layout, lambda i: (OTHER if 8 <= i < 16 else FIXED) + f"\nThe user is {NAMES[i % 3]}; task {i}.")
    assert len(_changes(rep)) == 1                                          # into OTHER; back to FIXED is known



# ---------------------------------------------------------------- the cases a review found (each failed before)
def _calls(layout, prompts, *, gap=0.0, model="claude-haiku-5-5"):
    """One call per (session, prompt) pair, in order; `gap` seconds between calls (the timestamps are rewritten)."""
    client = wrap(FakeSDK(), store=layout, tenant=TENANT, job="agent")
    for i, (session, system) in enumerate(prompts):
        time.sleep(0.002)
        with client.run("writer", session=session):
            client.messages.create(model=model, max_tokens=50, system=system,
                                   messages=[{"role": "user", "content": f"turn {i}"}])
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                    since="30d")
    if gap:
        t0 = min(c["start"] for c in rep.calls)
        for i, c in enumerate(sorted(rep.calls, key=lambda c: c["start"])):
            c["start"] = t0 + i * gap
    return rep


def _fixed(a, b, n=4):
    """Prompt a in n conversations, then prompt b in n more: two fixed prompts, each sent whole again and again."""
    return [(f"a{i}", a) for i in range(n)] + [(f"b{i}", b) for i in range(n)]


def test_replacing_a_fixed_prompt_is_a_change(layout):
    assert len(_changes(_calls(layout, _fixed("A" * 4096, "B" * 4096)))) == 1


def test_adding_instructions_to_a_fixed_prompt_is_a_change(layout):
    assert len(_changes(_calls(layout, _fixed("A" * 4096, "A" * 4096 + "B" * 4096)))) == 1


def test_cutting_instructions_from_a_fixed_prompt_is_a_change(layout):
    assert len(_changes(_calls(layout, _fixed("A" * 4096, "A" * 2048)))) == 1


def test_one_odd_brief_under_the_minimum_is_not_a_change(layout):
    x = [(f"x{i}", "F" * 1024 + "X" * 3072 + str(i)) for i in range(3)]
    rep = _calls(layout, x + [("y", "F" * 1024 + "Y" * 3072)] + [(f"z{i}", "F" * 1024 + "X" * 3072 + str(i + 5)) for i in range(3)])
    assert _changes(rep) == []


def test_coming_back_to_an_earlier_fixed_part_is_not_a_change_however_the_traffic_runs(layout):
    a0 = "A" * 4096 + "0"
    prompts = [("s0", a0)] + [(f"b{i}", "B" * 4096 + str(i)) for i in range(5)] + [(f"c{i}", a0 + f" brief {i}") for i in range(5)]
    assert len(_changes(_calls(layout, prompts))) == 1                      # into B; back to A is known


def test_a_prompt_logged_across_the_upgrade_is_not_a_change(layout):
    rep = _calls(layout, [(f"s{i}", FIXED) for i in range(3)])
    first = min(rep.calls, key=lambda c: c["start"])
    for k in ("setting_rest", "sys_prefix", "sys_suffix"):                  # as 0.4.0 logged it
        first[k] = None
    assert cs._fingerprint_changes(rep.calls) == []


def test_the_ending_sized_is_one_every_prompt_shares(layout):
    prompts = [(f"s{i}", str(i) + "F" * 4096) for i in range(4)] + [("s4", "4" + "G" * 4096)]
    lv = next(lv for lv in cs.advise(_calls(layout, prompts)) if lv.id == "fix.system_order.writer")
    assert lv.evidence["shared_end_tokens"] == 0 and lv.saves_usd is None and "the same in all" not in lv.note


def test_calls_a_day_apart_read_nothing_from_the_cache(layout, tmp_path):
    from recant.store import StoreLayout
    prompts = [(f"s{i}", str(i) + "F" * 4096) for i in range(3)]
    lv = next(lv for lv in cs.advise(_calls(layout, prompts, gap=86400)) if lv.id == "fix.system_order.writer")
    assert lv.saves_usd is None                                             # every call would write: no saving
    other = StoreLayout(tmp_path / "other")                                 # the same traffic, in a store of its own
    lv = next(lv for lv in cs.advise(_calls(other, prompts, gap=10)) if lv.id == "fix.system_order.writer")
    assert lv.saves_usd and lv.saves_usd > 0                                # within the lifetime: reads


def test_two_fixed_prompts_in_turn_are_not_called_uncacheable(layout):
    prompts = [(f"s{i}", ("A" if i % 2 else "B") * 4096) for i in range(8)]
    rep = _calls(layout, prompts)
    assert not any(lv.id.startswith("fix.system_order") for lv in cs.advise(rep))


def test_the_providers_own_calls_carry_the_hashes_too(tmp_path):
    from recant.provider import ModelRequest, Message, text
    from recant.log import read_run
    from recant.store import StoreLayout
    import recant.metering as m
    req = ModelRequest(model="claude-haiku-5-5", job="writer", system=(text(FIXED + "The user is Alice."),),
                       messages=(Message(role="user", content=(text("hi"),)),), max_tokens=10)
    parts = m.setting_parts(req)
    assert parts["system_prefix"] and parts["system_suffix"] and parts["setting_fp_rest"]
    assert "house style" not in json.dumps(parts) and "Alice" not in json.dumps(parts)


# ---------------------------------------------------------------- the cases a second review found (each failed before)
def _calls2(layout, rows, *, run_per_call=True):
    """rows: (session or None, system or None, model, max_tokens, batch); one run per call unless run_per_call=False."""
    import contextlib
    client = wrap(FakeSDK(), store=layout, tenant=TENANT, job="agent")
    whole = client.run("writer") if not run_per_call else contextlib.nullcontext()
    with whole:
        for i, (session, system, model, cap) in enumerate(rows):
            time.sleep(0.002)
            ctx = client.run("writer", session=session) if run_per_call else contextlib.nullcontext()
            with ctx:
                kw = {"system": system} if system is not None else {}
                client.messages.create(model=model, max_tokens=cap, messages=[{"role": "user", "content": f"turn {i}"}], **kw)
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    return cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                     since="30d")


H = "claude-haiku-5-5"


def test_a_cap_change_on_content_only_prompts_is_still_a_change(layout):
    rows = [(f"s{i}", str(i) + "F" * 4096, H, 50 if i < 2 else 500) for i in range(4)]
    assert len(_changes(_calls2(layout, rows))) == 1


def test_a_brief_repeated_in_a_template_family_is_not_a_change(layout):
    base = "F" * 4096
    rows = [("a1", base + " brief 1", H, 50), ("a2", base + " brief 2", H, 50)] + \
        [(f"b{i}", base + " brief A", H, 50) for i in range(2)] + [(f"c{i}", base + " brief B", H, 50) for i in range(2)]
    assert _changes(_calls2(layout, rows)) == []


def test_fixed_prompts_in_one_run_with_no_session_are_fixed(layout):
    rows = [(None, "A" * 4096, H, 50), (None, "A" * 4096, H, 50), (None, "B" * 4096, H, 50), (None, "B" * 4096, H, 50)]
    rep = _calls2(layout, rows, run_per_call=False)
    assert len(_changes(rep)) == 1
    assert not any(lv.id.startswith("fix.system_order") for lv in cs.advise(rep))


def test_a_later_model_does_not_rewrite_an_earlier_familys_heads(layout):
    rows = [(f"s{i}", "F" * 4096 + f" brief {i}", H, 50) for i in range(5)] + [("s5", "F" * 4096 + " brief 5", "claude-haiku-4-5", 50)]
    rep = _calls2(layout, rows)
    st = cs.stable_starts(rep.calls)["writer"]
    early = [c for c in rep.calls if c["model"] == H]
    assert all(st["heads"][id(c)][0] >= 4096 for c in early)                # the Haiku 5.5 family stands
    assert len(_changes(rep)) == 1                                          # the model change is one


def test_the_upgrade_seed_seeds_the_fixed_variant_too(layout):
    """The exact case: A logged by 0.4.0, A once by 0.4.1, then A+B twice: A opened two requests in all."""
    rows = [(f"s{i}", "A" * 4096, H, 50) for i in range(2)] + [(f"t{i}", "A" * 4096 + "B" * 4096, H, 50) for i in range(2)]
    rep = _calls2(layout, rows)
    first = min(rep.calls, key=lambda c: c["start"])
    for k in ("setting_rest", "sys_prefix", "sys_suffix"):
        first[k] = None                                                     # the first call as 0.4.0 logged it
    assert len(cs._fingerprint_changes(rep.calls)) == 1                     # A to A+B is still a change


def _content(n, model=H):
    return [(f"s{i}", str(i) + "F" * 4096, model, 50) for i in range(n)]


def test_the_hour_lifetime_pays_its_own_write_premium(layout):
    import dataclasses
    rep = _calls2(layout, _content(2))
    rep.host = dataclasses.replace(rep.host, prefix_ttl="1h")
    lv = next(lv for lv in cs.advise(rep) if lv.id == "fix.system_order.writer")
    assert lv.saves_usd is None                                             # one read does not pay a 2x write


def test_a_cache_entry_is_per_model(layout):
    rows = [("s0", "0" + "F" * 4096, H, 50), ("s1", "1" + "F" * 4096, "claude-sonnet-5-5", 50),
            ("s2", "2" + "F" * 4096, "claude-opus-5-5", 50)]
    for lv in cs.advise(_calls2(layout, rows)):
        assert not (lv.id.startswith("fix.system_order") and lv.saves_usd)


def test_batch_calls_are_sized_at_the_batch_rate(layout):
    rep = _calls2(layout, _content(4))
    full = next(lv for lv in cs.advise(rep) if lv.id == "fix.system_order.writer").saves_usd
    for c in rep.calls:
        c["batch"] = True
    half = next(lv for lv in cs.advise(rep) if lv.id == "fix.system_order.writer").saves_usd
    p = rep.profiles[H]
    assert half == pytest.approx(full * p.batch)


def test_no_line_once_the_latest_call_sends_no_system_prompt(layout):
    rows = _content(3) + [("s9", None, H, 50)]
    assert not any(lv.id.startswith("fix.system_order") for lv in cs.advise(_calls2(layout, rows)))



# ---------------------------------------------------------------- the cases a third review found (each failed before)
def test_a_content_calls_new_cap_is_one_change_not_two(layout):
    rows = [("a0", "A" * 4096, H, 50), ("a1", "A" * 4096, H, 50), ("z", "z" + "F" * 4096, H, 500),
            ("a2", "A" * 4096, H, 500), ("a3", "A" * 4096, H, 500)]
    assert len(_changes(_calls2(layout, rows))) == 1


def test_the_providers_conversation_turns_are_not_fresh_openings(machinery, log):
    from recant.provider import Message, ModelRequest, Usage, text as t_
    from recant.provider.base import Raw
    from recant.provider.fake import FakeProvider
    from recant.log import read_run

    p = FakeProvider(machinery, log, lambda model, r: Raw((t_("ok"),), "end_turn", Usage(input=1200, output=5)))
    for conv, opening in (("c0", "0"), ("c1", "1")):
        msgs: tuple = ()
        for k in range(3):
            msgs = msgs + (Message("user", (t_(f"turn {k}"),)),)
            p.call(ModelRequest(model="claude-haiku-5-5", job="writer", session=conv, max_tokens=10, messages=msgs,
                                system=(t_(opening + "F" * 4096),)))
            msgs = msgs + (Message("assistant", (t_("ok"),)),)
    recs = [r for r in read_run(log.path, typed=False) if r["kind"] == "model_call"]
    assert [r["extends_previous"] for r in recs] == [None, True, True, None, True, True]
    calls, _ = cs.calls_from_records(recs)
    st = cs.stable_starts(calls)["writer"]
    assert all(st["heads"][id(c)][0] == 0 for c in calls)                  # both prompts are content, opened once each
    assert cs._fingerprint_changes(calls) == []



# ---------------------------------------------------------------- the cases a fourth review found (each failed before)
def test_the_saving_is_sized_across_conversations_never_above_the_bill(layout):
    rep = _run(layout, lambda i: f"The user is {NAMES[i % 3]}; their brief: task {i}.\n" + FIXED)   # 24 tasks, 3 turns
    lv = next(lv for lv in cs.advise(rep) if lv.id == "fix.system_order.writer")
    assert lv.evidence["conversations"] == 24 and lv.evidence["calls"] == 72
    assert 0 < lv.saves_usd / rep.per_week < rep.job_bill({"writer"})       # a part of the bill, never more
    assert "no conversation can reuse another's" in lv.note and "over these 24 conversations" in lv.note


def test_the_providers_batched_conversation_turns_are_not_fresh_openings(machinery, log):
    from recant.provider import Message, ModelRequest, Usage, text as t_
    from recant.provider.base import Raw
    from recant.provider.fake import FakeProvider
    from recant.log import read_run

    p = FakeProvider(machinery, log, lambda model, r: Raw((t_("ok"),), "end_turn", Usage(input=1200, output=5)))
    hist = {"c0": (), "c1": ()}
    for k in range(3):
        reqs = []
        for conv, opening in (("c0", "0"), ("c1", "1")):
            hist[conv] = hist[conv] + (Message("user", (t_(f"turn {k}"),)),)
            reqs.append(ModelRequest(model="claude-haiku-5-5", job="writer", session=conv, max_tokens=10,
                                     messages=hist[conv], system=(t_(opening + "F" * 4096),), custom_id=f"{conv}-{k}"))
            hist[conv] = hist[conv] + (Message("assistant", (t_("ok"),)),)
        list(p.batch_results(p.submit_batch(reqs)))
    recs = sorted((r for r in read_run(log.path, typed=False) if r["kind"] == "model_call"), key=lambda r: r["custom_id"])
    assert [r["extends_previous"] for r in recs] == [None, True, True, None, True, True]
    calls, _ = cs.calls_from_records(recs)
    assert cs._fingerprint_changes(calls) == []



# ---------------------------------------------------------------- the cases a fifth review found (each failed before)
def test_a_conversation_that_switches_its_system_prompt_opens_the_new_one(layout):
    """Both conversations grow their histories while switching to A, then to B: A and B each open in two
    conversations, so they are fixed prompts, and A to B is a change."""
    client = wrap(FakeSDK(), store=layout, tenant=TENANT, job="agent")
    hist = {"c0": [], "c1": []}
    for k, system_for in enumerate((lambda c: c[-1] + "F" * 4096, lambda c: "A" * 4096, lambda c: "B" * 4096)):
        for conv in ("c0", "c1"):
            time.sleep(0.002)
            hist[conv].append({"role": "user", "content": f"turn {k}"})
            with client.run("writer", session=conv):
                client.messages.create(model=H, max_tokens=50, system=system_for(conv), messages=list(hist[conv]))
            hist[conv].append({"role": "assistant", "content": "ok"})
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                    since="30d")
    st = cs.stable_starts(rep.calls)["writer"]
    later = sorted(rep.calls, key=lambda c: c["start"])[2:]
    assert all(st["heads"][id(c)][0] >= 4096 for c in later)                # A and B are fixed prompts
    assert len(_changes(rep)) == 1                                          # A to B


def test_a_rejected_batch_leaves_the_conversation_history_as_it_was(machinery, log):
    import pytest as _pytest
    from recant.provider import Message, ModelRequest, Usage, text as t_
    from recant.provider.base import Raw
    from recant.provider.fake import FakeProvider

    p = FakeProvider(machinery, log, lambda model, r: Raw((t_("ok"),), "end_turn", Usage(input=10, output=1)))
    one = (Message("user", (t_("turn 0"),)),)
    bad = [ModelRequest(model=H, job="writer", session=s, max_tokens=5, messages=one, custom_id="same") for s in ("a", "b")]
    with _pytest.raises(ValueError):
        p.submit_batch(bad)
    assert p._last == {}                                                    # nothing was sent, nothing remembered



def test_dropping_the_system_prompt_between_uses_counts_each_return_as_an_opening(layout):
    """One conversation, its history growing: A, none, A, B, none, B. A and B each open twice, so both are fixed
    prompts, and A to B is a change."""
    client = wrap(FakeSDK(), store=layout, tenant=TENANT, job="agent")
    hist = []
    with client.run("writer", session="c"):
        for k, system in enumerate(["A" * 4096, None, "A" * 4096, "B" * 4096, None, "B" * 4096]):
            time.sleep(0.002)
            hist.append({"role": "user", "content": f"turn {k}"})
            kw = {"system": system} if system else {}
            client.messages.create(model=H, max_tokens=50, messages=list(hist), **kw)
            hist.append({"role": "assistant", "content": "ok"})
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                    since="30d")
    st = cs.stable_starts(rep.calls)["writer"]
    with_system = [c for c in rep.calls if c.get("sys_prefix")]
    assert len(with_system) == 4 and all(st["heads"][id(c)][0] == 4096 for c in with_system)
    b_first = sorted(with_system, key=lambda c: c["start"])[2]["start"]
    assert any(ch["at"] == b_first for ch in _changes(rep))                 # A to B, at B's first call

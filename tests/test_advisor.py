"""The cost advisor over Recant's own log — the prefix lifetime priced both ways, apply only past the
floor and with enough runs, the check prices the same calls both ways, a lever that does not deliver is reverted and
not retried on the same shape; the five detectors; the post-reply write sized and left unbuilt below 2%; qac; the
`recant cost` CLI."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from recant import costshape as cs
from recant.cli import main as recant_main, run_cost
from recant_fixtures import TENANT

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
HOST = cs.HostShape(split_on="turn", agent_jobs=("turn",), agent_run_jobs=("session",), background_jobs=("nominate", "extract"),
                    background_run_jobs=("extract_live",), prefix_ttl="1h", memory_budgets={"core": 25000, "slice": 12500},
                    carries_out=("ttl.prefix",))
MODEL = "claude-opus-5-5"


class Log:
    """Raw log records with chosen timestamps: the gaps are the point of these tests."""

    def __init__(self, layout, run, job, t, **params):
        self.path = layout.log_dir(TENANT) / f"{run}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run, self.seq, self.lines = run, 0, []
        self.emit("run_started", t, job=job, params=params)

    def emit(self, kind, t, **f):
        self.seq += 1
        self.lines.append({"v": 1, "ts": t.isoformat().replace("+00:00", "Z"), "tenant": TENANT, "run": self.run, "seq": self.seq,
                           "artifact": "a", "kernel": "k", "kind": kind, **f})

    def call(self, t, *, job="turn", read=0, write=0, out=200, session=None, core="c1", stop="end_turn", batch=None, fp=None):
        segs = [{"at": "system.0", "hash": "head", "ttl": "1h"}, {"at": "system.1", "hash": core, "ttl": "1h"}]
        self.emit("model_call", t, model_id=MODEL, job=job, request_hash="h", latency_ms=2000, session=session, segments=segs,
                  usage={"input": 5, "output": out, "cache_read": read, "cache_write": write}, cost_usd=0.01, stop_reason=stop,
                  batch_id=batch, setting_fp=fp or f"{job}-v1")

    def close(self, t):
        self.emit("run_finished", t, stopped_by="done", spent_usd=0.0, examined=0, promoted=0)
        self.path.write_text("".join(json.dumps(x) + "\n" for x in self.lines))


def chat(layout, *, sessions=10, turns=3, gap=timedelta(hours=2), prefix=30000, core_change_at=None, start=T0):
    """Sessions of `turns` turns each, `gap` apart; every turn three calls. The first call of a turn reads the prefix
    when its previous use is within the lifetime the host caches at (1h here), else writes it."""
    t, last = start, None
    for s in range(sessions):
        lg = Log(layout, f"S{s:03d}", "session", t, session=f"sess{s}")
        for k in range(turns):
            warm = last is not None and (t - last) <= timedelta(hours=1)
            core = "c2" if core_change_at == (s, k) else "c1"
            lg.call(t, read=prefix if warm and core == "c1" else 1900, write=2000 if warm and core == "c1" else prefix, session=f"sess{s}", core=core)
            lg.call(t + timedelta(seconds=20), read=prefix + 2000, write=500, session=f"sess{s}", core=core)
            lg.call(t + timedelta(seconds=40), read=prefix + 2500, write=400, session=f"sess{s}", core=core)
            last = t + timedelta(seconds=42)
            lg.emit("turn", t + timedelta(seconds=45), session=f"sess{s}", served={"core_tokens_est": 25000, "slice_tokens_est": 9000})
            lg.emit("served", t + timedelta(seconds=1), session=f"sess{s}", ref=f"m{k}", model_id=MODEL, core_tokens=25000,
                    slice_tokens=9000, measured=True)
            t += gap
        lg.close(t)
    return t


def test_the_prefix_lifetime_is_priced_both_ways_on_the_measured_gaps(layout):
    end = chat(layout, gap=timedelta(hours=2))                         # every turn start is over an hour after the last
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    lv = cs.lever_ttl_prefix(rep)
    assert lv.action == {"ttl": "5m", "was": "1h"} and lv.saves_usd > 0 and lv.evidence["over_60m_or_new"] == 30
    near = cs.HostShape(**{**vars(HOST), "prefix_ttl": "5m"})
    end2 = chat(layout, sessions=10, gap=timedelta(minutes=20), start=end + timedelta(days=1))
    rep2 = cs.report(layout, TENANT, near, since=(end + timedelta(days=1)).isoformat(), until=end2.isoformat())
    lv2 = cs.lever_ttl_prefix(rep2)
    assert lv2.action == {"ttl": "1h", "was": "5m"} and lv2.saves_usd > 0                    # 20-minute gaps: the long lifetime pays


def test_apply_needs_the_floor_and_twenty_runs_writes_the_overlay_and_a_log_line(layout):
    end = chat(layout, sessions=3, gap=timedelta(hours=2))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    lv = cs.lever_ttl_prefix(rep)
    ok, why = cs.eligible(rep, lv)
    assert not ok and "under 20" in why                              # 9 runs
    end = chat(layout, sessions=8, gap=timedelta(hours=2), start=end)
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    lv = cs.lever_ttl_prefix(rep)
    assert cs.eligible(rep, lv)[0]
    cs.apply(layout, TENANT, rep, lv, at=end.timestamp())
    assert cs.overlay(layout, TENANT)["ttl.prefix"]["action"] == {"ttl": "5m", "was": "1h"}
    (e,) = cs.ledger(layout, TENANT)
    assert e["event"] == "applied" and "check after 50 runs" in e["note"]
    lv.state = "available"
    lv.share = 0.001
    assert "under 1%" in cs.eligible(rep, lv)[1]


def test_the_check_prices_the_same_calls_both_ways_and_a_lever_that_does_not_deliver_is_reverted(layout):
    end = chat(layout, sessions=12, gap=timedelta(hours=2))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    lv = cs.lever_ttl_prefix(rep)
    cs.apply(layout, TENANT, rep, lv, at=end.timestamp())
    # after it: turns come back every 20 minutes, where the 1h lifetime it replaced would have read the prefix
    after = chat(layout, sessions=6, turns=10, gap=timedelta(minutes=20), start=end + timedelta(minutes=1))
    out = cs.nightly(layout, TENANT, cs.HostShape(**{**vars(HOST), "prefix_ttl": "5m"}), now=after.timestamp() + 60, apply_safe=False)
    checked = [e for e in out["events"] if e["event"] == "checked"]
    assert checked and checked[0]["realized_usd"] < 0                 # priced both ways on the same calls: it cost money
    assert any(e["event"] == "reverted" for e in out["events"]) and "ttl.prefix" not in cs.overlay(layout, TENANT)
    rep2 = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    lv2 = cs.lever_ttl_prefix(rep2)
    led = cs.ledger(layout, TENANT)
    led[-1]["shape_sig"] = cs._shape_sig(rep2, "agent")               # the same shape as when it was reverted
    assert not cs.eligible(rep2, lv2, ledger_=led)[0]
    assert (layout.costshape_dir(TENANT) / "reports").exists()


def test_a_cached_segment_changed_within_a_session_is_found_with_where(layout):
    end = chat(layout, sessions=2, turns=3, gap=timedelta(minutes=2), core_change_at=(0, 1))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    lv = cs.detect_segment_changes(rep)
    assert lv.kind == "fix_cause" and lv.evidence["cases"][0]["at"] == "system.1" and lv.saves_usd > 0


def test_duplicate_background_runs_output_caps_and_batching(layout):
    for i in range(2):                                                # the same day extracted twice, direct
        lg = Log(layout, f"X{i}", "extract_live", T0 + timedelta(hours=i), day="2026-09-30")
        lg.call(T0 + timedelta(hours=i, minutes=1), job="nominate", write=0, out=8000, stop="max_tokens")
        lg.call(T0 + timedelta(hours=i, minutes=2), job="extract", out=900)
        lg.close(T0 + timedelta(hours=i, minutes=3))
    _session_at(layout, "S-next", T0 + timedelta(hours=10))           # the next session, hours later: the slack batching needs
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(days=1)).isoformat())
    dup = cs.detect_duplicate_background(rep)
    assert dup and dup.evidence["runs"][0]["day"] == "2026-09-30"
    (cap,) = cs.detect_output_cap(rep)
    assert cap.evidence["stopped"] == 2 and cap.id == "fix.output_cap.nominate"
    (b,) = cs.lever_batch(rep)
    assert b.id == "batch.extract_live" and b.evidence["direct_runs"] == 2 and b.saves_usd > 0
    b.share = 0.5
    assert cs.eligible(rep, b)[1] == "the host does not carry this lever out yet"           # not in the host's carries_out


def test_a_trimmed_result_fetched_again_is_counted(layout):
    lg = Log(layout, "S1", "session", T0, session="s")
    lg.call(T0, session="s", write=30000)
    for i, (cut, tg) in enumerate(((True, ["aa"]), (False, ["aa"]), (False, ["bb"]))):
        lg.emit("tool_call", T0 + timedelta(seconds=i + 1), session="s", name="run_shell", targets=tg,
                bytes_fetched=20000 if cut else 100, bytes_to_model=8000 if cut else 100)
    lg.emit("turn", T0 + timedelta(seconds=10), session="s")
    lg.close(T0 + timedelta(seconds=11))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(hours=1)).isoformat())
    lv = cs.detect_refetch(rep)
    assert lv.evidence["cut"] == 1 and lv.evidence["refetched_after_cut"] == 1 and lv.state == "host_advice"


def test_the_post_reply_write_is_sized_and_left_unbuilt_below_two_percent(layout):
    end = chat(layout, sessions=6, gap=timedelta(hours=3))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    lv = cs.lever_postreply_write(rep)
    assert lv.state == "not_worth_building" and lv.saves_usd < 0     # nothing comes back within the hour: it only costs


def test_qac_and_the_cli(layout, capsys):
    assert cs.qac(12.0, 80, 120) == {"usd_per_correct": 0.15, "usd_per_question": 0.1, "correct": 80, "asked": 120}
    end = chat(layout, sessions=2)
    text_ = run_cost(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    assert text_.startswith("recant cost:") and "ttl.prefix" in text_
    secs = [line for line in text_.splitlines() if line.startswith("== ")]
    assert secs == ["== cost ==", "== quality ==", "== latency ==", "== health =="]     # the sections, in order (latency)
    cost = text_.split("== cost ==")[1].split("== quality ==")[0]
    assert "automatic" in next(line for line in cost.splitlines() if "ttl.prefix" in line)
    assert "predicted against realized: no applied lever has been checked yet" in cost   # the bad news is said
    quality = text_.split("== quality ==")[1].split("== latency ==")[0]
    assert "memory served: median" in quality and "what memory adds or costs is not measured here" in quality
    recant_main(["cost", "--store", str(layout.root), "--tenant", TENANT, "--since", T0.isoformat(), "--until", end.isoformat(), "--json"])
    assert "levers" in json.loads(capsys.readouterr().out)


def test_a_rerun_after_a_run_that_wrote_nothing_is_not_a_duplicate(layout):
    def night(i, promoted):
        lg = Log(layout, f"N{i}", "extract_live", T0 + timedelta(hours=i), day="2026-10-06")
        lg.call(T0 + timedelta(hours=i, minutes=1), job="nominate", out=900)
        lg.emit("run_finished", T0 + timedelta(hours=i, minutes=2), stopped_by="done", spent_usd=0.25, examined=2, promoted=promoted)
        lg.path.write_text("".join(json.dumps(x) + "\n" for x in lg.lines))
    night(0, 0)                                                  # spent and wrote nothing
    night(1, 12)                                                 # the rerun
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(days=1)).isoformat())
    assert cs.detect_duplicate_background(rep) is None
    night(2, 3)                                                  # a third run after one that wrote something: a duplicate
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(days=1)).isoformat())
    dup = cs.detect_duplicate_background(rep)
    assert dup and dup.evidence["reruns_after_a_failed_run"] == 1 and len(dup.evidence["runs"]) == 1


def test_a_batch_lever_whose_deadline_is_missed_trips_its_guard(layout):
    host = cs.HostShape(**{**vars(HOST), "carries_out": ("ttl.prefix", "batch.extract_live")})
    lg = Log(layout, "B0", "extract_live", T0, day="2026-10-01")
    lg.call(T0 + timedelta(minutes=1), job="nominate", batch="b1")
    lg.path.write_text("".join(json.dumps(x) + "\n" for x in lg.lines))           # never finished
    lg = Log(layout, "B1", "extract_live", T0 + timedelta(hours=5), day="2026-10-01")
    lg.call(T0 + timedelta(hours=5, minutes=1), job="nominate")                     # the direct rerun
    lg.close(T0 + timedelta(hours=5, minutes=2))
    rep = cs.report(layout, TENANT, host, since=(T0 - timedelta(hours=1)).isoformat(), until=(T0 + timedelta(days=8)).isoformat())
    res = cs.check("batch.extract_live", rep, {"applied_at": (T0 - timedelta(minutes=1)).timestamp(), "action": {"batch": True}})
    assert res["guard"] == "1 batch deadlines missed"


def test_an_evidence_run_pairs_the_hosts_harness_item_by_item_and_respects_its_ceiling(layout):
    said = []
    items = {f"y{i}": "yield" for i in range(10)} | {f"n{i}": "noise" for i in range(5)}

    def harness(setting):
        small = setting["core"] < 10000
        return {i: (t, not (small and i in ("y1", "y2", "n0")) or (small and i == "y9")) for i, t in items.items()}, 2.0
    out = cs.evaluate("memory.budgets", current={"core": 25000}, candidate={"core": 5000}, ceiling_usd=10, estimate_usd=4.0,
                      harness=harness, say=said.append, layout=layout, tenant=TENANT)
    assert said and "estimated $4.00" in said[0] and out["spent_usd"] == 4.0
    y, n = out["by_type"]["yield"], out["by_type"]["noise"]
    assert (y["better"], y["worse"], n["worse"]) == (0, 2, 1) and y["interval"][0] < y["delta"] < y["interval"][1]
    assert cs.ledger(layout, TENANT)[-1]["event"] == "evidence"
    refused = cs.evaluate("memory.budgets", current={}, candidate={}, ceiling_usd=3, estimate_usd=4.0, harness=harness, say=said.append)
    assert refused["started"] is False and "over the $3.00 ceiling" in refused["why"]


def test_a_replayed_evidence_run_is_judged_blind_leaves_tool_runs_out_and_stops_at_its_ceiling():
    runs = [{"id": i, "used_tools": i % 4 == 0} for i in range(20)]
    seen = []

    def replay(run, setting):
        return f"{setting['h']}:{run['id']}", 0.1

    def judge(first, second):
        seen.append((first, second))
        return ("a" if first.startswith("full") else "b"), 0.05      # always prefers the full history, wherever it sits
    out = cs.evaluate("handoff.session", current={"h": "full"}, candidate={"h": "300"}, ceiling_usd=100, estimate_usd=1,
                      replay=replay, judge=judge, runs=runs, say=lambda *_: None)
    p = out["pairwise"]
    assert p["left_out_with_tools"] == 5 and p["share_with_tools"] == 0.25 and p["n"] == 15 and p["worse"] == 15
    assert {s[0].split(":")[0] for s in seen} == {"full", "300"}      # the order was shuffled: blind
    out = cs.evaluate("handoff.session", current={"h": "full"}, candidate={"h": "300"}, ceiling_usd=1.0, estimate_usd=1,
                      replay=replay, judge=judge, runs=runs, say=lambda *_: None)
    assert "ceiling was reached" in out["stopped"] and out["pairwise"]["n"] < 15


def test_the_batch_check_credits_only_the_nights_the_lever_changed(layout):
    """A night started in the batch window was batched before the lever too; only one the host marks as changed by
    the lever (a late night it batched) counts toward the realized saving."""
    host = cs.HostShape(**{**vars(HOST), "lever_runs": {"batch.extract_live": {"started_by": "watchdog:batch_lever"}}})
    for i, by in enumerate(("watchdog", "watchdog:batch_lever")):
        lg = Log(layout, f"L{i}", "extract_live", T0 + timedelta(days=i), day=f"2026-10-0{i + 1}", started_by=by)
        lg.call(T0 + timedelta(days=i, minutes=1), job="nominate", batch="b", out=4000)
        lg.close(T0 + timedelta(days=i, minutes=30))
    rep = cs.report(layout, TENANT, host, since=(T0 - timedelta(hours=1)).isoformat(), until=(T0 + timedelta(days=3)).isoformat())
    one_night = cs.LIST_PRICES[MODEL].po * 4000 + cs.LIST_PRICES[MODEL].p * 5      # a direct night's price of the one call
    res = cs.check("batch.extract_live", rep, {"applied_at": (T0 - timedelta(minutes=1)).timestamp(), "action": {"batch": True}})
    assert res["realized_usd"] == pytest.approx(one_night * 0.5, rel=1e-3)          # the late night only, at half price saved


def test_levers_are_priced_only_since_the_last_change_to_their_jobs_settings(layout):
    """Batching must not be sized on nights from before a rule changed. Direct nights before a logged setting change do
    not count; nor do agent runs before one for the agent's levers; the lever says where its window starts."""
    for i in range(6):                                            # six direct nights, then the host changes its setup
        lg = Log(layout, f"D{i}", "extract_live", T0 + timedelta(days=i), day=f"2026-09-{20 + i}")
        lg.call(T0 + timedelta(days=i, minutes=1), job="nominate", out=3000)
        lg.close(T0 + timedelta(days=i, minutes=5))
        _session_at(layout, f"S{i}", T0 + timedelta(days=i, hours=9))      # the day's chat, hours after the night
    ch = Log(layout, "CH", "deploy", T0 + timedelta(days=6, hours=1))
    ch.emit("setting_change", T0 + timedelta(days=6, hours=1), jobs=["extract_live"], what="make update: the night starts in the window")
    ch.close(T0 + timedelta(days=6, hours=1, seconds=1))
    for i in range(6, 9):                                         # batched nights after it
        lg = Log(layout, f"D{i}", "extract_live", T0 + timedelta(days=i, hours=2), day=f"2026-09-{20 + i}")
        lg.call(T0 + timedelta(days=i, hours=2, minutes=1), job="nominate", out=3000, batch="b")
        lg.close(T0 + timedelta(days=i, hours=2, minutes=5))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(days=10)).isoformat())
    assert cs.lever_batch(rep)                                    # the whole window: the old direct nights make a lever
    levers = cs.advise(rep)
    assert not [lv for lv in levers if lv.id.startswith("batch.")]       # since the change: nothing left to batch
    start, what = rep.window_start("extract_live")
    assert what.startswith("make update") and start == (T0 + timedelta(days=6, hours=1)).timestamp()


def _session_at(layout, name, t):
    """An agent run starting at t: what a background run's output is next needed by (a background job's slack)."""
    lg = Log(layout, name, "session", t, session=name)
    lg.call(t, session=name, read=1000)
    lg.emit("turn", t + timedelta(seconds=30), session=name)
    lg.close(t + timedelta(seconds=31))


def _night(layout, i, t, *, fp, batch=None):
    lg = Log(layout, f"N{i:02d}", "extract_live", t, day=f"2026-09-{10 + i}")
    lg.call(t + timedelta(minutes=1), job="nominate", out=3000, fp=fp, batch=batch)
    lg.close(t + timedelta(minutes=5))


def test_an_extraction_prompt_change_leaves_the_chats_window_running(layout):
    """A job's window restarts when that job sends settings it never sent before; other jobs' windows run on."""
    end = chat(layout, sessions=4, gap=timedelta(hours=2))
    for i in range(4):
        _night(layout, i, T0 + timedelta(days=i, hours=3), fp="nominate-v1")
    _night(layout, 4, T0 + timedelta(days=4, hours=3), fp="nominate-v2")           # a new extraction prompt
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(days=6)).isoformat())
    assert rep.window_start("extract_live")[1] == "nominate sends new settings"
    assert rep.window_start("extract_live")[0] == (T0 + timedelta(days=4, hours=3, minutes=1)).timestamp() - 2   # its start: ts less latency
    assert rep.window_start("agent") == (rep.since, "")                              # the chat's window runs on


def test_a_deploy_that_changes_nothing_a_job_sends_restarts_nothing(layout):
    """A commit-only deploy logs nothing and the jobs send what they sent: no window restarts. A job that goes back to
    settings it already uses (another model now and then) does not restart either."""
    end = chat(layout, sessions=4, gap=timedelta(hours=2))
    for i in range(3):
        _night(layout, i, T0 + timedelta(days=i, hours=3), fp="nominate-v1")
    _night(layout, 3, T0 + timedelta(days=3, hours=3), fp="nominate-v2")
    _night(layout, 4, T0 + timedelta(days=4, hours=3), fp="nominate-v1")            # back to known settings
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(days=6)).isoformat())
    assert [c["what"] for c in rep.changes] == ["nominate sends new settings"]      # once, at v2; not again at v1
    assert rep.window_start("agent") == (rep.since, "")


def test_a_host_change_naming_a_job_restarts_only_it_and_one_naming_none_restarts_all(layout):
    end = chat(layout, sessions=3, gap=timedelta(hours=2))
    _night(layout, 0, T0 + timedelta(hours=3), fp="nominate-v1")
    lg = Log(layout, "H1", "deploy", T0 + timedelta(hours=7))
    lg.emit("setting_change", T0 + timedelta(hours=7), jobs=["turn"], what="served budget changed")
    lg.close(T0 + timedelta(hours=7, seconds=1))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    assert rep.window_start("agent")[1] == "served budget changed" and rep.window_start("extract_live") == (rep.since, "")
    lg = Log(layout, "H2", "deploy", T0 + timedelta(hours=8))
    lg.emit("setting_change", T0 + timedelta(hours=8), jobs=[], what="everything")
    lg.close(T0 + timedelta(hours=8, seconds=1))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    assert rep.window_start("agent")[1] == rep.window_start("extract_live")[1] == "everything"


def test_an_applied_lever_starts_a_new_window_for_its_job(layout):
    end = chat(layout, sessions=12, gap=timedelta(hours=2))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    lg = Log(layout, "ADV", "costshape", end)
    lg.emit("advice", end, lever="ttl.prefix", lever_kind="applied", event="applied", action={"ttl": "5m"})
    lg.close(end + timedelta(seconds=1))
    after = chat(layout, sessions=2, gap=timedelta(hours=2), start=end + timedelta(minutes=5))
    rep2 = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=after.isoformat())
    assert rep2.window_start("agent")[1] == "applied ttl.prefix"
    lv = next(lv for lv in cs.advise(rep2) if lv.id == "ttl.prefix")
    assert lv.evidence["starts"] == 6 and lv.evidence["since_change"] == "applied ttl.prefix"    # only the 6 runs after it


def test_breakpoints_where_a_host_sets_none_are_sized_and_checked_both_ways(layout):
    """calls with no cache activity whose requests only append: the previous context would be read at the
    cache-read rate; the check prices the same calls as measured and as uncached."""
    from recant.metering import Meter
    host = cs.HostShape(agent_jobs=("agent",), carries_out=("breakpoint.add",))
    m = Meter(layout, TENANT)
    for t in range(25):
        m.run("task", task=t)
        msgs = [{"role": "system", "content": "policy"}, {"role": "user", "content": f"task {t}"}]
        for k in range(10):
            m.record(job="agent", model=MODEL, usage={"input": 8000 + 500 * k, "output": 200}, session=f"t{t}", messages=msgs)
            msgs = msgs + [{"role": "assistant", "content": f"a{k}"}, {"role": "tool", "content": f"r{k}"}]
    m.close()
    rep = cs.report(layout, TENANT, host, since="30d")
    lv = cs.lever_breakpoint_add(rep)
    assert lv and lv.saves_usd > 0 and lv.share > 0.5 and lv.evidence["calls"] == 250 and cs.eligible(rep, lv)[0]
    res = cs.check("breakpoint.add", rep, {"applied_at": 0, "action": lv.action, "predicted_usd": lv.saves_usd})
    assert res["realized_usd"] == pytest.approx(0, abs=1e-9)          # nothing was cached yet: measured equals uncached


def test_breakpoints_are_sized_per_session_when_one_log_run_carries_several(layout, tmp_path):
    """Two sessions interleaved in one log run size the same as each in its own run: a call extends its own
    session's previous request, never the other session's."""
    from recant.metering import Meter
    from recant.store import StoreLayout
    host = cs.HostShape(agent_jobs=("agent",), carries_out=("breakpoint.add",))

    def calls(m, t, k, msgs):
        m.record(job="agent", model=MODEL, usage={"input": 8000 + 500 * k + 3000 * t, "output": 200}, session=f"t{t}", messages=msgs)
        return msgs + [{"role": "assistant", "content": f"a{k}"}, {"role": "tool", "content": f"r{k}"}]

    one = Meter(layout, TENANT)
    one.run("all")
    msgs = {t: [{"role": "user", "content": f"task {t}"}] for t in range(2)}
    for k in range(10):
        for t in range(2):
            msgs[t] = calls(one, t, k, msgs[t])
    one.close()
    apart_layout = StoreLayout(tmp_path / "apart")
    apart = Meter(apart_layout, TENANT)
    for t in range(2):
        apart.run("task", task=t)
        m = [{"role": "user", "content": f"task {t}"}]
        for k in range(10):
            m = calls(apart, t, k, m)
    apart.close()
    a = cs.lever_breakpoint_add(cs.report(layout, TENANT, host, since="30d"))
    b = cs.lever_breakpoint_add(cs.report(apart_layout, TENANT, host, since="30d"))
    assert a.evidence == b.evidence and a.evidence["reused_tokens"] > 0


def test_the_check_scales_the_prediction_by_the_work_since_not_the_clock(layout):
    """A bench runs a week's work in an hour: the prediction in the check's window is the predicted share of what the
    window's calls would have cost under the old setting, not the weekly figure times the elapsed fraction of a week."""
    import time as _t
    from recant.metering import Meter
    host = cs.HostShape(agent_jobs=("agent",), carries_out=("breakpoint.add",))
    m = Meter(layout, TENANT)

    def task(t, cached):
        m.run("task", task=t)
        prev = 0
        for k in range(10):
            x = 8000 + 500 * k
            u = {"input": x - prev, "cache_read": prev, "output": 200} if cached else {"input": x, "output": 200}
            m.record(job="agent", model=MODEL, usage=u, session=f"{cached}-{t}", messages=[{"k": j} for j in range(k + 1)])
            prev = x
    for t in range(25):
        task(t, False)
    rep = cs.report(layout, TENANT, host, since="30d")
    lv = cs.lever_breakpoint_add(rep)
    _t.sleep(0.01)
    cs.apply(layout, TENANT, rep, lv, at=_t.time())
    _t.sleep(0.01)
    for t in range(25):
        task(t, True)
    m.close()
    rep = cs.report(layout, TENANT, host, since="30d")
    res = cs.check("breakpoint.add", rep, cs.overlay(layout, TENANT)["breakpoint.add"])
    assert res["realized_usd"] > 0
    assert res["predicted_usd"] == pytest.approx(lv.share * (res["realized_usd"] + sum(
        rep.cost(c) for c in rep.calls if c["start"] >= cs.overlay(layout, TENANT)["breakpoint.add"]["applied_at"])), rel=1e-3)
    assert 0.75 < res["realized_usd"] / res["predicted_usd"] < 1.25


def test_a_tool_call_repeated_with_the_same_arguments_or_after_its_own_error_is_a_loop(layout):
    """fix.tool_loop: an agent stuck calling open_task over and over. Priced by the model calls
    that issued the repeats, each by its share of the tool calls it made; a call made twice without an error is not one."""
    lg = Log(layout, "S1", "session", T0, session="s")
    for k in range(5):                                  # five model calls, each issuing the same open_task and one other
        t = T0 + timedelta(seconds=10 * k)
        lg.call(t, session="s", read=10000)
        lg.emit("tool_call", t + timedelta(seconds=1), session="s", name="open_task", input={"actor": "principal", "note": "look"})
        lg.emit("tool_call", t + timedelta(seconds=2), session="s", name="run_shell", input={"command": f"ls {k}"})
    t = T0 + timedelta(seconds=60)
    lg.call(t, session="s", read=10000)                 # an error answered by the same call: a loop at two
    lg.emit("tool_call", t + timedelta(seconds=1), session="s", name="web_fetch", input={"url": "u"}, error="Timeout")
    lg.call(t + timedelta(seconds=10), session="s", read=10000)
    lg.emit("tool_call", t + timedelta(seconds=11), session="s", name="web_fetch", input={"url": "u"})
    lg.call(t + timedelta(seconds=20), session="s", read=10000)   # the same command twice, no error: not a loop
    lg.emit("tool_call", t + timedelta(seconds=21), session="s", name="run_shell", input={"command": "git status"})
    lg.call(t + timedelta(seconds=30), session="s", read=10000)
    lg.emit("tool_call", t + timedelta(seconds=31), session="s", name="run_shell", input={"command": "git status"})
    lg.emit("turn", t + timedelta(seconds=40), session="s")
    lg.close(t + timedelta(seconds=41))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(hours=1)).isoformat())
    lv = cs.detect_tool_loop(rep)
    loops = {x["tool"]: x for x in lv.evidence["loops"]}
    assert set(loops) == {"open_task", "web_fetch"} and lv.kind == "fix_cause" and lv.state == "finding"
    assert loops["open_task"]["calls"] == 5 and not loops["open_task"]["after_error"]
    assert loops["web_fetch"]["after_error"] and lv.evidence["repeats"] == 4 + 1
    one = rep.cost(next(c for c in rep.calls))
    assert loops["open_task"]["usd"] == pytest.approx(4 * one / 2, abs=1e-4)  # four repeats, each half its model call
    assert loops["web_fetch"]["usd"] == pytest.approx(one, abs=1e-4)
    assert "open_task 5x" in lv.note


def test_a_break_or_a_capped_call_from_before_the_jobs_settings_changed_is_not_a_finding(layout):
    """fix.segment_changed and fix.output_cap read the job's window since its settings last changed. A break
    from before a fix (say, core rewrites that a later setting change stopped) is not a finding; one after the change is."""
    end = chat(layout, sessions=2, turns=3, gap=timedelta(minutes=2), core_change_at=(0, 1))
    lg = Log(layout, "N1", "extract_live", T0 + timedelta(minutes=1), day="2026-09-30")
    lg.call(T0 + timedelta(minutes=2), job="nominate", write=0, out=8000, stop="max_tokens")
    lg.close(T0 + timedelta(minutes=3))
    before = Log(layout, "H0", "deploy", T0 - timedelta(minutes=30))
    before.emit("setting_change", T0 - timedelta(minutes=30), jobs=[], what="an earlier deploy")
    before.close(T0 - timedelta(minutes=29))
    rep = cs.report(layout, TENANT, HOST, since=(T0 - timedelta(hours=1)).isoformat(), until=end.isoformat())
    assert cs.detect_segment_changes(rep) is not None and [lv.id for lv in cs.detect_output_cap(rep)] == ["fix.output_cap.nominate"]
    fix = Log(layout, "H1", "deploy", end - timedelta(seconds=2))
    fix.emit("setting_change", end - timedelta(seconds=2), jobs=["turn", "nominate"], what="the core is stable now; output cap raised")
    fix.close(end - timedelta(seconds=1))
    rep = cs.report(layout, TENANT, HOST, since=(T0 - timedelta(hours=1)).isoformat(), until=end.isoformat())
    assert cs.detect_segment_changes(rep) is None and cs.detect_output_cap(rep) == []


def test_the_effort_lever_is_priced_per_ten_percent_less_output_and_needs_evidence(layout):
    """output and effort settings — a your-call lever, never applied by itself, sized on the job's own price tag."""
    for s_ in range(4):                                  # output-heavy turns: 3,000 tokens written per call
        lg = Log(layout, f"E{s_}", "session", T0 + timedelta(hours=s_), session=f"e{s_}")
        for k in range(5):
            lg.call(T0 + timedelta(hours=s_, seconds=10 * k), session=f"e{s_}", read=20000 + 3000 * k, write=3000, out=3000)
        lg.emit("turn", T0 + timedelta(hours=s_, minutes=1), session=f"e{s_}")
        lg.close(T0 + timedelta(hours=s_, minutes=1, seconds=1))
    end = T0 + timedelta(hours=5)
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    (lv,) = cs.lever_effort(rep)
    tags, _ = cs.price_tags(rep.runs({"turn"}), rep.profiles, memory_known=False)
    assert lv.id == "effort.turn" and lv.kind == "your_call" and lv.state == "needs_evidence" and not lv.safe
    assert lv.share == pytest.approx(-tags["10% less output"]) and lv.saves_usd > 0
    assert not cs.eligible(rep, lv)[0]                                    # your call: never applied by the advisor


def test_the_runtime_guard_and_the_loop_finding_share_one_definition(layout):
    """ToolLoopGuard removes a tool after TOOL_LOOP_MIN identical failed calls; the same calls, logged, are the
    advisor's fix.tool_loop. A call with other arguments, or one that succeeded, does not count toward the guard."""
    from recant.toolloop import TOOL_LOOP_MIN, ToolLoopGuard
    g = ToolLoopGuard()
    bad = {"actor": "principal", "note": "x"}
    assert g.observe("open_task", bad, "ValueError") is None and g.observe("open_task", {"note": "y"}, "ValueError") is None
    assert g.observe("run_shell", {"command": "ls"}, None) is None and g.allowed("open_task")
    assert g.observe("open_task", dict(reversed(list(bad.items()))), "ValueError") is None    # the same arguments, any order
    assert "no longer offered" in g.observe("open_task", bad, "ValueError") and not g.allowed("open_task") and g.allowed("run_shell")
    lg = Log(layout, "S1", "session", T0, session="s")
    for k in range(TOOL_LOOP_MIN):
        lg.call(T0 + timedelta(seconds=10 * k), session="s", read=10000)
        lg.emit("tool_call", T0 + timedelta(seconds=10 * k + 1), session="s", name="open_task", input=bad, error="ValueError")
    lg.emit("turn", T0 + timedelta(seconds=40), session="s")
    lg.close(T0 + timedelta(seconds=41))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(hours=1)).isoformat())
    (loop,) = cs.detect_tool_loop(rep).evidence["loops"]
    assert loop["tool"] == "open_task" and loop["calls"] == TOOL_LOOP_MIN and loop["after_error"]


def test_the_reasoning_lever_is_not_offered_on_a_model_that_cannot_turn_reasoning_off(layout):
    """on a model whose reasoning cannot be turned off (effort is its control) only the effort lever is offered
    for those calls. The host may say so (`fixed_reasoning`); the library also knows it for claude-opus-5-5
    and claude-fable-5-1, which refuse every off form (probed 2026-11-03), so the host need not."""
    lg = Log(layout, "R1", "session", T0, session="r")
    for k in range(5):
        lg.call(T0 + timedelta(seconds=10 * k), session="r", read=20000, write=2000, out=3000)
    lg.emit("turn", T0 + timedelta(minutes=1), session="r")
    lg.close(T0 + timedelta(minutes=1, seconds=1))
    import json as _json
    for p in layout.log_dir(TENANT).glob("*.jsonl"):                     # give the calls reasoning tokens
        p.write_text("".join(_json.dumps({**r, "usage": {**r["usage"], "reasoning": 800}}) + "\n" if r.get("kind") == "model_call" else _json.dumps(r) + "\n"
                             for r in map(_json.loads, p.read_text().splitlines())))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(hours=1)).isoformat())
    assert MODEL == "claude-opus-5-5" and cs.lever_reasoning(rep) == [] and [lv.id for lv in cs.lever_effort(rep)] == ["effort.turn"]
    fixed = cs.HostShape(**{**vars(HOST), "fixed_reasoning": (MODEL,)})
    rep = cs.report(layout, TENANT, fixed, since=T0.isoformat(), until=(T0 + timedelta(hours=1)).isoformat())
    assert cs.lever_reasoning(rep) == [] and [lv.id for lv in cs.lever_effort(rep)] == ["effort.turn"]


def test_each_item_lands_in_its_section_with_its_label(layout):
    """A loop and a capped output are health, a cache break is a cost leak, a memory budget is quality;
    automatic for a safe lever, opt-in for a your-call one, pointed out for a cause to fix."""
    lv = lambda id_, kind, state="available": cs.Lever(id_, kind, "agent", 0.0, 1.0, 0.1, kind == "applied", state=state)   # noqa: E731
    cases = {"fix.tool_loop": ("fix_cause", "finding", "health", "pointed out"),
             "fix.output_cap.nominate": ("fix_cause", "finding", "health", "pointed out"),
             "fix.unmetered.live": ("fix_cause", "finding", "health", "pointed out"),
             "fix.egress.bypass.x": ("fix_cause", "finding", "health", "pointed out"),
             "fix.segment_changed": ("fix_cause", "finding", "cost", "pointed out"),
             "breakpoint.add": ("applied", "applied", "cost", "automatic"),
             "postreply.write": ("applied", "not_worth_building", "cost", "pointed out"),
             "effort.turn": ("your_call", "needs_evidence", "cost", "opt-in"),
             "advice.trim_tool_output": ("your_call", "host_advice", "cost", "pointed out"),
             "memory.core_budget": ("your_call", "needs_evidence", "quality", "opt-in")}
    for id_, (kind, state, sec, label) in cases.items():
        x = lv(id_, kind, state)
        assert (cs.section_of(x), cs.label_of(x)) == (sec, label), id_
    end = chat(layout, sessions=2)
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=end.isoformat())
    text_ = rep.text([lv("fix.tool_loop", "fix_cause", "finding")], [])
    health = text_.split("== health ==")[1]
    assert "model calls" in health and "pointed out  finding" in health
    assert "leaks found: none" in text_ and "levers: none sized in this window" in text_


def test_batching_is_offered_only_where_the_next_run_leaves_slack(layout, tmp_path):
    """A background run whose output the next agent run needs seconds later has no
    slack. Batching would only make it late, so it is pointed out as such and never applied, on any host. Where the
    host's own batch runs finish well inside the slack, it is offered, priced C·(β − P_late)."""
    from recant.store import StoreLayout
    for i in range(5):                                              # extraction right after each session, the next one at once
        t = T0 + timedelta(hours=i)
        lg = Log(layout, f"X{i}", "extract_live", t, day="2026-10-01")
        lg.call(t + timedelta(seconds=40), job="nominate", out=2000)
        lg.close(t + timedelta(seconds=60))
        _session_at(layout, f"S{i}", t + timedelta(seconds=65))
    rep = cs.report(layout, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(days=1)).isoformat())
    (b,) = cs.lever_batch(rep)
    assert b.state == "no_slack" and b.saves_usd < 0 and not b.safe and cs.label_of(b) == "pointed out"
    assert not cs.eligible(rep, b)[0] and 60 <= b.evidence["median_slack_s"] <= 66 and b.evidence["runs_late"] == 5
    other = StoreLayout(tmp_path / "slack")
    for i in range(4):                                              # the host's own batches took 2 minutes
        t = T0 + timedelta(days=i)
        lg = Log(other, f"B{i}", "extract_live", t, day=f"2026-10-0{i + 1}")
        lg.call(t + timedelta(minutes=1), job="nominate", out=2000, batch="b")
        lg.close(t + timedelta(minutes=2))
    for i in range(4, 8):                                           # direct runs, the next session ten minutes later
        t = T0 + timedelta(days=i)
        lg = Log(other, f"D{i}", "extract_live", t, day=f"2026-10-0{i + 1}")
        lg.call(t + timedelta(minutes=1), job="nominate", out=2000)
        lg.close(t + timedelta(minutes=2))
        _session_at(other, f"T{i}", t + timedelta(minutes=10))
    rep = cs.report(other, TENANT, HOST, since=T0.isoformat(), until=(T0 + timedelta(days=9)).isoformat())
    (b,) = cs.lever_batch(rep)
    assert b.state == "available" and b.safe and b.saves_usd > 0 and b.evidence["runs_late"] == 0 and b.evidence["batch_latencies"] == 4

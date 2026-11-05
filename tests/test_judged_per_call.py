"""A delivered lever is judged per affected call against the prediction per call, with the volume it
covers; under 50 affected calls it is "too little traffic to judge" and never reverted for it; a memory lever's "before"
is the week before it was applied, not this window; billing findings cover the window's days only, in words, a known
key's off the health line."""
import json

from recant import costshape as cs
from recant.store import StoreLayout
from recant_fixtures import TENANT


def test_judged_per_call_with_a_floor():
    entry = {"predicted_usd": 7.0, "assumed_calls": 700}                    # $0.01 a call predicted
    assert cs.judge({"realized_usd": 0.0, "calls": 12}, entry)["verdict"] == "too_little_traffic"
    j = cs.judge({"realized_usd": 0.55, "calls": 61}, entry)
    assert j["verdict"] == "met" and abs(j["ratio"] - 0.9016) < 1e-3
    assert cs.judge({"realized_usd": 0.1, "calls": 100}, entry)["verdict"] == "missed"
    assert cs.judge({"realized_usd": 0.1, "calls": 100}, {"predicted_usd": 7.0})["verdict"] == "no_prediction_per_call"


def test_the_line_names_the_volume_and_never_reads_realized_zero_as_a_failure():
    e = {"id": "ttl.prefix", "verdict": "too_little_traffic", "window_calls": 61, "assumed_calls": 1900, "realized_usd": 0.0,
         "predicted_window_usd": 2.41}
    line = cs.delivered_line(e)
    assert line == "ttl.prefix: too little traffic to judge (61 calls this window; the prediction assumed about 1,900 a week)"
    assert "$0" not in line and "against" not in line


def _reconcile(layout, rows):
    d = layout.costshape_dir(TENANT)
    d.mkdir(parents=True, exist_ok=True)
    (d / "reconcile.json").write_text(json.dumps({"rows": rows, "days_compared": 30, "keys": {}}))


def row(day, key, status, **kw):
    base = {"day": day, "key": key, "model": "claude-sonnet-5", "status": status, "calls_metered": kw.pop("calls", 0),
            "unmetered_tokens": 0, "unmetered_usd": 0.0, "metered_not_billed_tokens": 0, "metered_not_billed_usd": 0.0}
    return {**base, **kw}


def test_billing_findings_cover_the_window_in_words_and_a_known_key_is_off_the_health_line(tmp_path):
    layout = StoreLayout(tmp_path)
    layout.ensure(TENANT)
    # made-up rows: round figures on made-up days, one older than the week and two inside it
    _reconcile(layout, [row("2026-01-05", "live", "metered_not_billed", calls=1000, metered_not_billed_tokens=5_000_000,
                            metered_not_billed_usd=10.00),
                        row("2026-01-20", "live", "unmetered", unmetered_tokens=100_000, unmetered_usd=0.50, model="claude-opus-5-5"),
                        row("2026-01-21", "bench", "unmetered", unmetered_tokens=50_000, unmetered_usd=0.05)])
    week = cs.billing_findings(layout, TENANT, since_day="2026-01-19", known_keys=("bench",))
    ids = {lv.id: lv for lv in week}
    assert set(ids) == {"fix.unmetered.live", "fix.unmetered.bench"}             # 01-05 is not this week's
    assert ids["fix.unmetered.live"].note.startswith("billed on the live key but in no store this host meters: $0.50 on 01-20")
    assert cs.section_of(ids["fix.unmetered.bench"]) == "known" and cs.section_of(ids["fix.unmetered.live"]) == "health"
    allofit = {lv.id: lv for lv in cs.billing_findings(layout, TENANT)}
    assert "1000 calls, 5,000,000 tokens on 01-05" in allofit["fix.metered_not_billed.live"].note


def test_too_little_traffic_is_not_reverted(layout):
    """An applied lever with almost no traffic since: checked, judged too little, and left in place."""
    from recant.metering import Meter
    import time
    m = Meter(layout, TENANT)
    for i in range(3):
        m.run("agent")
        m.record(job="agent", model="claude-sonnet-5", usage={"input": 9000, "output": 10}, session=f"s{i}",
                 messages=[{"role": "user", "content": "x"}])
    m.close()
    now = time.time()
    host = cs.HostShape(agent_jobs=("agent",), agent_run_jobs=("agent",), carries_out=("breakpoint.add",))
    ov = {"breakpoint.add": {"kind": "applied", "action": {}, "applied_at": now - 8 * 86400, "predicted_usd": 50.0,
                             "predicted_share": 0.5, "job": "agent", "assumed_calls": 2000}}
    d = layout.costshape_dir(TENANT)
    d.mkdir(parents=True, exist_ok=True)
    (d / "overlay.json").write_text(json.dumps(ov))
    out = cs.nightly(layout, TENANT, host, now=now + 60)
    ev = [e for e in out["events"] if e.get("id") == "breakpoint.add"]
    assert ev[0]["verdict"] == "too_little_traffic" and ev[0]["window_calls"] == 3
    assert "breakpoint.add" in cs.overlay(layout, TENANT) and not any(e["event"] == "reverted" for e in out["events"])

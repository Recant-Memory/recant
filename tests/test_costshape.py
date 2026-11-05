"""costshape in the library — prices from the consumer's table, the dated list, or learned from
billed cost (tokens only when none applies); long-lifetime writes at their own rate; runs read from Recant's own
log records; cache lifetime from measured gaps; the whole shape as data."""
from datetime import date

import pytest

from recant import costshape as cs
from recant.provider import Prices

M = 1e-6


def call(run, seq, *, start, model="m", uncached=10, write=0, read=0, output=100, long_=0, cost=None, session="s", job="turn"):
    return dict(run=run, seq=seq, session=session, start=start, latency=1.0, model=model, job=job, uncached=uncached,
                cache_write=write, cache_write_long=long_, cache_read=read, output=output, reasoning=0, cost=cost)


def test_prices_are_learned_from_billed_cost_exactly():
    true = cs.Profile(p=3 * M, po=15 * M, w=1.25, r=0.1)
    calls = [call("r", i, start=i, uncached=100 + (37 * i * i) % 211, write=1000 * (i % 3), read=5000 + (811 * i) % 3001,
                  output=50 + (13 * i * i * i) % 97) for i in range(12)]
    for c in calls:
        c["cost"] = true.bill(c["uncached"], c["cache_write"], c["cache_read"], c["output"])
    pf, worst = cs.learn_prices(calls, "m")
    assert worst < 1e-9 and abs(pf.p - true.p) < 1e-12 and abs(pf.po - true.po) < 1e-12
    assert abs(pf.w - 1.25) < 1e-6 and abs(pf.r - 0.1) < 1e-6
    assert cs.learn_prices(calls[:5], "m") is None                       # too few to learn from


def test_the_price_source_is_the_consumers_table_then_the_list_then_billed_cost_then_tokens_only():
    calls = [call("a", 0, start=0, model="claude-opus-5-5"), call("b", 0, start=0, model="mine"),
             call("c", 0, start=0, model="theirs")]
    calls += [call("d", i, start=i, model="theirs", uncached=100 + i, output=10 + 3 * i, cost=(100 + i) * 1e-6 + (10 + 3 * i) * 5e-6)
              for i in range(8)]
    profiles, src, notes = cs.profiles_for(calls, {"mine": Prices(input=1.0, output=2.0, cache_read=0.1, cache_write=1.25,
                                                                 cache_write_long=2.0)}, today=date(2026, 10, 10))
    assert src["mine"] == "consumer's table" and profiles["mine"].w_long == 2.0 and profiles["mine"].r == pytest.approx(0.1)
    assert src["claude-opus-5-5"] == f"list prices as of {cs.LIST_PRICES_AS_OF}" and notes == []
    assert src["theirs"].startswith("learned from 8 billed calls")
    assert set(profiles) == {"mine", "claude-opus-5-5", "theirs"}
    _, src2, _ = cs.profiles_for([call("x", 0, start=0, model="nobody")])
    assert src2["nobody"] == "none: tokens only"
    with pytest.warns(UserWarning, match="days old"):
        _, _, notes = cs.profiles_for(calls[:1], today=date(2027, 3, 1))
    assert notes and "pass your own table" in notes[0]


def test_long_lifetime_writes_cost_their_own_rate():
    pf = cs.LIST_PRICES["claude-opus-5-5"]
    assert pf.bill(write=1000, write_long=600) == pytest.approx(4 * M * (1.25 * 400 + 2.0 * 600))
    r = cs.Run("r", "s", "claude-opus-5-5", "turn", calls=[call("r", 0, start=0, write=1000, long_=1000, output=0, uncached=0)])
    assert cs.run_shape(r, pf)["start_write"] == pytest.approx(4 * M * 2.0 * 1000)


def test_runs_are_read_from_recants_log_records_split_where_the_host_says():
    def mc(seq, ts, job="turn", **u):
        return {"kind": "model_call", "run": "L1", "seq": seq, "ts": ts, "model_id": "m", "job": job, "latency_ms": 2000,
                "usage": {"input": 5, "output": 7, "cache_read": 100, "cache_write": 50, **u}, "billed_usd": None, "session": "S"}
    recs = [mc(1, "2026-10-09T10:00:02Z"), {"kind": "served", "run": "L1", "seq": 2, "core_tokens": 900, "slice_tokens": 100,
                                            "measured": True},
            mc(3, "2026-10-09T10:00:05Z", cache_write_long=50), {"kind": "turn", "run": "L1", "seq": 4},
            mc(5, "2026-10-09T10:20:00Z"), {"kind": "turn", "run": "L1", "seq": 6},
            mc(7, "2026-10-09T10:21:00Z", job="other")]
    calls, meta = cs.calls_from_records(recs, split_on="turn", jobs={"turn"})
    assert [c["run"] for c in calls] == ["L1:0", "L1:0", "L1:1"] and [c["seq"] for c in calls] == [0, 1, 0]
    assert calls[1]["cache_write_long"] == 50 and calls[0]["start"] == pytest.approx(calls[0]["start"]) and calls[0]["session"] == "S"
    assert meta == {"L1:0": {"memory": 1000, "measured": True, "core": 900, "slice": 100}}
    whole, _ = cs.calls_from_records(recs)                                # no split: the log run is the run
    assert {c["run"] for c in whole} == {"L1"} and len(whole) == 4


def test_a_long_lifetime_pays_when_runs_in_a_session_come_back_between_five_and_sixty_minutes():
    pf = cs.LIST_PRICES["claude-opus-5-5"]
    calls = []
    for k, t0 in enumerate((0, 1200, 2400)):                               # three runs, 20 minutes apart
        calls += [call(f"r{k}", 0, start=t0, model="claude-opus-5-5", uncached=5, write=30000 if k else 30000, read=0),
                  call(f"r{k}", 1, start=t0 + 10, model="claude-opus-5-5", uncached=5, write=2000, read=30000)]
    runs = cs.group(calls)
    got = cs.cache_lifetime(runs, {"claude-opus-5-5": pf})
    assert got["runs"] == 2 and got["save"] > got["cost"] > 0
    s = cs.shape(calls, today=date(2026, 10, 10))
    assert s["gaps"] == {"n": 2, "under_5_min": 0, "5_to_60_min": 2, "over_60_min": 0}
    assert s["cache_lifetime"]["change_to_bill"] < 0 and s["prices"]["claude-opus-5-5"]["read_x"] == 0.05
    assert "price tags" in cs.render(s) and s["jobs"]["turn"]["runs"] == 3


def test_a_gap_is_measured_from_the_previous_request_start_not_its_reply():
    """the cache lifetime runs from the request. A 4-minute reply followed 2 minutes later by the next run is
    a 6-minute gap: the 5-minute entry has lapsed, though only 2 minutes passed since the reply ended."""
    a = call("a", 0, start=0)
    a["latency"] = 240.0
    b = call("b", 0, start=360)
    (gap, _, _), = cs.session_gaps(cs.group([a, b]))
    assert gap == 360
    assert cs.shape([a, b], today=date(2026, 10, 10))["gaps"]["5_to_60_min"] == 1


def test_a_fold_pays_only_past_its_break_even():
    pf = cs.LIST_PRICES["claude-opus-5-5"]
    pays, m = cs.fold_pays(pf, removed=80_000, rewritten=50_000, summary_cost=0.25, remaining=5)
    assert not pays and m > 5
    assert cs.fold_pays(pf, removed=80_000, rewritten=50_000, summary_cost=0.25, remaining=m + 1)[0]

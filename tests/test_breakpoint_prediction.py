"""breakpoint.add's prediction against the two measured runs it under-predicted by 23–24% (sync:
$1.406 realized against $1.148 predicted; async streamed: $1.3084 against $1.0533; CHANGELOG 0.2.0). The cause, read from the
per-call logs: a run's first call reads the system block other runs on the same setting cached (after the apply, ~2,356
of its 2,430 tokens; 71 written), and the predictor priced it as written in full. The pass line: each run re-predicted
from its own shadow pass lands within ±10% of what it realized (the line itself stays within a quarter)."""
import json
from pathlib import Path

import pytest

from recant import costshape as cs

DATA = json.loads((Path(__file__).parent / "data" / "breakpoint_acceptance.json").read_text())
# The runs were reported at the Sonnet 5.5 row as it stood before 0.3.2 (cache reads 0.10x); the vendor page now reads
# 0.05x. The realized figures are checked at the row they were reported at, the prediction at both.
REPORTED_AT = {"claude-sonnet-5-5": cs.Profile(p=2 * cs.M, po=10 * cs.M, w=1.25, r=0.10, w_long=2.0, batch=0.5)}


def _report(calls, table=None):
    table = table or cs.LIST_PRICES
    profiles = {m: table[m] for m in {c["model"] for c in calls}}
    calls = [dict(c, cost=None, batch=False, segments=None, error=None, stop_reason=None) for c in calls]
    return cs.Report(cs.HostShape(), min(c["start"] for c in calls), max(c["start"] for c in calls) + 1, calls, {}, [], [], [], {},
                     profiles, {})


def _predict_and_realize(run, *, drop_setting=False, table=None):
    table = table or REPORTED_AT
    calls = [dict(c, setting_fp=None) if drop_setting else c for c in run["calls"]]
    shadow = [c for c in calls if c["start"] < run["applied_at"]]
    lv = cs.lever_breakpoint_add(_report(shadow, table))
    res = cs.check("breakpoint.add", _report(calls, table), {"applied_at": run["applied_at"], "predicted_share": lv.share, "job": "agent"})
    return lv, res


@pytest.mark.parametrize("name", sorted(DATA["runs"]))
def test_each_acceptance_run_re_predicted_from_its_shadow_pass_lands_within_ten_percent(name):
    run = DATA["runs"][name]
    lv, res = _predict_and_realize(run)
    assert res["runs"] == run["reported"]["runs"]
    assert res["realized_usd"] == pytest.approx(run["reported"]["realized_usd"], abs=0.001)      # the same calls as the check
    assert 0.90 <= res["realized_usd"] / res["predicted_usd"] <= 1.10, res
    assert lv.evidence["first_calls_warm"] >= 24                                                # of 30 shadow runs
    lv2, res2 = _predict_and_realize(run, table=cs.LIST_PRICES)                                 # at today's list row
    assert 0.90 <= res2["realized_usd"] / res2["predicted_usd"] <= 1.10, res2


@pytest.mark.parametrize("name", sorted(DATA["runs"]))
def test_without_the_shared_system_block_the_old_bias_comes_back(name):
    """No setting fingerprint, no shared block: the first call is priced as written in full, as before the fix, and
    the realized saving runs 20% or more over the prediction, as both checks found."""
    _, res = _predict_and_realize(DATA["runs"][name], drop_setting=True)
    assert res["realized_usd"] / res["predicted_usd"] >= 1.20, res


def test_runs_started_together_find_the_block_cold():
    """The async run's first five tasks start at once: none of them can read a block none has written yet."""
    run = DATA["runs"]["async-streamed"]
    shadow = [c for c in run["calls"] if c["start"] < run["applied_at"]]
    firsts = sorted((c for c in shadow if c["seq"] == 0), key=lambda c: c["start"])
    head = cs._shared_head(shadow, firsts, cs.TTL_SECONDS["5m"])
    assert [id(c) in head for c in firsts[:5]] == [False] * 5 and all(id(c) in head for c in firsts[6:])

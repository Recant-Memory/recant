"""The cost check's arithmetic: a task run from its model calls, each arm summarized with seeded
intervals, the gate judged on intervals (met / missed / unclear), the flat export's fields."""
import json

from recant.harness.costcheck import EXPORT_FIELDS, TaskRun, export_row, gate, summarize, task_run, write_outputs
from recant.machinery.loader import CostCheck
from recant.provider import Prices

G = CostCheck(reps=3, level=0.95, resamples=500, seed=0, gate={"cost_cut_min": 0.15, "cache_hit_drop_max_pp": 1.0})


def call(model="m", input=100, read=900, write=0, output=50, cost=0.01):
    return {"kind": "model_call", "model_id": model, "usage": {"input": input, "cache_read": read, "cache_write": write, "output": output},
            "cost_usd": cost}


def run(arm, rep, *, calls, cost_each, ok=True, read=900, task="t1"):
    return task_run([call(cost=cost_each, read=read) for _ in range(calls)], task=task, arm=arm, rep=rep, run_id=f"{arm}{rep}{task}",
                    started="2026-10-07T10:00:00Z", turns=2, duration_ms=1000, outcome=ok)


def test_a_task_run_sums_its_calls_and_splits_cost_when_priced():
    r = task_run([call(), call(input=10, read=0, write=500, output=5, cost=None)], task="t", arm="on", rep=1, run_id="r",
                 started="x", turns=1, duration_ms=5, outcome=True)
    assert (r.calls, r.input, r.cache_read, r.cache_write, r.output) == (2, 110, 900, 500, 55)
    assert r.cost_usd is None and r.cost_split == {}                     # one unpriced call: no dollars claimed
    priced = task_run([call(), call(input=10, read=0, write=500, output=5)], task="t", arm="on", rep=1, run_id="r", started="x",
                      turns=1, duration_ms=5, outcome=True, prices={"m": Prices(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)})
    assert priced.cost_split == {"input": round(110 * 3 / 1e6, 6), "cache_read": round(900 * 0.3 / 1e6, 6),
                                 "cache_write": round(500 * 3.75 / 1e6, 6), "output": round(55 * 15 / 1e6, 6)}
    assert priced.cost_usd == round(sum(priced.cost_split.values()), 6)


def test_each_arm_is_summarized_with_seeded_intervals():
    runs = [run("on", i, calls=4, cost_each=0.01, ok=i != 2) for i in range(1, 4)] + [run("off", i, calls=8, cost_each=0.01) for i in range(1, 4)]
    s = summarize(runs, resamples=500)
    assert s["on"]["n"] == 3 and s["on"]["successes"] == 2 and abs(s["on"]["success_rate"] - 2 / 3) < 1e-6
    assert s["on"]["calls_per_task"] == 4 and s["on"]["cost_per_success"] == 0.06 and s["off"]["cost_per_success"] == 0.08
    assert s["on"]["cache_hit_ratio"] == 0.9 and s["on"]["tokens"]["cache_read"] == 3 * 4 * 900
    assert s == summarize(runs, resamples=500)                           # seeded: the same runs, the same report
    lo, hi = s["on"]["intervals"]["success_rate"]
    assert lo <= 2 / 3 <= hi


def test_the_gate_passes_fails_or_asks_for_more_by_the_interval():
    cheap = [run("on", i, calls=4, cost_each=0.01) for i in range(12)] + [run("off", i, calls=8, cost_each=0.01) for i in range(12)]
    v = gate(cheap, G)
    assert v["verdict"] == "pass" and v["criteria"]["cost_per_success_cut"]["point"] == 0.5
    assert v["criteria"]["calls_per_task_diff"]["verdict"] == "met" and v["criteria"]["cache_hit_diff_pp"]["verdict"] == "met"
    dear = [run("on", i, calls=10, cost_each=0.01) for i in range(12)] + [run("off", i, calls=8, cost_each=0.01) for i in range(12)]
    assert gate(dear, G)["verdict"] == "fail" and gate(dear, G)["criteria"]["calls_per_task_diff"]["verdict"] == "missed"
    # a cache that breaks with memory on fails the gate even when calls fall
    broken = [run("on", i, calls=4, cost_each=0.01, read=100) for i in range(12)] + [run("off", i, calls=8, cost_each=0.01) for i in range(12)]
    assert gate(broken, G)["criteria"]["cache_hit_diff_pp"]["verdict"] == "missed"
    # three noisy runs a side: the point passes, the interval straddles the bar, so: run more
    noisy = [run("on", 1, calls=2, cost_each=0.01), run("on", 2, calls=12, cost_each=0.01), run("on", 3, calls=3, cost_each=0.01),
             run("off", 1, calls=8, cost_each=0.01), run("off", 2, calls=9, cost_each=0.01), run("off", 3, calls=7, cost_each=0.01)]
    nv = gate(noisy, G)
    assert nv["criteria"]["cost_per_success_cut"]["point"] > 0.15 and nv["verdict"] == "run more"


def test_the_export_has_the_flat_fields_and_the_report_is_written(tmp_path):
    runs = [run("on", 1, calls=3, cost_each=0.01), run("off", 1, calls=5, cost_each=0.01, ok=False)]
    row = export_row(runs[1])
    assert tuple(row) == EXPORT_FIELDS and row["turns"] == 5 and row["outcome"] == "failed" and row["tokens_in"] == 500
    paths = write_outputs(tmp_path / "out", runs, G, title="Placeholder", notes=["a note"])
    lines = [json.loads(l) for l in paths["runs"].read_text().splitlines()]
    assert [l["arm"] for l in lines] == ["on", "off"]
    report = paths["report"].read_text()
    assert report.startswith("# Placeholder") and "Gate: **" in report and "- a note" in report
    assert json.loads(paths["summary"].read_text())["gate"]["n_per_arm"] == {"on": 1, "off": 1}
    assert isinstance(runs[0], TaskRun)


def test_the_write_path_counts_in_the_cost_and_the_figure_without_it_is_shown_beside():
    from dataclasses import replace
    base = [run("on", i, calls=4, cost_each=0.01) for i in range(12)] + [run("off", i, calls=8, cost_each=0.01) for i in range(12)]
    priced = [replace(r, write_path_usd=0.04) if r.arm == "on" else r for r in base]     # extraction doubles the on arm's cost
    s = summarize(priced, resamples=200)
    assert s["on"]["cost_per_success"] == 0.08 and s["on"]["cost_per_success_without_extraction"] == 0.04
    assert s["on"]["write_path_usd"] == 0.48 and s["off"]["write_path_usd"] == 0
    v = gate(priced, G)
    assert v["criteria"]["cost_per_success_cut"]["point"] == 0.0 and v["verdict"] == "fail"           # gated with extraction
    assert v["beside"]["cost_per_success_cut_without_extraction"]["point"] == 0.5                    # shown without it
    assert export_row(priced[0])["cost_usd"] == round(priced[0].cost_usd + 0.04, 6)

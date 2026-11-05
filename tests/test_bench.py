"""`recant bench` offline, on a simulated Messages API that caches by exact prefix (so a salted prefix can only be
read by the arm that wrote it). An off arm sends the host's request plus the salt; the safe arm carries its lever out with the
prediction frozen first; the arms never share a cache or a store; a failed task stays in its arm; a changed predictions
file fails the report; a resumed bench reruns nothing and refuses another Recant version; async agents run."""
import asyncio
import hashlib
import json
import textwrap
from types import SimpleNamespace

import pytest

from recant import bench


def toks(x) -> int:
    return len(json.dumps(x, sort_keys=True)) // 4


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def marked(x) -> bool:
    return "cache_control" in json.dumps(x)


class SimAPI:
    """Bills as the API would, roughly: a breakpoint writes the prefix up to it; a later request reads the longest
    written prefix it starts with. Records every request body."""

    def __init__(self):
        self.messages, self.cache, self.bodies = self, {}, []

    def _bill(self, kw):
        self.bodies.append(json.loads(json.dumps(kw)))
        sys_, msgs = kw.get("system"), kw.get("messages") or []
        total = toks(strip(sys_)) + toks(strip(msgs))
        prefixes = [("s", json.dumps(strip(sys_), sort_keys=True), toks(strip(sys_)))] + \
            [(f"m{k}", json.dumps([strip(sys_), strip(msgs[:k])], sort_keys=True), toks(strip(sys_)) + toks(strip(msgs[:k])))
             for k in range(1, len(msgs) + 1)]
        hit = max([t for _, key, t in prefixes if key in self.cache] or [0])
        writes = []
        if marked(sys_):
            writes.append(prefixes[0])
        if msgs and marked(msgs[-1]):
            writes.append(prefixes[-1])
        written = max([t for _, key, t in writes if key not in self.cache] or [0])
        for _, key, t in writes:
            self.cache[key] = t
        cw = max(0, written - hit)
        usage = SimpleNamespace(input_tokens=total - hit - cw, output_tokens=60, cache_read_input_tokens=hit,
                                cache_creation_input_tokens=cw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Send a photo.")], usage=usage, stop_reason="end_turn",
                               model=kw["model"])

    def create(self, **kw):
        return self._bill(kw)

    def count_tokens(self, **kw):
        return SimpleNamespace(input_tokens=toks(kw))


class AsyncSimAPI(SimAPI):
    def __init__(self):
        super().__init__()
        self.messages = SimpleNamespace(create=self.acreate, count_tokens=self.count_tokens)

    async def acreate(self, **kw):
        await asyncio.sleep(0)
        return self._bill(kw)


AGENT = textwrap.dedent('''
    HANDBOOK = "You are the support agent. " + "Policy line. " * 400
    def run_task(task, client):
        msgs = []
        for q in ("What now?", "Can I return it?", "How long?", "Express?"):
            if task.get("explode") and client.messages._w.client.salt.endswith("arm B]"):
                raise RuntimeError("the agent fell over")
            msgs.append({"role": "user", "content": f"order {task['id']}: {q}"})
            r = client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK, messages=msgs)
            msgs.append({"role": "assistant", "content": r.content[0].text})
        return len(msgs)
    async def run_task_async(task, client):
        msgs = []
        for q in ("What now?", "Can I return it?"):
            msgs.append({"role": "user", "content": f"order {task['id']}: {q}"})
            r = await client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK, messages=msgs)
            msgs.append({"role": "assistant", "content": r.content[0].text})
        return len(msgs)
    def run_task_two_models(task, client):
        r = client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK, messages=[{"role": "user", "content": "a"}])
        client.messages.create(model="house-model-1", max_tokens=64, messages=[{"role": "user", "content": r.content[0].text}])
        return 1
    def grade(task, result):
        return task["id"] % 5 != 0
''')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / "bench_agent.py").write_text(AGENT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    def tasks(n, explode=()):
        p = tmp_path / "tasks.jsonl"
        p.write_text("".join(json.dumps({"id": i, **({"explode": True} if i in explode else {})}) + "\n" for i in range(n)))
        return str(p)
    return tmp_path, tasks


def go(tasks_path, out, api, **kw):
    lines = []
    rep = bench.run(tasks_path=tasks_path, run_spec=kw.pop("run_spec", "bench_agent:run_task"), grade_spec="bench_agent:grade",
                    out=str(out), yes=True, sdk_factory=lambda aio: api, say=lines.append, **kw)
    return rep, lines


def test_an_off_arm_sends_the_hosts_request_plus_the_salt_and_nothing_else(setup):
    """an off/off run says nothing about cache isolation (neither arm writes a cache); the A/A check is the
    on/on run in test_bench_hosts. This keeps what an off arm sends: the host's request, the salt line, no markers."""
    tmp, tasks = setup
    api = SimAPI()
    rep, lines = go(tasks(12), tmp / "aa", api, calibrate=0, arm_b="off")
    assert rep["header"]["arms"] == {"A": "off", "B": "off"}
    assert not any(marked(b) for b in api.bodies)
    assert {b["system"].split("\n", 1)[0] for b in api.bodies} == set(rep["header"]["salts"].values())
    assert len({len(s) for s in rep["header"]["salts"].values()}) == 1                      # the same length in every arm


def test_the_safe_arm_carries_its_lever_out_against_a_prediction_frozen_first(setup):
    tmp, tasks = setup
    api = SimAPI()
    rep, lines = go(tasks(16), tmp / "safe", api, calibrate=6)
    pred = json.loads((tmp / "safe" / "predictions.json").read_text())
    assert [lv["id"] for lv in pred["arms"]["B"]["levers"]] == ["breakpoint.add"] and pred["arms"]["A"]["levers"] == []
    assert rep["header"]["predictions_sha256"] == hashlib.sha256((tmp / "safe" / "predictions.json").read_bytes()).hexdigest()
    assert any(rep["header"]["predictions_sha256"] in x for x in lines)                      # printed before the arms ran
    assert rep["cost"]["per_task"]["B"] < rep["cost"]["per_task"]["A"] * 0.7
    assert rep["equivalence"]["B"]["equivalent"] == 10 * 4 and rep["equivalence"]["B"]["not_equivalent"] == 0
    assert rep["quality"]["held"] and rep["tokens"]["B"]["per_task"]["cache_read"] > 0 == rep["tokens"]["A"]["per_task"]["cache_read"]
    p = next(p for p in rep["predictions"] if p["id"] == "breakpoint.add")
    assert p["ratio"] is not None and any(x.startswith("prediction ") for x in rep["verdict"])
    assert rep["estimate"]["tasks_run"] == 6 + 2 * 10 and rep["cost"]["total_spend_usd"] > 0
    # isolation: arm B's first call reads nothing (cold), and no arm ever read another's salted prefix
    b_bodies = [b for b in api.bodies if (b["system"][0]["text"] if isinstance(b["system"], list) else b["system"]).startswith(rep["header"]["salts"]["B"])]
    assert b_bodies and not marked(b_bodies[0]["messages"][:-1])
    salts = rep["header"]["salts"]
    for key in api.cache:
        assert sum(s in key for s in salts.values()) == 1
    assert (tmp / "safe" / "report.md").read_text().startswith("# recant bench")


def test_a_failed_task_stays_in_its_arm_as_a_failure(setup):
    tmp, tasks = setup
    rep, _ = go(tasks(10, explode={5, 7}), tmp / "fail", SimAPI(), calibrate=2)
    f = rep["failures"]
    assert f["B"]["failed_tasks"] == 2 and f["A"]["failed_tasks"] == 0 and f["B"]["errors"] == {"RuntimeError": 2}
    assert rep["header"]["tasks_paired"] == 8                                                # none dropped
    assert any("failures differ" in x for x in rep["verdict"])
    assert rep["quality"]["pass_rate"]["B"] < rep["quality"]["pass_rate"]["A"]


def test_a_changed_predictions_file_fails_the_report(setup):
    tmp, tasks = setup
    go(tasks(8), tmp / "tamper", SimAPI(), calibrate=2)
    p = tmp / "tamper" / "predictions.json"
    p.write_text(p.read_text().replace("breakpoint.add", "breakpoint.add "))
    with pytest.raises(SystemExit, match="changed after it was written"):
        bench.build_report(tmp / "tamper")


def test_a_resumed_bench_reruns_nothing_and_refuses_another_version(setup):
    tmp, tasks = setup
    t = tasks(8)
    api = SimAPI()
    go(t, tmp / "res", api, calibrate=2)
    n = len(api.bodies)
    go(t, tmp / "res", api, calibrate=2)
    assert len(api.bodies) == n
    m = tmp / "res" / "manifest.json"
    man = json.loads(m.read_text())
    man["recant"]["commit"] = "0" * 40
    m.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="not compared across versions"):
        go(t, tmp / "res", api, calibrate=2)
    rep, _ = go(t, tmp / "res", api, calibrate=2, allow_mixed=True)
    assert rep["header"]["mixed_versions"]


def test_an_async_agent_runs_and_the_estimate_comes_first(setup):
    tmp, tasks = setup
    rep, lines = go(tasks(8), tmp / "aio", AsyncSimAPI(), calibrate=2, run_spec="bench_agent:run_task_async")
    assert rep["header"]["tasks_paired"] == 6 and rep["cost"]["per_task"]["B"] < rep["cost"]["per_task"]["A"]
    assert lines[0].startswith("recant bench") and "estimated spend" in lines[0]
    api = SimAPI()
    out = bench.run(tasks_path=tasks(8), run_spec="bench_agent:run_task", out=str(tmp / "est"), estimate_only=True, calibrate=2,
                    sdk_factory=lambda aio: api, say=lambda *_: None)
    e = out["estimate"]
    assert e["tasks_run"] == 2 + 2 * 6 and e["estimated_usd"] == pytest.approx(e["first_task_usd"] * 14) and e["first_task_usd"] > 0
    assert len(api.bodies) == 4 and not (tmp / "est" / "arm-B").exists()                     # one task run, then stopped


def test_the_estimate_names_the_models_it_has_no_price_for_before_anything_is_spent(setup):
    tmp, tasks = setup
    lines = []
    out = bench.run(tasks_path=tasks(8), run_spec="bench_agent:run_task_two_models", out=str(tmp / "unp"), estimate_only=True,
                    calibrate=2, sdk_factory=lambda aio: SimAPI(), say=lines.append)
    assert out["estimate"]["unpriced_models"] == ["house-model-1"] and out["estimate"]["unpriced_calls"] == 1
    warn, = [x for x in lines if x.startswith("warning:")]
    assert "1 of the first task's 2 calls" in warn and "house-model-1" in warn and "--prices" in warn
    lines.clear()
    bench.run(tasks_path=tasks(8), run_spec="bench_agent:run_task", out=str(tmp / "pri"), estimate_only=True, calibrate=2,
              sdk_factory=lambda aio: SimAPI(), say=lines.append)
    assert not any(x.startswith("warning:") for x in lines)                                 # every model priced: no warning


def test_modes_that_do_not_exist_are_refused(setup):
    tmp, tasks = setup
    with pytest.raises(SystemExit, match="does not carry managed notes"):
        go(tasks(4), tmp / "n", SimAPI(), calibrate=1, arm_b="notes")
    with pytest.raises(SystemExit, match="an arm is off, safe"):
        go(tasks(4), tmp / "x", SimAPI(), calibrate=1, arm_b="fast")
    with pytest.raises(SystemExit, match="not a terminal"):
        bench.run(tasks_path=tasks(4), run_spec="bench_agent:run_task", out=str(tmp / "q"), calibrate=1,
                  sdk_factory=lambda aio: SimAPI(), say=lambda *_: None)


def test_the_bench_says_how_far_it_has_got_after_the_estimate(setup):
    import re
    tmp, tasks = setup
    rep, lines = go(tasks(8), tmp / "prog", SimAPI(), calibrate=2)
    prog = [x for x in lines if re.match(r"^\[\d+/\d+\] task \d+ arm [ABC]: (passed|failed|done|\w+Error)", x)]
    assert len(prog) == 1 + 2 * 6 and prog[0].startswith("[2/14]") and prog[-1].startswith("[14/14]") and "min left" in prog[-1]   # task 0 priced the run
    assert lines.index(prog[0]) > next(i for i, x in enumerate(lines) if "estimated spend" in x)      # after the estimate


def test_a_report_rebuilt_at_other_prices_restates_both_sides_of_each_prediction(setup):
    """What is frozen is each lever's share; the dollars are at the report's prices, cost per task's base included. At ten
    times every price the ratios are unchanged (in 0.3.1 cost per task kept its dollars from the run's table)."""
    from recant.provider import Prices
    tmp, tasks = setup
    go(tasks(10), tmp / "rp", SimAPI(), calibrate=3)
    a = bench.build_report(tmp / "rp")
    b = bench.build_report(tmp / "rp", prices={"claude-sonnet-5-5": Prices(input=20, output=100, cache_read=1, cache_write=25, cache_write_long=40)})
    ka = {(p["arm"], p["id"]): p for p in a["predictions"]}
    kb = {(p["arm"], p["id"]): p for p in b["predictions"]}
    for k in ka:
        if ka[k].get("ratio") is not None:
            assert kb[k]["ratio"] == pytest.approx(ka[k]["ratio"], rel=5e-3), k   # check() rounds dollars to 4 places
    cpt = kb[("A", "cost per task")]
    assert cpt["predicted_usd"] == pytest.approx(10 * ka[("A", "cost per task")]["predicted_usd"], rel=1e-6)

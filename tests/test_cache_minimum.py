"""`breakpoint.add` is sized only on prefixes the API will cache: a shorter one is sent with its marker and silently
not cached (no write, no read), so a saving sized on it would never come (an outside developer found a Haiku 5.5
writer's 457-token prompts uncached that the report had counted)."""
import json

from recant import costshape as cs, wrap
from test_stranger import FakeSDK, TENANT


def _report(layout, system: str, model: str = "claude-haiku-5-5"):
    client = wrap(FakeSDK(), store=layout, tenant=TENANT, job="agent")
    for i in range(25):
        with client.run("agent", session=f"s{i}"):
            msgs = []
            for k in range(3):
                msgs.append({"role": "user", "content": f"turn {k} of run {i}"})
                client.messages.create(model=model, max_tokens=50, system=system, messages=list(msgs))
                msgs.append({"role": "assistant", "content": "ok"})
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    shape = cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()})
    return cs.report(layout, TENANT, shape, since="30d")


def test_the_minimum_table():
    assert cs.cache_min("claude-haiku-5-5") == 512 and cs.cache_min("claude-sonnet-5") == 1024
    assert cs.cache_min("claude-haiku-4-5-20251001") == 4096 and cs.cache_min("some-new-model") == 4096


def test_no_saving_is_sized_on_prompts_under_the_minimum(layout):
    rep = _report(layout, "You are terse. " * 20)                          # about 100 tokens: under Haiku 5.5's 512
    assert rep.calls and all(c["uncached"] < 512 for c in rep.calls)
    lv = cs.lever_breakpoint_add(rep)
    assert lv is not None and lv.evidence["reused_tokens"] == 0 and lv.evidence["write_premium"] == 0
    assert not cs.worth_listing(lv)


def test_the_same_traffic_over_the_minimum_is_sized(layout):
    rep = _report(layout, "You are terse. " * 200)                         # about 750 tokens: over it
    lv = cs.lever_breakpoint_add(rep)
    assert lv is not None and lv.evidence["reused_tokens"] > 0 and lv.saves_usd > 0


def test_a_saving_shared_by_two_agents_is_split_by_job(layout):
    """Two agents under one lever: the line names each one's part (one under the minimum gets none)."""
    client = wrap(FakeSDK(), store=layout, tenant=TENANT, job="agent")
    for i in range(25):
        for job, system in (("researcher", "You are terse. " * 200), ("writer", "Write it. ")):
            with client.run(job, session=f"s{i}"):
                msgs = []
                for k in range(3):
                    msgs.append({"role": "user", "content": f"turn {k} of run {i}"})
                    client.messages.create(model="claude-haiku-5-5", max_tokens=50, system=system, messages=list(msgs))
                    msgs.append({"role": "assistant", "content": "ok"})
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                    since="30d")
    lv = cs.lever_breakpoint_add(rep)
    split = lv.evidence["by_job"]
    assert set(split) == {"researcher", "writer"} and split["researcher"] > 0 and split["writer"] == 0
    assert "researcher $" in lv.line() and "writer $0.00/wk" in lv.line()


def test_a_job_the_breakpoint_would_cost_is_left_out_and_not_marked(layout):
    """A writer whose prompts the cache would not take (under the model's minimum) gains nothing from a marker: the
    lever is applied to the researcher only, and the writer's calls go out with no marker."""
    sdk = FakeSDK()
    client = wrap(sdk, store=layout, tenant=TENANT, job="agent")

    def traffic(runs):
        for i in range(runs):
            with client.run("researcher", session=f"s{i}"):
                msgs = []
                for k in range(3):
                    msgs.append({"role": "user", "content": f"turn {k} of run {i}"})
                    client.messages.create(model="claude-haiku-5-5", max_tokens=50, system="You are terse. " * 200,
                                           messages=list(msgs))
                    msgs.append({"role": "assistant", "content": "ok"})
            with client.run("writer", session=f"s{i}"):
                client.messages.create(model="claude-haiku-5-5", max_tokens=50, system="Write it.",
                                       messages=[{"role": "user", "content": f"notes for task {i}"}])
    traffic(25)
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                    since="30d")
    lv = cs.lever_breakpoint_add(rep)
    assert lv.evidence["by_job"]["writer"] <= 0 < lv.evidence["by_job"]["researcher"]
    assert lv.action["jobs"] == ["researcher"]
    assert abs(lv.saves_usd - lv.evidence["by_job"]["researcher"]) < 1e-3   # sized on the job it is applied to
    cs.apply(layout, TENANT, rep, lv)
    sdk.sent.clear()
    traffic(1)
    marked = ["cache_control" in json.dumps(k) for k in sdk.sent]
    assert marked == [True, True, True, False]                              # three researcher calls, then the writer's


def _per_job_store(layout, runs=25):
    sdk = FakeSDK()
    client = wrap(sdk, store=layout, tenant=TENANT, job="agent")
    for i in range(runs):
        with client.run("researcher", session=f"s{i}"):
            msgs = []
            for k in range(3):
                msgs.append({"role": "user", "content": f"turn {k} of run {i}"})
                client.messages.create(model="claude-haiku-5-5", max_tokens=50, system="You are terse. " * 200, messages=list(msgs))
                msgs.append({"role": "assistant", "content": "ok"})
        with client.run("writer", session=f"s{i}"):
            client.messages.create(model="claude-haiku-5-5", max_tokens=50, system="Write it.",       # one setting, and
                                   messages=[{"role": "user", "content": f"notes for task {i}"}])        # under the minimum
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    return client, cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                             since="30d")


def test_a_lever_applied_to_some_jobs_is_readied_predicted_and_judged_on_those_jobs(layout):
    client, rep = _per_job_store(layout)
    lv = cs.lever_breakpoint_add(rep)
    assert lv.action["jobs"] == ["researcher"]
    assert abs(lv.share - lv.saves_usd / rep.per_week / rep.job_bill({"researcher"})) < 1e-9   # of the researcher's bill
    assert cs.evidence(rep, cs.lever_jobs(rep, lv.job, lv.action)) == 25                  # the writer's runs not counted
    entry = cs.apply(layout, TENANT, rep, lv)
    ov = cs.overlay(layout, TENANT)["breakpoint.add"]
    assert ov["assumed_calls"] == round(75 * rep.per_week)                                 # the researcher's 75 calls only
    assert "of job researcher's bill" in lv.line()


def test_a_lever_is_not_ready_on_the_runs_of_a_job_it_is_not_applied_to(layout):
    _, rep = _per_job_store(layout, runs=8)
    lv = cs.lever_breakpoint_add(rep)
    lv.action = dict(lv.action, jobs=["researcher"])
    rep.calls += [dict(c, job="writer", run=f"w{i}") for i, c in enumerate(rep.calls[:30])]   # writer traffic: no help
    ok, why = cs.eligible(rep, lv)
    assert not ok and "8 runs" in why


def test_apply_names_the_jobs_it_covers_and_the_one_it_leaves_off(layout, capsys):
    from recant import cli
    _per_job_store(layout)
    assert cli.main(["apply", "--dry-run", "--store", str(layout.root), "--tenant", TENANT]) == 0
    out = capsys.readouterr().out
    assert "would apply breakpoint.add on job researcher" in out and "Left off writer" in out

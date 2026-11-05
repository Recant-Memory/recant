"""A multi-agent host names each agent's runs: `run("researcher")`, `run("writer")`. The calls inside a run are that
job's, in the log, in `recant cost` and in the levers sized per job; both are the host's agent work. (Before 0.3.2 the
run's name only named the log run, and every call was the wrapper's job: a two-agent host was reported as one job
"agent", with its reasoning and effort levers priced over both agents mixed.)"""
import json

from recant import costshape as cs, wrap
from test_wrap_paths import API, TENANT, records, sdk


def two_agents(client, n=3):
    for i in range(n):
        with client.run("researcher", session=f"t{i}"):
            client.messages.create(model="claude-sonnet-5-5", max_tokens=64, thinking={"type": "adaptive"},
                                   messages=[{"role": "user", "content": f"find {i}"}])
        with client.run("writer", session=f"t{i}"):
            client.messages.create(model="claude-haiku-5-5", max_tokens=64, messages=[{"role": "user", "content": f"draft {i}"}])


def test_each_run_s_calls_are_its_job_and_both_are_agent_work(layout):
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    two_agents(client)
    jobs = sorted({r["job"] for r in records(layout)})
    assert jobs == ["researcher", "writer"]
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    assert {"researcher", "writer"} <= set(host["agent_jobs"]) and {"researcher", "writer"} <= set(host["agent_run_jobs"])
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: tuple(v) if isinstance(v, list) else v for k, v in host.items()}),
                    since=0)
    text = rep.text(cs.advise(rep), [])
    assert "job researcher" in text and "job writer" in text and "job agent" not in text


def test_an_opted_in_lever_reaches_only_its_own_job(layout):
    api = API()
    client = wrap(sdk(api), store=layout, tenant=TENANT)
    cs.opt_in(layout, TENANT, "effort.writer", action={"effort": "low"}, evidence="test")
    two_agents(client, n=1)
    researcher, writer = api.bodies
    assert "output_config" not in researcher and writer["output_config"] == {"effort": "low"}


def test_a_call_with_no_run_or_an_unnamed_run_is_the_wrapper_s_job(layout):
    client = wrap(sdk(API()), store=layout, tenant=TENANT, job="support")
    with client.run(session="s"):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": "a"}])
    client.messages.create(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": "b"}])
    client.close()
    assert {r["job"] for r in records(layout)} == {"support"}

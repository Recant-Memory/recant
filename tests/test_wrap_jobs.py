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


def test_the_report_names_the_jobs_a_share_is_of_and_says_its_weekly_scale(layout):
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    two_agents(client, n=4)
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: tuple(v) if isinstance(v, list) else v for k, v in host.items()}), since="7d")
    levers = cs.advise(rep)
    text = rep.text(levers, [])
    head = text.splitlines()[0]
    assert "UTC (7.0 days)" in head and "; traffic in it from " in head      # minutes of traffic, said as such
    assert "what each lever would have saved on the calls it is priced on, as measured" in text
    assert "/wk" not in text and "of job researcher's bill)" in text   # under a day: dollars measured, no weekly rate
    assert "of agent)" not in text
    rep.calls[0]["start"] -= 2 * 86400                                 # the same traffic spread over two days
    text = rep.text(levers, [])
    assert "over this whole 7.0-day window that is ×1.00" in text and "/wk" in text


def test_a_relative_store_is_where_it_was_when_wrapped(tmp_path, monkeypatch):
    """The quickstart's store="store" is fixed when wrap is called: a host that changes directory later still writes
    there."""
    from recant.store import StoreLayout
    monkeypatch.chdir(tmp_path)
    client = wrap(sdk(API()), store="store", tenant=TENANT)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    with client.run("agent", session="s"):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": "a"}])
    assert records(StoreLayout(tmp_path / "store")) and not (tmp_path / "elsewhere" / "store").exists()


def test_a_share_names_the_default_job_when_it_has_calls_beside_a_named_one(layout):
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    for i in range(3):
        with client.run(session=f"t{i}"):                                    # wrap's own job, "agent"
            client.messages.create(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": f"a {i}"}])
        with client.run("researcher", session=f"t{i}"):
            client.messages.create(model="claude-sonnet-5-5", max_tokens=8, messages=[{"role": "user", "content": f"b {i}"}])
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: tuple(v) if isinstance(v, list) else v for k, v in host.items()}), since="7d")
    text = rep.text(cs.advise(rep), [])
    assert "jobs agent + researcher's bill" in text

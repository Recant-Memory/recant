"""The first step needs no configuration: `recant.wrap(client)` with nothing else, then `recant cost` with no flags.
The store is `$RECANT_HOME` or `./.recant/`, the tenant "default", and every command finds the same; a store made this
way carries its own .gitignore."""
import pytest

from recant import cli, wrap
from recant.store import StoreLayout
from recant.log import read_run
from test_wrap_paths import API, REQ, sdk


def calls_in(store, tenant="default"):
    d = StoreLayout(store).log_dir(tenant)
    return [r for p in sorted(d.glob("*.jsonl")) for r in read_run(p, typed=False) if r["kind"] == "model_call"] if d.is_dir() else []


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RECANT_HOME", raising=False)
    return tmp_path


def test_wrap_with_no_arguments_writes_to_dot_recant_under_the_default_tenant(here):
    client = wrap(sdk(API()))
    with client.run(session="s"):
        client.messages.create(**REQ)
    store = here / ".recant"
    assert (store / ".gitignore").read_text().splitlines()[-1] == "*"
    assert len(calls_in(store)) == 1


def test_recant_cost_with_no_flags_reads_the_same_store(here, capsys):
    client = wrap(sdk(API()))
    for i in range(3):
        with client.run(session=f"s{i}"):
            client.messages.create(**REQ)
    assert cli.main(["cost"]) == 0
    out = capsys.readouterr().out
    assert "3 runs, 3 calls" in out


def test_recant_cost_with_nothing_metered_says_what_to_do(here):
    with pytest.raises(SystemExit) as e:
        cli.main(["cost"])
    assert "no calls metered yet" in str(e.value) and "recant.wrap(client)" in str(e.value)


def test_recant_home_is_the_store_when_set(here, monkeypatch):
    monkeypatch.setenv("RECANT_HOME", str(here / "elsewhere"))
    client = wrap(sdk(API()))
    with client.run(session="s"):
        client.messages.create(**REQ)
    assert len(calls_in(here / "elsewhere")) == 1 and not (here / ".recant").exists()


def test_the_switch_with_no_store_is_the_default_store_s(here, capsys):
    client = wrap(sdk(API()))
    assert cli.main(["off"]) == 0                                             # no --store: the default one
    assert client.messages.create(**REQ).content[-1].text == "ok"
    assert calls_in(here / ".recant") == []                                   # off: nothing metered
    assert cli.main(["on"]) == 0
    with client.run(session="s"):
        client.messages.create(**REQ)
    assert len(calls_in(here / ".recant")) == 1


def _store_with_calls(here, n_runs):
    from recant import costshape as cs
    client = wrap(sdk(API()))
    for i in range(n_runs):
        msgs = []
        with client.run("agent", session=f"s{i}"):
            for q in ("a", "b"):
                msgs.append({"role": "user", "content": f"{q} {i}"})
                client.messages.create(model="claude-sonnet-5-5", max_tokens=64, system="Policy. " * 50, messages=msgs)
                msgs.append({"role": "assistant", "content": "ok"})
    return cs.report(StoreLayout(here / ".recant"), "default", since="7d")


def test_the_first_report_leads_with_spent_can_save_and_measured_on(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)                                     # 24 calls
    bill = sum(rep.cost(c) for c in rep.calls)
    saves = cs.Lever("breakpoint.add", "applied", "agent", 0, 0.5, 0.40, True)          # 40% of the agent's bill
    costs = cs.Lever("ttl.prefix", "applied", "agent", 0, -0.1, -0.25, True)            # would cost: never offered
    lines = cs.lead(rep, [saves, costs])
    assert lines[0].startswith(f"You spent ${bill:,.2f} on 24 calls over ")
    assert lines[1] == (f"Recant can save about ${0.40 * bill:,.2f} of that (40%) with no change to what the model sees: "
                        "breakpoint.add. Run `recant apply`.")                    # in dollars on what was measured;
    assert "a week" not in lines[1]                                       # no weekly rate on traffic under a day
    assert lines[2].startswith("Measured on 24 calls over ")


def test_too_little_traffic_is_said_and_nothing_is_sized(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 5)                                      # 10 calls
    lines = cs.lead(rep, [cs.Lever("breakpoint.add", "applied", "agent", 0, 0.5, 0.40, True)])
    assert lines[1].startswith("Too little traffic to size a saving yet: 10 calls metered, and Recant needs about 20")
    assert not any("save about" in x for x in lines)
    assert cs._period(1) == "1 second" and cs._period(30) == "30 seconds"


def test_every_json_output_names_its_schema(here, capsys):
    import json as _json
    client = wrap(sdk(API()))
    for i in range(3):
        with client.run(session=f"s{i}"):
            client.messages.create(**REQ)
    for argv, schema in ((["cost", "--json"], "recant-cost/1"), (["status"], "recant-switch/1"),
                         (["apply", "--json", "--dry-run"], "recant-apply/1"), (["status", "--json"], "recant-switch/1"),
                         (["doctor", "--json"], "recant-doctor/1"), (["feedback", "--json"], "recant-feedback/1")):
        assert cli.main(argv) == 0
        assert _json.loads(capsys.readouterr().out)["schema"] == schema


def test_cost_json_carries_the_three_lead_lines_as_its_summary(here, capsys):
    import json as _json
    _store_with_calls(here, 12)
    assert cli.main(["cost", "--json"]) == 0
    summary = _json.loads(capsys.readouterr().out)["summary"]
    assert len(summary) == 3 and summary[0].startswith("You spent $") and summary[2].startswith("Measured on 24 calls")


def test_feedback_json_writes_nothing_and_asks_nothing(here, capsys, monkeypatch):
    import json as _json
    monkeypatch.setattr("builtins.input", lambda *_: (_ for _ in ()).throw(AssertionError("asked")))
    assert cli.main(["feedback", "--json"]) == 0
    assert _json.loads(capsys.readouterr().out)["written"] is None
    assert not list(here.glob("recant-feedback-*.zip"))


def test_calls_outside_any_run_count_toward_apply_readiness(here):
    """recant.wrap with no client.run keeps one run open per client for the process: its calls are the evidence."""
    from recant import costshape as cs
    client = wrap(sdk(API()))
    msgs = []
    for i in range(12):
        for q in ("a", "b"):
            msgs.append({"role": "user", "content": f"{q} {i}"})
            client.messages.create(model="claude-sonnet-5-5", max_tokens=64, system="Policy. " * 50, messages=msgs)
            msgs.append({"role": "assistant", "content": "ok"})
    client.close()
    layout = StoreLayout(here / ".recant")
    rep = cs.report(layout, "default", cli._host(None, layout, "default"), since="7d")     # the host recant.wrap declared
    assert len(rep.runs(cs._agent(rep))) == 1 and cs.evidence(rep, cs._agent(rep)) == 24
    ok, why = cs.eligible(rep, cs.Lever("breakpoint.add", "applied", "agent", 0, 0.5, 0.40, True))
    assert ok, why


def test_the_spent_line_counts_failed_calls_and_an_all_failed_window_says_so(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    ok_bill = sum(rep.cost(c) for c in rep.calls)
    rep.failed = [dict(rep.calls[0], error="overloaded")]
    lines = cs.lead(rep, [])
    assert lines[0].startswith(f"You spent ${ok_bill + rep.cost(rep.calls[0]):,.2f} on 25 calls over ")
    c0 = rep.cost(rep.calls[0])
    assert (f"${c0:,.2f}" if c0 >= 0.005 else "under $0.01") + " of it on 1 that failed" in lines[0]
    rep.failed, rep.calls = list(rep.calls), []
    lines = cs.lead(rep, [])
    assert "every one of them failed" in lines[0] and not any("No calls metered" in x for x in lines)


def test_a_saving_is_sized_on_the_bill_since_the_jobs_last_setting_change(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    later = sorted(c["start"] for c in rep.calls)[12]                 # a change halfway through the window
    rep.changes = [{"at": later - 0.001, "jobs": ["*"], "what": "setting change"}]
    since = rep.restrict("agent")
    assert since is not rep and 0 < since.job_bill(cs._agent(since)) < rep.job_bill(cs._agent(rep))
    lv = cs.Lever("breakpoint.add", "applied", "agent", 0, 0.5, 0.40, True)
    line = cs.lead(rep, [lv])[1]
    assert f"${0.40 * since.job_bill(cs._agent(since)):,.2f}" in line
    assert f"${0.40 * rep.job_bill(cs._agent(rep)):,.2f}" not in line


def test_the_latency_line_names_slow_calls_a_p95_would_hide():
    from recant import costshape as cs
    calls = [dict(run="r", session=f"s{i}", start=float(i), latency=1.0, ttft=0.5, overhead=0.001) for i in range(80)]
    calls += [dict(run="r", session=f"t{i}", start=100.0 + i, latency=62.0, ttft=61.7, overhead=0.001) for i in range(4)]
    line = cs.render_latency(cs.latency(calls))[0]
    assert "p95 1.00 s" in line and "4 calls over 30 s, the longest 62 s with its first token at 62 s" in line


def test_the_leads_weekly_figure_is_the_tables(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    rep.calls[0]["start"] -= 2 * 86400                                    # two days of traffic in a seven-day window
    lv = cs.Lever("breakpoint.add", "applied", "agent", 0, 8400.0, 0.40, True)
    line = cs.lead(rep, [lv])[1]
    assert "about $8,400.00 a week" in line



def test_a_failed_call_with_no_billed_usage_is_said_so_and_the_counts_agree(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    rep.failed = [dict(rep.calls[0], uncached=0, cache_read=0, cache_write=0, output=0, error="timeout")]
    lines = cs.lead(rep, [cs.Lever("breakpoint.add", "applied", "agent", 0, 0.5, 0.40, True)])
    assert "on 25 calls" in lines[0] and "1 of them failed with no usage reported" in lines[0] and "$0.00" not in lines[0]
    assert lines[2].startswith("Measured on 24 calls over ") and "(the ones that succeeded)" in lines[2]


def test_the_output_cap_line_points_at_thinking_when_thinking_fills_the_output(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    for c in rep.calls:
        c["output"], c["reasoning"], c["stop_reason"] = 700, 400, "max_tokens"
    note = next(lv for lv in cs.detect_output_cap(rep)).note
    assert "57% of the job's output was thinking" in note and "Thinking you didn't ask for" in note
    for c in rep.calls:
        c["reasoning"] = 0
    assert "thinking" not in next(lv for lv in cs.detect_output_cap(rep)).note



def test_a_failed_call_on_an_unpriced_model_is_usage_not_none(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    rep.failed = [dict(rep.calls[0], model="some-unpriced-model", uncached=1000, cache_read=0, cache_write=0, output=100,
                       error="timeout")]
    assert "1 of them failed, with 1,100 tokens of usage on models with no price" in cs.lead(rep, [])[0]


def test_the_lead_and_the_header_measure_the_same_span(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    t0 = rep.calls[0]["start"]
    for i, c in enumerate(rep.calls):
        c["start"], c["latency"] = t0 + i * (86350 / (len(rep.calls) - 1)), 120.0   # requests over 86,350 s; replies past a day
    lv = cs.Lever("breakpoint.add", "applied", "agent", 0, 70.0, 0.40, True)
    assert "a week" in cs.lead(rep, [lv])[1]                                         # past a day, as the table sees it
    assert cs.traffic_span(rep.calls) == 86350 + 120



def test_a_late_failed_call_does_not_stretch_the_leads_span(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    t0 = min(c["start"] for c in rep.calls)
    for i, c in enumerate(rep.calls):
        c["start"], c["latency"] = t0 + i, 0.0                           # 24 calls over 23 seconds
    rep.failed = [dict(rep.calls[0], start=t0 + 86500, error="timeout")]
    lv = cs.Lever("breakpoint.add", "applied", "agent", 0, 70.0, 0.40, True)
    lines = cs.lead(rep, [lv])
    assert "over 23 seconds" in lines[0] and "a week" not in lines[1]



def test_an_all_failed_window_gives_its_span_in_the_header_too(here):
    from recant import costshape as cs
    rep = _store_with_calls(here, 12)
    rep.failed, rep.calls = [dict(c, error="timeout") for c in rep.calls], []
    head = rep.text([], []).splitlines()[0]
    assert "; traffic in it from " in head

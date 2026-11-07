"""`recant apply`: the safe levers recant cost found, turned on in one command; `--dry-run` says what each changes in a
request and changes nothing; a lever not ready says why. Run on the bench suite's caching simulator, where an agent
that only appends to its conversation makes breakpoint.add worth applying."""
import json

import pytest

from recant import cli, costshape as cs, wrap
from recant.store import StoreLayout
from test_bench import SimAPI, marked

HANDBOOK = "You are the support agent. " + "Policy line. " * 400


@pytest.fixture
def busy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RECANT_HOME", str(tmp_path / "rh"))
    api = SimAPI()
    client = wrap(api, store=tmp_path / "store", tenant="t")
    for i in range(25):
        msgs = []
        with client.run("agent", session=f"s{i}"):
            for q in ("What now?", "Can I return it?", "How long?"):
                msgs.append({"role": "user", "content": f"order {i}: {q}"})
                r = client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK, messages=msgs)
                msgs.append({"role": "assistant", "content": r.content[0].text})
    return tmp_path, api, client


def test_dry_run_says_what_changes_and_changes_nothing(busy, capsys):
    tmp, api, client = busy
    assert cli.main(["apply", "--store", str(tmp / "store"), "--tenant", "t", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("would apply breakpoint.add") and "the model is sent the same input" in out
    assert cs.overlay(StoreLayout(tmp / "store"), "t") == {}


def test_apply_turns_it_on_and_the_next_call_carries_it_out(busy, capsys):
    tmp, api, client = busy
    assert cli.main(["apply", "--store", str(tmp / "store"), "--tenant", "t", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in res["applied"]] == ["breakpoint.add"] and res["not_ready"] == []
    assert "breakpoint.add" in cs.overlay(StoreLayout(tmp / "store"), "t")
    n = len(api.bodies)
    with client.run("agent", session="after"):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK,
                               messages=[{"role": "user", "content": "one more"}])
    assert marked(api.bodies[n]["system"]) and marked(api.bodies[n]["messages"][-1])
    assert cli.main(["cost", "--store", str(tmp / "store"), "--tenant", "t"]) == 0
    assert "The safe savings are on (breakpoint.add)" in capsys.readouterr().out
    assert cli.main(["apply", "--store", str(tmp / "store"), "--tenant", "t"]) == 0
    assert capsys.readouterr().out.strip() == "already on: breakpoint.add; nothing more to apply."


def test_a_lever_not_ready_says_why(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECANT_HOME", str(tmp_path / "rh"))
    api = SimAPI()
    client = wrap(api, store=tmp_path / "store", tenant="t")
    for i in range(8):                                                 # 8 runs of 3 calls: sized, under 20 runs
        msgs = []
        with client.run("agent", session=f"s{i}"):
            for q in ("a", "b", "c"):
                msgs.append({"role": "user", "content": f"order {i}: {q}"})
                r = client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK, messages=msgs)
                msgs.append({"role": "assistant", "content": r.content[0].text})
    assert cli.main(["apply", "--store", str(tmp_path / "store"), "--tenant", "t"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("not yet breakpoint.add: 8 runs of the job, under 20")
    assert cs.overlay(StoreLayout(tmp_path / "store"), "t") == {}
    assert cli.main(["cost", "--store", str(tmp_path / "store"), "--tenant", "t"]) == 0
    assert "`recant apply` turns it on once the job has 20 runs (8 so far)" in capsys.readouterr().out

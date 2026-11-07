"""`recant doctor`: one line per check, a fix for each problem, a non-zero exit while anything needs fixing."""
import json

import pytest

from recant import cli, doctor, wrap
from test_wrap_paths import API, REQ, sdk


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RECANT_HOME", raising=False)
    return tmp_path


def checks(capsys):
    return {c["check"]: c for c in json.loads(capsys.readouterr().out)["checks"]}


def test_before_anything_is_wrapped_it_says_to_wrap_and_makes_nothing(here, capsys):
    assert cli.main(["doctor", "--json"]) == 1
    c = checks(capsys)
    assert c["store"]["ok"] and "does not exist yet" in c["store"]["detail"]
    assert not c["metering"]["ok"] and "recant.wrap(anthropic.Anthropic())" in c["metering"]["fix"]
    assert not (here / ".recant").exists()


def test_a_wrapped_app_with_calls_is_all_good(here, capsys):
    client = wrap(sdk(API()))
    with client.run(session="s"):
        client.messages.create(**REQ)
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("all good") and "1 model calls metered" in out


def test_the_switch_off_and_an_unpriced_model_are_each_a_line_to_fix(here, capsys):
    client = wrap(sdk(API()))
    with client.run(session="s"):
        client.messages.create(**dict(REQ, model="house-model-1"))
    assert cli.main(["off"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    c = checks(capsys)
    assert not c["switch"]["ok"] and c["switch"]["fix"].startswith("recant on")
    assert not c["prices"]["ok"] and "house-model-1" in c["prices"]["detail"] and "--prices" in c["prices"]["fix"]


def test_doctor_writes_nothing_anywhere_not_even_the_friction_log(tmp_path, monkeypatch):
    home = tmp_path / "new-store"
    monkeypatch.setenv("RECANT_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.chdir(tmp_path)
    before = sorted(p for p in tmp_path.rglob("*"))
    cli.main(["doctor"])
    assert sorted(p for p in tmp_path.rglob("*")) == before and not home.exists()


def test_the_writable_check_leaves_an_existing_file_of_the_same_kind_alone(tmp_path, monkeypatch):
    import os
    store = tmp_path / "s"
    store.mkdir()
    keep = store / f".doctor-{os.getpid()}"
    keep.write_text("someone's data")
    doctor.run(store, "default")
    assert keep.read_text() == "someone's data" and sorted(p.name for p in store.iterdir()) == [keep.name]


def test_coverage_names_another_providers_calls_without_calling_them_a_fault(monkeypatch, tmp_path):
    from recant import costshape as cs

    class Rep:
        watched, calls, profiles = True, [], {}

        def coverage(self):
            return {"calls_seen": 10, "calls_unseen": 4,
                    "unseen": [{"provider": "openai", "host": "api.openai.com", "model": "gpt-4o-mini", "requests": 4}]}
    store = tmp_path / "s"
    client = wrap(sdk(API()), store=store, tenant="default")
    with client.run(session="x"):
        client.messages.create(**REQ)
    monkeypatch.setattr(cs, "report", lambda *a, **k: Rep())
    checks = doctor.run(store, "default")
    cov = [c for c in checks if c["check"] == "coverage"]
    assert len(cov) == 1 and cov[0]["ok"] and "api.openai.com" in cov[0]["detail"] and "Anthropic's only" in cov[0]["detail"]
    Rep.coverage = lambda self: {"calls_seen": 10, "calls_unseen": 2, "unseen": [
        {"provider": "anthropic", "host": "api.anthropic.com", "model": "claude-haiku-5-5", "requests": 2}]}
    cov = [c for c in doctor.run(store, "default") if c["check"] == "coverage"]
    assert not cov[0]["ok"] and "recant.wrap" in cov[0]["fix"]

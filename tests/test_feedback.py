"""the friction log and `recant feedback`. Every command is logged (help calls, failures with their class and
the Recant function, config changes, the first report); the bundle passes the allow-list on every string but the
form's own answers, and holds none of the run's prompts, replies, tasks, salts, paths, hostname or user name."""
import getpass
import json
import platform
import zipfile

import pytest

from recant import feedback, friction
from recant.cli import main
from test_bench import SimAPI, go, setup  # noqa: F401 — the bench fixture and its simulated API


def strings(x, path=()):
    if isinstance(x, str):
        yield path, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from strings(k, path + ("<key>",))
            yield from strings(v, path + (k,))
    elif isinstance(x, list):
        for v in x:
            yield from strings(v, path)


def test_the_friction_log_counts_help_failures_retries_and_config_changes(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    with pytest.raises(SystemExit):
        main(["bench"])                                                   # no --tasks: a usage error
    with pytest.raises(Exception):
        main(["cost", "--store", str(tmp_path / "nowhere"), "--tenant", "bad tenant!"])
    main(["off", "--store", str(tmp_path / "s")])
    main(["cost", "--store", str(tmp_path / "s2"), "--tenant", "00000000-0000-0000-0000-000000000001"])
    s = friction.summary()
    assert s["commands"] == 5 and s["help_calls"] == 1 and s["config_changes"] == 1 and s["failed_commands"] == 2
    assert s["first_report_ts"] is not None and s["minutes_to_first_report"] is not None
    assert any(k.endswith(" in unknown") or " in recant." in k for k in s["errors"])
    rows = [json.loads(x) for x in friction.path().read_text().splitlines()]
    assert not any(str(tmp_path) in json.dumps(r) for r in rows)              # no arguments' values, no paths


def test_the_bundle_passes_the_allow_list_and_holds_nothing_from_the_runs(setup, monkeypatch):
    tmp, tasks = setup
    api = SimAPI()
    go(tasks(10, explode={6}), tmp / "bench" / "r1", api, calibrate=3)
    answers = iter(["", "the --run hook's signature", "", "latency per tool", "not yet, want more runs", "", "y"])
    said = []
    p = feedback.run(bench_dirs=[tmp / "bench" / "r1"], eval_dirs=[], out_dir=tmp, ask=lambda q: next(answers), say=said.append)
    bundle = json.loads(zipfile.ZipFile(p).read("bundle.json"))
    assert bundle["form"]["had_to_read_code_or_guess"] == "the --run hook's signature"
    assert bundle["reports"][0]["report"]["cost"]["per_task"]["B"] > 0 and bundle["features"]["bench_clients"] == ["sync"]
    form = bundle.pop("form")
    bad = [(path, s) for path, s in strings(bundle)                            # keys: Recant's field names; values: their field's rule
           if not (path[-1:] == ("<key>",) and (s in feedback.FIELDS or feedback.error_name(s) == s))
           and not (path[-1:] != ("<key>",) and feedback.scrub_string(s, path[-1] if path else None) == s)]
    assert not bad, bad[:5]
    text_ = json.dumps(bundle) + json.dumps(form)
    leaks = ["Policy line", "What now?", "order 3", "Send a photo", "support agent", "recant-bench", str(tmp), platform.node(),
             getpass.getuser(), "RuntimeError: the agent fell over", "the agent fell over"]
    for body in api.bodies:                                                    # every prompt the run sent
        leaks += [m["content"] for m in body["messages"] if isinstance(m["content"], str)]
    assert [x for x in leaks if x and x in text_] == []
    assert bundle["reports"][0]["report"]["failures"]["B"]["errors"] == {"RuntimeError": 1}       # a builtin class: kept
    assert any("Recant sent nothing" in x and feedback.ISSUES in x for x in said)


def test_nothing_is_written_without_a_yes(setup):
    tmp, _ = setup
    said = []
    out = feedback.run(bench_dirs=[], eval_dirs=[], out_dir=tmp, form=False, ask=lambda q: "n", say=said.append)
    assert out is None and not list(tmp.glob("recant-feedback-*.zip")) and said[-1] == "Nothing written."


def test_two_jobs_levers_are_told_apart_without_their_names():
    feedback._JOBS.clear()
    ids = [feedback.lever_id(x) for x in ("effort.researcher", "effort.writer", "reasoning.researcher", "effort.agent")]
    assert ids == ["effort.<job 1>", "effort.<job 2>", "reasoning.<job 1>", "effort.agent"]
    assert not any("researcher" in i or "writer" in i for i in ids)


def test_what_the_allow_list_left_out_is_said_by_where_never_by_what():
    dropped: list = []
    out = feedback.scrub({"header": {"model": "house-model-3", "a customer's name": 1}, "levers": [{"id": "x.y"}]}, dropped=dropped)
    assert out == {"header": {"model": None}, "levers": [{"id": None}]}
    assert dropped == ["header.model", "header.<key>", "levers[0].id"]
    assert "house-model" not in json.dumps(dropped) and "customer" not in json.dumps(dropped)

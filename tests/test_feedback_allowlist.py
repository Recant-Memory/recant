"""The feedback bundle keeps only the field names Recant writes, so a scope, a job or any other name a developer
chose never reaches it; strings pass by field, not by shape; `recant feedback` always asks before writing."""
import json

import pytest

from recant import feedback, wrap
from recant import eval_memory as em
from recant.cli import main, run_cost, _host
from recant.store import StoreLayout
from test_bench import SimAPI
import test_eval_memory as TE

NAMES = ["jsmith", "acme_payroll", "orca_billing_agent", "JohnSmithAccountError", "deadbeefcafe1234"]


def eval_report(tmp_path):
    sessions = [dict(s, scope={"user:ana": "jsmith", "user:bo": "acme_payroll"}[s["scope"]]) for s in TE.SESSIONS]
    p = tmp_path / "s.jsonl"
    p.write_text("".join(json.dumps(s) + "\n" for s in sessions))
    sess = em.load_sessions(str(p))
    qp = tmp_path / "q.jsonl"
    sdk = TE.FakeSDK()
    em.make_questions(sess, raw=sdk, model="claude-sonnet-5-5", out_path=str(qp), root=tmp_path / "e")
    em.evaluate(sess, str(qp), raw=TE.FakeSDK(), model="claude-sonnet-5-5", out=str(tmp_path / "e"), extract_client=TE.Extractor(),
                say=lambda *_: None)
    return tmp_path / "e"


def cost_summary(tmp_path):
    layout = StoreLayout(tmp_path / "host-store")
    client = wrap(SimAPI(), store=layout, tenant="00000000-0000-0000-0000-000000000001", job="orca_billing_agent")
    for i in range(3):
        with client.run("orca_billing_agent", session=f"s{i}"):
            client.messages.create(model="claude-sonnet-5-5", max_tokens=10, system="x" * 500, messages=[{"role": "user", "content": "hi"}])
    c = json.loads(run_cost(layout, "00000000-0000-0000-0000-000000000001", _host(None, layout, "00000000-0000-0000-0000-000000000001"), as_json=True))
    return {"shape": c.get("shape"), "levers": [{"id": lv.get("id"), "kind": lv.get("kind"), "saves_usd": lv.get("saves_usd"),
                                                  "share": lv.get("share")} for lv in c.get("levers", [])]}


def test_names_a_developer_chose_never_reach_the_bundle(tmp_path):
    e = eval_report(tmp_path)
    raw = json.loads((e / "report.json").read_text())
    raw.setdefault("failures", {})["A"] = {"errors": {"JohnSmithAccountError": 2}}          # a host's own exception class
    raw["header"]["note_id"] = "deadbeefcafe1234"                                          # hex in a field Recant never names
    (e / "report.json").write_text(json.dumps(raw))
    bundle, _ = feedback.build(bench_dirs=[], eval_dirs=[e], cost_json=cost_summary(tmp_path))
    text_ = json.dumps(bundle)
    assert [n for n in NAMES if n in text_] == []
    assert bundle["reports"][0]["report"]["arms"]["notes"]["correct"] >= 0                 # the numbers survive


def test_every_field_recant_writes_is_on_the_list(tmp_path):
    """A report that gains a field must be reviewed: every key in a real bench and eval report is either on the bundle's
    list of field names, or a key of a dict the bundle turns into a list of values."""
    from test_bench import go
    import test_bench
    (tmp_path / "bench_agent.py").write_text(test_bench.AGENT)
    import sys, os
    sys.path.insert(0, str(tmp_path))
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        t = tmp_path / "t.jsonl"
        t.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(8)))
        go(str(t), tmp_path / "b", SimAPI(), calibrate=2)
    finally:
        os.chdir(cwd)
    e = eval_report(tmp_path)
    unlisted = set()

    def walk(x, parent=None):
        if isinstance(x, dict):
            listified = parent in feedback.LISTED_BY_VALUE
            for k, v in x.items():
                if not listified and k not in feedback.FIELDS:
                    unlisted.add(k)
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, parent)
    for d in (tmp_path / "b", e):
        walk(json.loads((d / "report.json").read_text()))
    walk(cost_summary(tmp_path))
    walk(feedback.build(bench_dirs=[tmp_path / "b"], eval_dirs=[e])[0])
    assert not unlisted, sorted(unlisted)


def test_feedback_has_no_yes_and_always_asks(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["feedback", "--help"])
    assert "--yes" not in capsys.readouterr().out
    asked = []
    out = feedback.run(bench_dirs=[], eval_dirs=[], out_dir=tmp_path, form=False, ask=lambda q: asked.append(q) or "y", say=lambda *_: None)
    assert out is not None and any("Write this bundle" in q for q in asked)

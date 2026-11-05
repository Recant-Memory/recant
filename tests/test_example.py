"""The README's quickstart runs as written, with a scripted client in place of a real one: the whole loop — write a day's turns, extract them into proposals and tasks, review, fit — against the
example machinery files."""
import importlib.util
import json
import re
from pathlib import Path

from recant.provider import Completion, Usage, text

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "memory_v2_agent.py"


def load_example():
    spec = importlib.util.spec_from_file_location("memory_v2_agent", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json(d):
    return Completion(content=(text(json.dumps(d)),), stop_reason="end_turn", usage=Usage(input=120, output=30))


class ScriptedClient:
    """A ModelClient: complete() and nothing else. Nominates the decision, the cause and the reminder; writes the
    first two up as memories, routes the reminder as a task, and writes that up as an open task."""

    def complete(self, req):
        user = req.messages[0].content[0].text
        if req.job == "nominate":
            marks = {"decided": "rule_or_decision", "caused": "root_cause", "Remind": "rule_or_decision"}
            spans = [{"marker": mk, "start_line": int(n), "end_line": int(n), "quote": line}
                     for n, line in re.findall(r"^(\d+): (.*)$", user, flags=re.M) for w, mk in marks.items() if w in line]
            return _json({"spans": spans})
        n = int(re.search(r"The moment is lines (\d+)-", user).group(1))
        line = re.search(rf"^{n}: (.*)$", user, flags=re.M).group(1)
        common = {"actor_class": "principal", "actor_id": "me", "quote": line, "quote_start_line": n, "quote_end_line": n,
                  "channel": "", "timestamp": "", "scope": "scanner", "notes": ""}
        if req.job == "extract_tasks":
            return _json({**common, "verdict": "task", "not_a_task_reason": "none", "claim": "Rotate the cache keys.",
                          "status": "open", "closed_line": 0})
        if "Remind" in line:
            return _json({**common, "verdict": "not_a_memory", "not_a_memory_reason": "task", "claim": "", "kind": "stated",
                          "supersession_condition": ""})
        claim = "The scanner ships on Fridays." if "Fridays" in line else "The stale cache key caused Monday's outage."
        return _json({**common, "verdict": "memory", "not_a_memory_reason": "none", "claim": claim, "kind": "stated",
                      "supersession_condition": "Until it changes."})


def test_the_example_writes_extracts_reviews_and_serves_with_a_one_method_client(tmp_path, capsys):
    mod = load_example()
    out = mod.run(tmp_path / "store", ScriptedClient(), question="why the outage")
    assert out["session"].startswith("hot/sessions/2026-01-05/") and out["lines"] == 5
    assert out["spans"] == 3 and out["proposed"] == 2 and out["tasks"] == 1 and out["entries"] == 2
    assert sum(u["calls"] for u in out["usage"].values()) == 5                  # one nomination, three extractions, one task
    assert out["core_tokens"] > 0 and "===== memory core =====" in out["prefix"] and "Fridays" in out["prefix"]
    printed = capsys.readouterr().out
    assert "proposed: The scanner ships on Fridays." in printed and "#L1-L1]" in printed
    # the example's own placeholder client is well-formed too: it runs and proposes nothing
    quiet = mod.run(tmp_path / "store2", mod.MyClient())
    assert quiet["proposed"] == 0 and quiet["entries"] == 0 and quiet["lines"] == 5


def test_the_default_machinery_is_real_and_host_neutral(tmp_path):
    """A second developer gets working memory without borrowing a consumer's files — real model IDs (each with a
    list price), full prompts, reasoning off, the default budgets; no consumer's name or channel in them."""
    from recant import costshape as cs
    from recant.machinery import DEFAULT_ARTIFACT, DEFAULT_KERNEL, Machinery
    from recant.store import StoreLayout
    layout = StoreLayout(tmp_path / "s")
    assert Machinery.seed_defaults(layout)
    m = Machinery.load(layout)
    a = m.artifact
    models = {a.extractor.first_pass.model, a.extractor.steady_state.model, a.extractor.tasks.model, a.nomination.model,
              a.judge.model, a.reader.model, *a.routing.values()}
    assert models <= set(cs.LIST_PRICES), models - set(cs.LIST_PRICES)
    assert all(len(p) > 1500 for p in (a.extractor.steady_state.prompt, a.extractor.first_pass.prompt, a.nomination.prompt))
    assert a.extractor.steady_state.reasoning == "off" and "own_tool_failure" in a.extractor.steady_state.prompt
    assert (m.kernel.serve.core_tokens, m.kernel.serve.slice_tokens) == (25000, 12500)
    from test_boundary import PATHS
    text_ = (DEFAULT_ARTIFACT.read_text() + DEFAULT_KERNEL.read_text()).lower()
    assert not [w for w in PATHS + ["slack"] if w.lower() in text_]                  # host-neutral: no path, no channel


# ---------------------------------------------------------------- the other two examples, by their README commands
def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE.parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_notes_example_keeps_what_was_said_refuses_what_was_not_and_answers_from_the_prefix(tmp_path):
    out = _load("notes_agent").run(tmp_path / "store", _load("notes_agent").OfflineClient())
    assert out["notes"] == ["company/-/accountant = Alice Example", "company/-/rent = 3,300 a month"]        # the auditor was never said
    assert out["replies"][1] == ["Alice Example; the rent is 3,300 a month."]                              # served in session two's prefix


def test_the_cost_quickstart_runs_its_readme_sequence_end_to_end(tmp_path, capsys):
    from recant import cli
    store, cq = str(tmp_path / "store"), _load("cost_quickstart")
    cost = lambda *a: (cli.main(["cost", "--store", store, "--tenant", "demo", *a]), capsys.readouterr().out)[1]   # noqa: E731
    cq.main([store, "--tasks", "30", "--offline"])
    assert "levers carried out on the last call: none" in capsys.readouterr().out
    assert re.search(r"available\s+breakpoint\.add\s+\$\d", cost())
    night = cost("--nightly")
    assert re.search(r"applied\s+breakpoint\.add", night) and "ttl.prefix" not in night                  # $0.00 levers left out

    cq.main([store, "--tasks", "50", "--offline"])
    assert "system.0 (5m)" in capsys.readouterr().out                                                     # carried out, no host code
    cost("--nightly")
    # judged per affected call against the prediction per call, with the volume it covers
    assert re.search(r"breakpoint\.add: (met|MISSED) — \$[\d.]+ a call against \$[\d.]+ predicted \(\d+%; 300 calls this window; "
                     r"the prediction assumed about [\d,]+ a week\)", cost())

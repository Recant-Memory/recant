import ast
from pathlib import Path

import pytest
from ulid import ULID

from recant.eval import NOTHING, GuardError, ItemError, append_draft, load, validate_item
from recant.eval.pools import GUARDED, PURPOSES, write_pool
from recant.eval.sample import stratified_sample, stratum_key
from recant.eval.stats import exact_interval, rate
from recant_fixtures import TENANT


def item(kind="yield", marker="root_cause", month="2026-01", hard=False, **over):
    base = {"id": str(ULID()), "question": "What caused the outage?", "answer": "A stale cache key.",
            "episode": "notes/2026-01-02.md#L6-L6", "exposure": 0,
            "tests": {"type": kind, "marker": marker, "hard": hard, "stratum": month}}
    if kind == "noise":
        base.update(answer=NOTHING, episode=None)
        base["tests"]["marker"] = None
    base.update(over)
    return base


def test_the_item_is_five_fields_and_an_id_and_nothing_else():
    it = validate_item(item())
    assert set(it.to_dict()) == {"id", "question", "answer", "episode", "tests", "exposure"}
    for extra in ("candidate_answer", "reviewer", "confidence", "drafted_by"):
        with pytest.raises(ItemError, match=extra):
            validate_item(item(**{extra: "x"}))


def test_noise_items_have_no_episode_and_the_fixed_key_and_only_they_do():
    validate_item(item("noise"))
    with pytest.raises(ItemError, match="episode"):
        validate_item(item("noise", episode="notes/2026-01-02.md#L1-L1"))
    with pytest.raises(ItemError, match="episode"):
        validate_item(item(episode=None))
    with pytest.raises(ItemError, match="noise"):
        validate_item(item(answer=NOTHING))
    with pytest.raises(ItemError, match="noise"):
        validate_item(item("noise", answer="something"))


@pytest.mark.parametrize("bad", [dict(id="x"), dict(question=" "), dict(answer=""), dict(exposure=-1),
                                 dict(episode="notes/../x.md#L1-L1"), dict(exposure="0")])
def test_bad_items_are_refused(bad):
    with pytest.raises(ItemError):
        validate_item(item(**bad))
    with pytest.raises(ItemError):
        validate_item(item(tests={"type": "yield", "marker": None, "hard": False, "stratum": "January"}))


def test_the_key_hash_ignores_exposure_and_tests_and_moves_with_the_key():
    a = validate_item(item())
    same_key = validate_item({**a.to_dict(), "exposure": 9, "tests": {**a.to_dict()["tests"], "hard": True}})
    assert a.key_hash() == same_key.key_hash()
    assert a.key_hash() != validate_item({**a.to_dict(), "answer": "Something else."}).key_hash()


# --- the guard ---------------------------------------------------------------------

def test_guarded_pools_open_for_very_few_purposes(layout):
    write_pool(layout, TENANT, "heldout", [validate_item(item())])
    write_pool(layout, TENANT, "reserve", [validate_item(item())])
    write_pool(layout, TENANT, "founding", [validate_item(item())])
    for pool in GUARDED:
        for purpose in ("scoring", "drafting", "model_review", "audit"):
            with pytest.raises(GuardError):
                load(layout, TENANT, pool, purpose)
        assert len(load(layout, TENANT, pool, "overfitting_check")) == 1
        assert len(load(layout, TENANT, pool, "human_review")) == 1
    assert len(load(layout, TENANT, "founding", "scoring")) == 1
    with pytest.raises(GuardError, match="says what it is for"):
        load(layout, TENANT, "founding", "just looking")


def test_no_harness_module_names_a_guarded_pool():
    """The extractor, the judge and the harness never see the held-out slice."""
    harness = Path(__file__).resolve().parents[1] / "src/recant/harness"
    files = sorted(harness.glob("*.py"))
    assert len(files) >= 5
    offenders = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and node.value in GUARDED | {"overfitting_check"}:
                offenders.append(f"{path.name}:{node.lineno} names {node.value!r}")
    assert not offenders, offenders
    assert {"scoring", "drafting", "model_review"} <= PURPOSES


def test_the_pool_files_are_never_opened_except_through_the_door():
    src = Path(__file__).resolve().parents[1] / "src/recant"
    offenders = [p.name for p in src.rglob("*.py") if p.name != "pools.py" and "pool_path(" in p.read_text()]
    assert not offenders, offenders


def test_drafts_append_once(layout):
    it = validate_item(item())
    append_draft(layout, TENANT, it)
    with pytest.raises(ValueError):
        append_draft(layout, TENANT, it)
    assert load(layout, TENANT, "drafts", "drafting") == [it]


# --- sampling and intervals -----------------------------------------------------------

def pool():
    out = []
    for month in ("2026-01", "2026-02", "2026-03"):
        out += [validate_item(item("yield", "root_cause", month)) for _ in range(6)]
        out += [validate_item(item("yield", "rule_or_decision", month)) for _ in range(6)]
        out += [validate_item(item("noise", month=month)) for _ in range(4)]
    out += [validate_item(item("yield", "root_cause", "2026-02", hard=True)) for _ in range(2)]
    return out


def test_sampling_is_deterministic_under_a_seed_and_differs_across_seeds():
    items = pool()
    a, _ = stratified_sample(items, 12, seed=7)
    b, _ = stratified_sample(list(reversed(items)), 12, seed=7)
    c, _ = stratified_sample(items, 12, seed=8)
    assert [i.id for i in a] == [i.id for i in b] != [i.id for i in c]
    assert len({i.id for i in a}) == 12


def test_every_stratum_is_represented_before_any_is_doubled_and_hard_cases_come_first():
    from recant.eval.sample import primary_key
    chosen, unfilled = stratified_sample(pool(), 4, seed=1)
    assert {primary_key(i) for i in chosen} == {"yield/root_cause", "yield/rule_or_decision", "noise"}   # every marker, then the largest again
    assert chosen[0].tests.hard and unfilled == []
    twelve, _ = stratified_sample(pool(), 12, seed=1)
    assert {i.tests.stratum for i in twelve} == {"2026-01", "2026-02", "2026-03"}


def test_no_marker_vanishes_when_there_are_more_strata_than_slots_and_months_stay_even():
    """A regression: question form can multiply the strata past the slots, and the largest marker must still get one."""
    forms, items = {}, []
    names = ["decision", "directive", "root_cause", "failure", "status", "claim_check", "reversal", "manner"]
    for m, size in [("rule_or_decision", 120), ("directive_to_act", 90), ("root_cause", 40), ("failure_with_reason", 20),
                    ("manner_correction", 8), ("non_compliance", 6), ("reversal", 4)]:
        for k in range(size):
            it = validate_item(item("yield", m, f"2026-0{k % 6 + 1}"))
            forms[it.id] = names[k % 8]
            items.append(it)
    chosen, _ = stratified_sample(items, 36, seed=5, forms=forms)
    by_marker = {}
    for c in chosen:
        by_marker[c.tests.marker] = by_marker.get(c.tests.marker, 0) + 1
    assert len(chosen) == 36 and set(by_marker) == {"rule_or_decision", "directive_to_act", "root_cause", "failure_with_reason",
                                                   "manner_correction", "non_compliance", "reversal"}
    assert by_marker["rule_or_decision"] > by_marker["root_cause"] > by_marker["reversal"] >= 1
    months = [sum(c.tests.stratum == f"2026-0{m}" for c in chosen) for m in range(1, 7)]
    assert max(months) - min(months) <= 2
    assert len({forms[c.id] for c in chosen}) == 8


def test_the_sampler_says_what_it_could_not_fill():
    chosen, unfilled = stratified_sample(pool(), 100, seed=1, expected_strata=["supersession/reversal/plain", "noise/none/plain"])
    assert len(chosen) == 50
    assert unfilled == ["supersession/reversal/plain", "pool exhausted at 50 of 100"]


def test_exact_intervals_match_the_figures_the_record_quotes():
    lo, hi = exact_interval(0, 30)      # "0 errors in 30 gives a 95% interval of [0, 11.6%]"
    assert lo == 0 and hi == pytest.approx(0.116, abs=0.001)
    lo, hi = exact_interval(2, 30)      # "2 in 30 ... gives [0.8%, 22.1%]"
    assert lo == pytest.approx(0.008, abs=0.001) and hi == pytest.approx(0.221, abs=0.001)
    lo, hi = exact_interval(2, 9)       # "two successes in nine gives a 95% interval of [3%, 60%]"
    assert lo == pytest.approx(0.03, abs=0.005) and hi == pytest.approx(0.60, abs=0.005)
    assert exact_interval(50, 50)[1] == 1.0
    assert rate(0, 0) == {"k": 0, "n": 0, "rate": None, "interval_95": None}
    with pytest.raises(ValueError):
        exact_interval(3, 2)

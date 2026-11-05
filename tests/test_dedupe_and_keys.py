"""Dedupe and keys: one question per fact per day, keys that are answers,
no secrets, noise that does not announce itself, items a person wrote, and a sample that carries
the suite's noise share and spreads question forms."""
import json
import re

import pytest
from ulid import ULID

from recant.eval import load, move_between, validate_item
from recant.eval.blind import draw_blind_sample
from recant.eval.ledger import Ledger
from recant.eval.pools import append_draft
from recant.eval.review import decide
from recant.harness.dedupe import dedupe
from recant.harness.draft import Drafter, key_problem, opener, redact_secrets, source_share
from recant.harness.nominate import Span
from recant.log import LogWriter, read_run
from recant.machinery import Machinery
from recant.provider import Usage, text
from recant.provider.base import Raw
from recant.provider.fake import FakeProvider
from recant.store import ingest
from pipeline_fixtures import build_corpus
from recant_fixtures import ALLOWLIST, TENANT

FP = None


def run(layout, machinery, job):
    return LogWriter(layout, TENANT, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash)


@pytest.fixture
def world(layout, machinery, tmp_path):
    ingest(layout, TENANT, build_corpus(tmp_path / "corpus"), ALLOWLIST)
    return layout, machinery


def put(layout, machinery, log, *, q, a="An answer.", kind="yield", marker="root_cause", episode="notes/2026-01-01.md#L6-L6",
        pool="drafts", near=None, form="root_cause"):
    item = validate_item({"id": str(ULID()), "question": q, "answer": "nothing relevant" if kind == "noise" else a,
                          "episode": None if kind == "noise" else episode, "exposure": 0,
                          "tests": {"type": kind, "marker": None if kind == "noise" else marker, "hard": False, "stratum": "2026-01"}})
    append_draft(layout, TENANT, item, pool=pool)
    log.emit("item_drafted", item=item.id, item_type=kind, origin="noise" if kind == "noise" else "marked",
             source_span=near or episode, marker=item.tests.marker, model_id="m", prompt_version="d1",
             fingerprint=machinery.drafting_fingerprint(), question=q, candidate_answer=item.answer, form=form)
    return item.id


def grouping(groups):
    return lambda model, req: Raw((text(json.dumps({"groups": groups(req.messages[0].content[0].text)})),), "end_turn", Usage(input=5, output=5))


def test_the_same_fact_written_three_ways_on_one_day_becomes_one_item(world):
    layout, machinery = world
    with run(layout, machinery, "draft") as log:
        a = put(layout, machinery, log, q="Is the site broken on our side?", a="The test machine cannot resolve example.com.",
                episode="notes/2026-01-01.md#L6-L6")
        b = put(layout, machinery, log, q="Is example.com blocking us, or is it our client?", a="Unreachable from the test machine; requests hang.",
                episode="notes/2026-01-01.md#L15-L15", marker="reversal")
        c = put(layout, machinery, log, q="Was the fetch failure a bug in our code?", a="A lookup problem on the test machine, not a bug in the client code.",
                episode="notes/.hidden/talk/2026-01-01.txt#L1-L2", marker="rule_or_decision")
        other_day = put(layout, machinery, log, q="Is the site broken on our side?", a="Still unreachable.", episode="notes/2026-01-02.md#L6-L6")
        same_span = put(layout, machinery, log, q="Completely different words about the cache", episode="notes/2026-01-03.md#L6-L6")
        same_span2 = put(layout, machinery, log, q="Another phrasing entirely, limits and scope", episode="notes/2026-01-03.md#L6-L6", marker="reversal")

    def groups(user):
        return [{"keep": a, "drop": [b, c], "reason": "all three ask whether the failure was ours"}] if "2026-01-01" in user else []

    with run(layout, machinery, "dedupe") as log:
        out = dedupe(layout, TENANT, machinery, FakeProvider(machinery, log, grouping(groups)), log)
    ids = {i.id for i in load(layout, TENANT, "drafts", "drafting")}
    assert a in ids and b not in ids and c not in ids and other_day in ids          # another day's is another item
    assert len({same_span, same_span2} & ids) == 1 and out["merged_by_function"] == 1 and out["merged_by_model"] == 2
    survivor = next(i for i in load(layout, TENANT, "drafts", "drafting") if i.id == a)
    merged = [r for r in read_run(log.path, typed=False) if r["kind"] == "draft_merged"]
    assert {m["item"] for m in merged} >= {b, c} and all(m["into"] for m in merged)
    assert sorted(next(m for m in merged if m["item"] == b)["markers"]) == ["reversal", "root_cause", "rule_or_decision"]
    assert survivor.tests.marker in ("reversal", "rule_or_decision")               # it keeps a rarer marker than root_cause


def test_a_noise_draft_is_held_when_its_twin_was_and_dropped_when_the_record_answers_it(world):
    layout, machinery = world
    with run(layout, machinery, "draft") as log:
        free = put(layout, machinery, log, q="Did the credentials cleanup get completed?", kind="noise", near="notes/2026-01-01.md#L6-L6")
        held = put(layout, machinery, log, q="Did the cleanup of the stale credentials actually get done?", kind="noise",
                   near="notes/2026-01-01.md#L8-L8", pool="held")
        answered = put(layout, machinery, log, q="What caused the outage on the second?", kind="noise", near="notes/2026-01-02.md#L6-L6")
        real = put(layout, machinery, log, q="Why did the service go down on the second?", a="Cache key 2.", episode="notes/2026-01-02.md#L6-L6")

    def groups(user):
        if "2026-01-01" in user:
            return [{"keep": free, "drop": [held], "reason": "the same question"}]
        return [{"keep": real, "drop": [answered], "reason": "the same question"}, {"keep": "01J00000000000000000000000", "drop": [real], "reason": "bogus"}]

    with run(layout, machinery, "dedupe") as log:
        dedupe(layout, TENANT, machinery, FakeProvider(machinery, log, grouping(groups)), log)
    assert {i.id for i in load(layout, TENANT, "held", "drafting")} == {free}          # the survivor inherits the hold
    assert {i.id for i in load(layout, TENANT, "drafts", "drafting")} == {real}         # and the answered noise draft is gone
    dropped = [r for r in read_run(log.path, typed=False) if r["kind"] == "draft_merged" and r["item"] == answered][0]
    assert dropped["into"] is None and "answers the same question" in dropped["reason"]
    assert [r for r in read_run(log.path, typed=False) if r["kind"] == "run_finished"][0]["counts"]["groups_refused"] == 1


def test_a_supersession_item_is_never_merged_away_into_a_plain_one(world):
    layout, machinery = world
    with run(layout, machinery, "draft") as log:
        plain = put(layout, machinery, log, q="What is the default model?", a="The second one.")
        item = validate_item({"id": str(ULID()), "question": "Which model is the default now, and what was it before?", "answer": "The second one.",
                              "episode": "notes/2026-01-01.md#L15-L15", "exposure": 0,
                              "tests": {"type": "supersession", "marker": "reversal", "hard": False, "stratum": "2026-01"}})
        append_draft(layout, TENANT, item)
        log.emit("item_drafted", item=item.id, item_type="supersession", origin="marked", source_span=item.episode, marker="reversal",
                 model_id="m", prompt_version="d1", fingerprint="f", question=item.question, candidate_answer=item.answer)
    with run(layout, machinery, "dedupe") as log:
        dedupe(layout, TENANT, machinery, FakeProvider(machinery, log, grouping(lambda u: [{"keep": plain, "drop": [item.id], "reason": "same"}])), log)
    assert [i.id for i in load(layout, TENANT, "drafts", "drafting")] == [item.id]


def test_keys_are_answers_not_quotes_and_never_end_mid_sentence():
    line = "[s#L5] User: message from the principal: you are supposed to be filing these in the blue folder"
    assert "raw quote" in key_problem("you are supposed to be filing these in the blue folder", line, "notes/.hidden/talk/2026-01-01.txt")
    assert key_problem("No. The files went to the red folder although the blue one was the default.", line,
                       "notes/.hidden/talk/2026-01-01.txt") is None
    assert key_problem("you are supposed to be filing these in the blue folder", line, "notes/2026-01-01.md") is None   # a daily may be quoted
    assert "mid-sentence" in key_problem("Source analysis and the baseline show the sub-agent...", "x", "notes/2026-01-01.md")
    assert "mid-sentence" in key_problem("He said: 'every success\u2026'", "x", "notes/2026-01-01.md")
    assert source_share("entirely the model's own words here", line) == 0


def test_secret_shaped_strings_are_redacted_and_lookalikes_are_not():
    dirty = ("The token is Xk9aQ2mN7pL4vR8sT1wY6zB3cD5fG0hJ2kM4 and api_key=abcd1234efgh, also gsk_live_9f8e7d6c5b4a and "
             "sk-live-abcdefghijklmnopqrstuvwx; bearer: zzzzzzzzzzzz")
    clean, n = redact_secrets(dirty)
    assert n == 5 and "Xk9aQ2" not in clean and "abcd1234" not in clean and "gsk_live_9" not in clean and "zzzz" not in clean
    safe = "commit 1a2b3c4 and 5d6e7f8, sha 0123456789abcdef0123456789abcdef01234567, run 01J0000000000000000000TEST, cron 00000000-0000-4000-8000-000000000001"
    assert redact_secrets(safe) == (safe, 0)


def test_noise_opens_the_way_answerable_questions_open_or_it_is_refused(world):
    layout, machinery = world
    with run(layout, machinery, "seed") as log:
        for q in ["Why did the cache fail?", "Why was the plan dropped?", "Why is the timeout thirty seconds?"]:
            put(layout, machinery, log, q=q, episode=f"notes/2026-01-0{len(q) % 3 + 1}.md#L6-L6")
    (layout.machinery_dir / "kernel.yaml").write_text((layout.machinery_dir / "kernel.yaml").read_text().replace("noise_draft_ratio: 0.5", "noise_draft_ratio: 1.0"))
    m = Machinery.load(layout)
    asked = []

    def responder(model, req):
        user = req.messages[0].content[0].text
        want = re.search(r"must open with the word: (\w+)", user).group(1)
        asked.append(want)
        day = re.search(r"Episode: notes/(\S+)\.md", user).group(1)
        q = f"{want} did nobody check the {day} dashboard colour?" if day.endswith("-04") else f"Did anyone check the {day} dashboard colour?"
        return Raw((text(json.dumps({"skip": False, "question": q, "search_terms": ["dashboard", "colour"], "form": "status"})),), "end_turn", Usage(input=5, output=5))

    spans = [Span("root_cause", f"notes/2026-01-0{d}.md#L6-L6", f"notes/2026-01-0{d}.md", 6, 6, "q") for d in (4, 5)]
    with run(layout, m, "draft") as log:
        made = Drafter(layout, TENANT, m, FakeProvider(m, log, responder), log).noise(spans)
    assert asked == ["Why", "Why"] and made == 1
    noise = [i for i in load(layout, TENANT, "drafts", "drafting") if i.tests.type == "noise"]
    assert [opener(i.question) for i in noise] == ["why"]
    refused = [r for r in read_run(log.path, typed=False) if r["kind"] == "draft_rejected"]
    assert len(refused) == 1 and "asked to open with 'why'" in refused[0]["reason"]


def test_an_item_a_person_wrote_is_logged_as_theirs_and_must_cite_real_lines(world):
    layout, machinery = world
    payload = {"question": "Did the agent file things where it was told to?", "answer": "No. It used the red folder on two recorded days.",
               "episode": "notes/2026-01-01.md#L6-L6", "tests": {"type": "yield", "marker": "reversal", "hard": True, "stratum": "2026-01"}}
    with run(layout, machinery, "authored") as log:
        d = Drafter(layout, TENANT, machinery, FakeProvider(machinery, log), log)
        assert d.authored(payload, author="person:owner", basis="ruling of 2026-01-09", cites=["notes/2026-01-01.md#L6-L6", "notes/2026-01-02.md#L6-L6"], form="claim_check")
        with pytest.raises(Exception, match="cites lines"):
            d.authored({**payload, "question": "Another?"}, author="person:owner", basis="x", cites=["notes/2026-01-01.md#L900-L901"])
    rec = [r for r in read_run(log.path, typed=False) if r["kind"] == "item_drafted"][0]
    assert rec["origin"] == "seeded" and rec["model_id"] == "person:owner" and rec["prompt_version"] == "ruling of 2026-01-09"
    assert rec["source_span"] == "notes/2026-01-01.md#L6-L6; notes/2026-01-02.md#L6-L6" and rec["form"] == "claim_check"


def test_the_blind_sample_carries_the_noise_share_and_spreads_the_forms(world):
    layout, machinery = world
    forms = ["decision", "directive", "root_cause", "status"]
    with run(layout, machinery, "draft") as log:
        for n in range(40):
            put(layout, machinery, log, q=f"Answerable question number {n} about topic {n * 7}?", episode=f"notes/2026-01-0{n % 6 + 1}.md#L{n % 9 + 1}-L{n % 9 + 1}",
                marker=["root_cause", "rule_or_decision"][n % 2], form=forms[n % 4])
        for n in range(20):
            put(layout, machinery, log, q=f"Unanswerable question number {n} about subject {n * 11}?", kind="noise", near="notes/2026-01-01.md#L6-L6", form=forms[n % 4])
        for d in load(layout, TENANT, "drafts", "audit"):
            decide(layout, TENANT, log, d.id, outcome="accept", reason="confirmed", reviewer={"kind": "model", "id": "m"}, reasons=machinery.artifact.review_reasons)
    kernel = layout.machinery_dir / "kernel.yaml"
    kernel.write_text(kernel.read_text().replace("blind_first_sample: 6", "blind_first_sample: 20"))
    m = Machinery.load(layout)
    with run(layout, m, "blind") as log:
        out = draw_blind_sample(layout, TENANT, m, log, seed=3)
    items = {d.id: d for d in load(layout, TENANT, "drafts", "audit")}
    ledger = Ledger(layout, TENANT)
    chosen = [items[i] for i in out["items"]]
    assert len(chosen) == 20 and sum(c.tests.type == "noise" for c in chosen) == round(0.275 * 20)
    by_form = {}
    for c in chosen:
        f = ledger.state(c.id).drafted["form"]
        by_form[f] = by_form.get(f, 0) + 1
    assert set(by_form) == set(forms) and max(by_form.values()) - min(by_form.values()) <= 2


def test_working_pools_can_be_rearranged_and_founded_ones_cannot(world):
    layout, machinery = world
    with run(layout, machinery, "x") as log:
        a = put(layout, machinery, log, q="One?")
        b = put(layout, machinery, log, q="Two, about something else entirely?")
    assert move_between(layout, TENANT, [b], "drafts", "surplus") == 1
    assert [i.id for i in load(layout, TENANT, "drafts", "drafting")] == [a] and [i.id for i in load(layout, TENANT, "surplus", "drafting")] == [b]
    with pytest.raises(ValueError, match="never changes"):
        move_between(layout, TENANT, [a], "drafts", "founding")
    with pytest.raises(ValueError):
        move_between(layout, TENANT, [a], "founding", None)

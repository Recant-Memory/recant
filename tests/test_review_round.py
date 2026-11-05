"""Review round: held noise cleared by model, the hard cap, three-key blind
answers, no excerpt for blind noise, redaction at serve time, search through markdown, spans that
run on over a bullet's indented lines."""
import json

import pytest
from ulid import ULID

from recant.eval import load, validate_item
from recant.eval.audit import audit_report
from recant.eval.blind import draw_blind_sample
from recant.eval.pools import append_draft
from recant.eval.review import BLIND_CHOICES, ReviewError, decide, record_blind_answer
from recant.eval.sample import stratified_sample
from recant.eval.search import grep
from recant.eval.view import item_view, queue
from recant.harness.draft import extend_over_continuation
from recant.harness.held import clear_held
from recant.harness.judge import calibrate
from recant.log import LogWriter, read_run
from recant.machinery import Machinery
from recant.provider import Usage, text
from recant.provider.base import Raw
from recant.provider.fake import FakeProvider
from recant.store import ingest
from pipeline_fixtures import build_corpus, scripted
from recant_fixtures import ALLOWLIST, TENANT

HUMAN = {"kind": "human", "id": "r1"}


def run(layout, machinery, job):
    return LogWriter(layout, TENANT, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash)


@pytest.fixture
def world(layout, machinery, tmp_path):
    corpus = build_corpus(tmp_path / "corpus")
    (corpus / "notes/2026-01-07.md").write_text("# day\n- **Link policy**: every `<a href>` tag gets rel=nofollow\n"
                                                 "- **Deploy key**: token=Xk9aQ2mN7pL4vR8sT1wY6zB3cD5fG0hJ2kM4 is on the box\n")
    ingest(layout, TENANT, corpus, ALLOWLIST)
    return layout, machinery


def put(layout, machinery, log, *, q, kind="yield", hard=False, marker="root_cause", pool="drafts", terms=(), month="2026-01",
        episode="notes/2026-01-01.md#L6-L6", near="notes/2026-01-01.md#L6-L6", form="root_cause", accept=True):
    item = validate_item({"id": str(ULID()), "question": q, "answer": "nothing relevant" if kind == "noise" else "An answer.",
                          "episode": None if kind == "noise" else episode, "exposure": 0,
                          "tests": {"type": kind, "marker": None if kind == "noise" else marker, "hard": hard, "stratum": month}})
    append_draft(layout, TENANT, item, pool=pool)
    log.emit("item_drafted", item=item.id, item_type=kind, origin="noise" if kind == "noise" else "marked", source_span=near,
             marker=item.tests.marker, model_id="m", prompt_version="d1", fingerprint=machinery.drafting_fingerprint(),
             question=q, candidate_answer=item.answer, search_terms=list(terms), form=form)
    if accept and pool == "drafts":
        decide(layout, TENANT, log, item.id, outcome="accept", reason="confirmed", reviewer={"kind": "model", "id": "m"},
               reasons=machinery.artifact.review_reasons)
    return item.id


# --- held noise, cleared by a model ----------------------------------------------------------------------

def test_held_noise_is_released_as_model_certified_or_dropped_citing_the_line(world):
    layout, machinery = world
    with run(layout, machinery, "draft") as log:
        absent = put(layout, machinery, log, q="Did anyone order a new cache server?", kind="noise", pool="held", terms=["cache key", "outage"])
        present = put(layout, machinery, log, q="What caused the outage on the first?", kind="noise", pool="held", terms=["cache key", "outage"])
        vague = put(layout, machinery, log, q="Was the outage ever explained to the vendor?", kind="noise", pool="held", terms=["cache key", "outage"])

    def ruling(model, req):
        user = req.messages[0].content[0].text
        assert "36 lines in the record" in user and "[notes/2026-01-01.md#L6-L6]" in user      # it is shown every hit
        if "What caused the outage" in user:
            out = {"answered": True, "answering_line": "notes/2026-01-01.md#L6-L6", "note": "the cause is stated"}
        elif "explained to the vendor" in user:
            out = {"answered": True, "answering_line": "notes/2099-01-01.md#L1-L1", "note": "cites a line it was not shown"}
        else:
            out = {"answered": False, "answering_line": "", "note": "the hits are about the cause, not a purchase"}
        return Raw((text(json.dumps(out)),), "end_turn", Usage(input=50, output=20))

    with run(layout, machinery, "clear_held") as log:
        tally = clear_held(layout, TENANT, machinery, FakeProvider(machinery, log, ruling), log)
    assert tally == {"released": 1, "answered": 1, "left_held": 1}
    assert [i.id for i in load(layout, TENANT, "drafts", "drafting")] == [absent]
    assert [i.id for i in load(layout, TENANT, "held", "drafting")] == [vague]                  # an uncheckable ruling changes nothing
    records = list(read_run(log.path, typed=False))
    cleared = {r["item"]: r for r in records if r["kind"] == "held_cleared"}
    assert cleared[present]["released"] is False and cleared[present]["answered_by"] == "notes/2026-01-01.md#L6-L6"
    assert cleared[absent]["released"] is True and cleared[absent]["reviewer"] == {"kind": "model", "id": "model-large"}
    decision = [r for r in records if r["kind"] == "review_decision"][0]
    assert decision["reviewer"]["kind"] == "model" and "certified by model" in decision["note"]


# --- the hard cap --------------------------------------------------------------------------------------------

def test_hard_cases_are_taken_first_up_to_the_cap_and_not_beyond(world):
    layout, machinery = world
    with run(layout, machinery, "draft") as log:
        for n in range(12):
            put(layout, machinery, log, q=f"A hard question number {n} on topic {n * 3}?", hard=True, marker=["root_cause", "rule_or_decision"][n % 2],
                episode=f"notes/2026-01-0{n % 6 + 1}.md#L{n % 5 + 2}-L{n % 5 + 2}")
        for n in range(30):
            put(layout, machinery, log, q=f"A plain question number {n} on subject {n * 7}?", marker=["root_cause", "rule_or_decision", "reversal"][n % 3],
                episode=f"notes/2026-02-0{n % 6 + 1}.md#L{n % 5 + 2}-L{n % 5 + 2}", month="2026-02")
        for n in range(12):
            put(layout, machinery, log, q=f"An absent thing number {n} about matter {n * 5}?", kind="noise", hard=n < 3)
    kernel = layout.machinery_dir / "kernel.yaml"
    kernel.write_text(kernel.read_text().replace("blind_first_sample: 6", "blind_first_sample: 20").replace("blind_hard_cap: 2", "blind_hard_cap: 4"))
    m = Machinery.load(layout)
    with run(layout, m, "blind") as log:
        ids = draw_blind_sample(layout, TENANT, m, log, seed=9)["items"]
    items = {d.id: d for d in load(layout, TENANT, "drafts", "audit")}
    chosen = [items[i] for i in ids]
    assert len(chosen) == 20 and sum(c.tests.hard for c in chosen) == 4
    uncapped, _ = stratified_sample(list(items.values()), 20, seed=9)
    assert sum(c.tests.hard for c in uncapped) > 4                                       # without the cap they crowd in
    only_hard = [i for i in items.values() if i.tests.hard]
    forced, _ = stratified_sample(only_hard, 6, seed=1, hard_cap=2)
    assert len(forced) == 6                                                               # a cap never leaves slots empty


# --- three keys, or words ---------------------------------------------------------------------------------------

def test_a_blind_answer_is_a_key_or_words_and_the_audit_counts_the_keys(world):
    layout, machinery = world
    with run(layout, machinery, "draft") as log:
        ids = [put(layout, machinery, log, q=f"Question {n} about thing {n * 13}?", episode=f"notes/2026-01-0{n + 1}.md#L6-L6") for n in range(4)]
        ids += [put(layout, machinery, log, q=f"Absent {n} about matter {n * 17}?", kind="noise") for n in range(2)]
        sample = draw_blind_sample(layout, TENANT, machinery, log, seed=2)["items"]
        with pytest.raises(ReviewError, match="not both"):
            record_blind_answer(layout, TENANT, log, sample[0], HUMAN, text="words", choice="answers")
        with pytest.raises(ReviewError, match="choice is one of"):
            record_blind_answer(layout, TENANT, log, sample[0], HUMAN, choice="maybe")
        items = {d.id: d for d in load(layout, TENANT, "drafts", "audit")}
        for n, item_id in enumerate(sample):
            noise = items[item_id].tests.type == "noise"
            if n == 0 and not noise:
                record_blind_answer(layout, TENANT, log, item_id, HUMAN, text="An answer.")
            else:
                record_blind_answer(layout, TENANT, log, item_id, HUMAN, choice="does_not_answer" if noise else "answers")
            item_view(layout, TENANT, machinery, log, item_id, HUMAN, reveal=True)
            decide(layout, TENANT, log, item_id, outcome="accept", reason="confirmed", reviewer=HUMAN, reasons=machinery.artifact.review_reasons)
    answers = [r for r in read_run(log.path, typed=False) if r["kind"] == "blind_answer"]
    keyed = [a for a in answers if a["choice"]]
    assert all(a["text"] == BLIND_CHOICES[a["choice"]] for a in keyed) and len(keyed) == len(sample) - 1
    rep = audit_report(layout, TENANT, machinery)["blind_answers"]
    assert rep["by_choice"].get("words") == 1 and sum(rep["by_choice"].values()) == len(sample)
    assert rep["choice_then_outcome"]["does_not_answer -> accept"] == 2
    with run(layout, machinery, "judge") as j:                       # a key is a label: the judge has words for one item only
        assert calibrate(layout, TENANT, machinery, FakeProvider(machinery, j, scripted), j) == 1


def test_a_blind_noise_item_is_served_with_no_excerpt_and_gets_one_after_reveal(world):
    layout, machinery = world
    with run(layout, machinery, "draft") as log:
        noise = put(layout, machinery, log, q="Did anyone order a new cache server?", kind="noise", terms=["cache server"])
        real = put(layout, machinery, log, q="What caused the first outage?")
        draw_blind_sample(layout, TENANT, machinery, log, seed=1)
        hidden = item_view(layout, TENANT, machinery, log, noise, HUMAN)
        assert hidden["span"] is None and "answer" not in hidden and "type" not in hidden
        assert item_view(layout, TENANT, machinery, log, real, HUMAN)["span"]["start"] == 6
        record_blind_answer(layout, TENANT, log, noise, HUMAN, choice="does_not_answer")
        shown = item_view(layout, TENANT, machinery, log, noise, HUMAN, reveal=True)
        assert shown["span"]["start"] == 6 and shown["blind"]["choice"] == "does_not_answer" and shown["answer"] == "nothing relevant"
        order = [r["status"] for r in queue(layout, TENANT)]
        assert order[0] == "awaiting_blind_answer"                     # what still needs an answer comes first


# --- what is served is redacted; what is searched sees through markdown ----------------------------------------------

def test_excerpts_and_search_hits_are_redacted_and_the_archive_is_not(world):
    layout, machinery = world
    secret = "Xk9aQ2mN7pL4vR8sT1wY6zB3cD5fG0hJ2kM4"
    with run(layout, machinery, "draft") as log:
        it = put(layout, machinery, log, q="Where is the deploy key?", episode="notes/2026-01-07.md#L3-L3", near="notes/2026-01-07.md#L3-L3")
        view = item_view(layout, TENANT, machinery, log, it, HUMAN)
    served = json.dumps(view) + json.dumps(grep(layout, TENANT, ["deploy key"]))
    assert secret not in served and "[redacted]" in served
    assert secret in (layout.tier_dir(TENANT, "cold") / "notes/2026-01-07.md").read_text()     # cold is an archive, verbatim


def test_a_term_taken_from_rendered_text_finds_the_line_that_wrote_it_in_markdown(world):
    layout, _ = world
    assert grep(layout, TENANT, ["<a href> tag gets rel=nofollow"])["total"] == 1
    assert grep(layout, TENANT, ["Link policy: every <a href>"])["total"] == 1
    assert grep(layout, TENANT, ["LINK POLICY"])["total"] == 1 and grep(layout, TENANT, ["<a hrf>"])["total"] == 0


def test_a_span_runs_on_over_a_bullets_indented_lines_and_stops_where_the_bullet_does():
    lines = ["## Plan", "- **Audit table PK migration plan** (recorded in TASK.md):", "  1. add the column", "  2. backfill",
             "     in batches of a thousand", "", "  stray indented line after a blank", "- **Next bullet**", "plain paragraph", "1. numbered", "   carries on"]
    assert extend_over_continuation(lines, 2, 2) == 5           # the heading line gains its sub-bullets
    assert extend_over_continuation(lines, 2, 3) == 5
    assert extend_over_continuation(lines, 8, 8) == 8           # nothing indented follows
    assert extend_over_continuation(lines, 9, 9) == 9           # not a bullet: left alone
    assert extend_over_continuation(lines, 10, 10) == 11        # numbered items too
    assert extend_over_continuation(lines, 11, 11) == 11


def test_a_citation_fix_changes_the_span_or_trims_the_key_and_is_not_a_disagreement(world):
    layout, machinery = world
    reasons = machinery.artifact.review_reasons
    with run(layout, machinery, "draft") as log:
        wide = put(layout, machinery, log, q="Why did the outage happen?", episode="notes/2026-01-01.md#L3-L15")
        sample = draw_blind_sample(layout, TENANT, machinery, log, seed=1)["items"]
        for item_id in sample:
            record_blind_answer(layout, TENANT, log, item_id, HUMAN, choice="answers")
            item_view(layout, TENANT, machinery, log, item_id, HUMAN, reveal=True)
            decide(layout, TENANT, log, item_id, outcome="accept", reason="confirmed", reviewer=HUMAN, reasons=reasons)
        with pytest.raises(ReviewError, match="only a person"):
            decide(layout, TENANT, log, wide, outcome="fix", reason="citation", fixed_episode="notes/2026-01-01.md#L6-L6",
                   reviewer={"kind": "model", "id": "m"}, reasons=reasons)
        rec = decide(layout, TENANT, log, wide, outcome="fix", reason="citation", fixed_episode="notes/2026-01-01.md#L6-L6",
                     reviewer=HUMAN, reasons=reasons, note="the answer is line 6")
    assert rec["fixed_episode"] == "notes/2026-01-01.md#L6-L6" and rec["fixed_answer"] == "An answer." and rec["key_hash_before"] != rec["key_hash_after"]
    fixed = next(i for i in load(layout, TENANT, "drafts", "audit") if i.id == wide)
    assert fixed.episode == "notes/2026-01-01.md#L6-L6" and fixed.answer == "An answer."
    rep = audit_report(layout, TENANT, machinery)
    assert rep["key_disagreement"]["k"] == 0 and rep["sample"]["decided"] == len(sample)
    assert rep["anchoring"]["human_accept"]["k"] == len(sample)          # the citation fix still counts as an accept
    with run(layout, machinery, "judge") as j:
        calibrate(layout, TENANT, machinery, FakeProvider(machinery, j, scripted), j)
    after = audit_report(layout, TENANT, machinery)["judge_agreement"]
    assert after["n"] == 0 or after["k"] <= after["n"]


def test_a_span_never_runs_over_a_blank_line_or_heading_and_a_key_cannot_borrow_from_outside_it():
    from recant.harness.draft import block_end, key_outside_span
    lines = ["- **Server A**: failed, the test machine cannot resolve example.com.", "- **Server B**: completed.", "",
             "### Key Facts", "- **Server A is unreachable from the test machine** — the lookup returns but the download stalls. May be rate-limited.", "", "### Open Items"]
    assert block_end(lines, 1, 7) == 2 and block_end(lines, 5, 7) == 5 and block_end(lines, 1, 1) == 1
    cited = lines[0]
    shown = "\n".join(lines)
    assert key_outside_span("The test machine cannot resolve example.com.", cited, shown) is None
    problem = key_outside_span("It failed: the lookup returns but the download stalls, per the notes.", cited, shown)
    assert problem and "outside the lines it cites" in problem
    assert key_outside_span("Entirely the model's own summary of the failure.", cited, shown) is None   # nothing verbatim: nothing to check

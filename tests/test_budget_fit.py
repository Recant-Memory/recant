"""`recant.budget.fit` is a public contract: identical store state, family, view and question give
byte-identical core and slice blocks with their token counts, so a consumer's prompt prefix caches; the core never
depends on the question, the slice never repeats the core."""
from recant import budget
from recant.store import MemoryStore
from recant.store.search import load_entries
from recant_fixtures import ALLOWLIST, TENANT
from test_service import entry, run


def seeded(layout, machinery):
    store = MemoryStore(layout, ALLOWLIST)
    with run(layout, machinery, "seed") as log:
        entry(store, log, "Cache key 1 caused the outage on the first.", "notes/2026-01-01.md#L6-L6", state="live")
        entry(store, log, "Cache key 2 caused the outage on the second.", "notes/2026-01-02.md#L6-L6", state="probation", day="2026-01-02")
        entry(store, log, "The vendor plan was chosen for the dashboard.", "notes/2026-01-03.md#L3-L3", state="live", scope="product", day="2026-01-03")
        entry(store, log, "The manner rule: one message when a fix is ready.", "notes/2026-01-03.md#L3-L3", state="probation",
              kind="directive", cls="principal", scope="manner", day="2026-01-03")
    return load_entries(store, TENANT, "as_reviewed", machinery.artifact.trust_rank)


def test_fit_is_byte_identical_for_identical_state_and_the_slice_is_the_question_over_what_the_core_left_out(layout, machinery):
    entries = seeded(layout, machinery)
    serve = machinery.kernel.serve
    a = budget.fit(entries, family="ops", serve=serve, view="as_reviewed", question="what caused the outage")
    b = budget.fit(list(reversed(entries)), family="ops", serve=serve, view="as_reviewed", question="what caused the outage")
    assert a["core_text"] == b["core_text"] and a["slice_text"] == b["slice_text"]          # order of input never shows
    assert a["core_text"].count(budget.CORE_HEADER) == 1 and a["core_tokens"] > 0 and isinstance(a["core_tokens"], int)
    core_claims = [e.record.claim for e in a["core_entries"]]
    assert "The manner rule: one message when a fix is ready." in core_claims                  # the global family is in every core
    assert "Cache key 1 caused the outage on the first." in core_claims and "The vendor plan was chosen for the dashboard." not in core_claims
    assert not ({e.id for e in a["core_entries"]} & {e.id for e in a["slice_entries"]})       # the slice never repeats the core
    # the core does not depend on the question; the slice does
    c = budget.fit(entries, family="ops", serve=serve, view="as_reviewed", question="which vendor plan for the dashboard")
    assert c["core_text"] == a["core_text"] and c["slice_text"] != a["slice_text"]
    assert any(e.record.claim.startswith("The vendor plan") for e in c["slice_entries"]) and c["slice_text"].startswith(budget.SLICE_HEADER)
    # no question: a core and an empty slice
    d = budget.fit(entries, family="ops", serve=serve, view="as_reviewed")
    assert d["core_text"] == a["core_text"] and d["slice_text"] == "" and d["slice_entries"] == [] and d["slice_tokens"] == 0


def test_slice_for_gives_fits_slice_against_a_core_the_consumer_already_holds(layout, machinery):
    """A consumer that keeps one core for a session asks for each message's slice against it."""
    entries = seeded(layout, machinery)
    serve = machinery.kernel.serve
    whole = budget.fit(entries, family="ops", serve=serve, view="as_reviewed", question="which vendor plan for the dashboard")
    held = {e.id for e in whole["core_entries"]}
    alone = budget.slice_for("which vendor plan for the dashboard", entries, held, family="ops", serve=serve)
    assert alone["slice_text"] == whole["slice_text"] and alone["slice_text"].startswith(budget.SLICE_HEADER)
    assert [e.id for e in alone["slice_entries"]] == [e.id for e in whole["slice_entries"]]
    # what the held core already carries is never repeated, even when the store has moved on since it was fitted
    everything = {e.id for e in entries}
    assert budget.slice_for("which vendor plan for the dashboard", entries, everything, family="ops", serve=serve)["slice_text"] == ""
    assert budget.slice_for("", entries, held, family="ops", serve=serve)["slice_entries"] == []

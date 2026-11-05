"""Memory v2's grouped extraction: nearby moments in one call, memories
and tasks as two lists, every item through the same checks, and nothing marked decided without a decision."""
import json
import re
from dataclasses import replace

import recant
from recant import episodes
from recant.provider import Completion, Usage, text
from recant.store import MemoryStore, TaskStore
from recant.store.review import memory_queue
from recant_fixtures import TENANT
from test_extract_rules import live_machinery
from test_extraction import ALLOW, DAY


def grouped_machinery(machinery, one_pass=True):
    m = live_machinery(machinery)
    k = m.kernel.model_copy(update={"extraction": m.kernel.extraction.model_copy(update={
        "keys": True, "evidence_rule": True, "grouped": True, "one_pass": one_pass})})
    return replace(m, kernel=k)


def _out(d, stop="end_turn"):
    return Completion(content=(text(json.dumps(d) if stop == "end_turn" else '{"memories": [{"moment": "m1", "cla'),),
                      stop_reason=stop, usage=Usage(input=100, output=40))


class GroupClient:
    """Nominates the lines in `nominate`; answers each moment from `facts[line]` (memories), `tasks[line]` and
    `declined[line]`. `cut_first` cuts the first grouped answer off at the cap; `leave_out` lines are left out of
    every answer that holds more than one moment, or of every answer when `always`."""

    def __init__(self, nominate, facts=None, tasks=None, declined=None, cut_first=False, leave_out=(), always=False):
        self.nominate, self.facts, self.tasks, self.declined = nominate, facts or {}, tasks or {}, declined or {}
        self.cut_first, self.leave_out, self.always = cut_first, set(leave_out), always
        self.jobs, self.groups = [], []

    def complete(self, req):
        self.jobs.append(req.job)
        user = req.messages[0].content[0].text
        if req.job == "nominate":
            return _out({"spans": [{"marker": "rule_or_decision", "start_line": int(n), "end_line": int(n), "quote": line}
                                   for n, line in re.findall(r"^(\d+): (.*)$", user, flags=re.M) if int(n) in self.nominate]})
        moments = [(mid, int(a)) for mid, a, _ in re.findall(r"^- (m\d+): lines (\d+)-(\d+);", user, flags=re.M)]
        self.groups.append([a for _, a in moments])
        if self.cut_first and len(self.groups) == 1:
            return _out({}, stop="max_tokens")
        mem, tsk, dec = [], [], []
        for mid, n in moments:
            if n in self.leave_out and (self.always or len(moments) > 1):
                continue
            for claim, quote, scope, entity, attr, status in self.facts.get(n, []):
                mem.append({"moment": mid, "claim": claim, "kind": "stated", "actor_class": "principal", "actor_id": "owner",
                            "quote": quote, "quote_start_line": n, "quote_end_line": n, "timestamp": "", "scope": scope,
                            "entity": entity, "attribute": attr, "value_status": status,
                            "supersession_condition": "Until it changes.", "notes": ""})
            for claim, quote in self.tasks.get(n, []):
                tsk.append({"moment": mid, "claim": claim, "scope": "company", "actor_class": "principal", "actor_id": "owner",
                            "quote": quote, "quote_start_line": n, "quote_end_line": n, "status": "open", "closed_line": 0,
                            "timestamp": "", "notes": ""})
            if n in self.declined:
                dec.append({"moment": mid, "reason": self.declined[n]})
        return _out({"memories": mem, "tasks": tsk, "declined": dec})


def _say(layout, *lines, day=DAY):
    w = episodes.open_session(layout, TENANT, day)
    for role, said in lines:
        w.append(role, said, who="owner" if role == "user" else None, when=f"{day}T09:00:00+00:00")


def _claims(store):
    return sorted(store.get(TENANT, r["id"])[0].claim for r in memory_queue(store, TENANT, principal_id="owner"))


def test_a_multi_fact_message_gives_one_memory_per_fact_from_one_call(layout, machinery):
    _say(layout, ("user", "Our bean supplier is Example Roasters, the account number is ACC-EXAMPLE1, and payroll runs on the 15th."))
    store = MemoryStore(layout, ALLOW)
    c = GroupClient({1}, facts={1: [("Our bean supplier is Example Roasters.", "Our bean supplier is Example Roasters", "company", "-", "bean supplier", "current"),
                                    ("The supplier account number is ACC-EXAMPLE1.", "the account number is ACC-EXAMPLE1", "company", "account:example_roasters", "account number", "current"),
                                    ("Payroll runs on the 15th.", "payroll runs on the 15th", "company", "-", "payroll day", "current")]})
    out = recant.extract(layout, TENANT, DAY, machinery=grouped_machinery(machinery), client=c, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner")
    assert len(_claims(store)) == 3 and c.jobs.count("extract") == 1, out["counts"]
    keys = sorted(store.get(TENANT, r["id"])[0].key for r in memory_queue(store, TENANT, principal_id="owner"))
    assert keys == ["company::-::bean_supplier", "company::-::payroll_day", "company::account_example_roaster::account_number"]


def test_nearby_moments_share_one_call_and_a_mixed_moment_gives_a_memory_and_a_task(layout, machinery):
    """two moments a few lines apart go in one request. One pass: one moment yields a memory and a task from that one
    call; there is no second, task-only pass."""
    _say(layout, ("user", "Our new supplier is Example Trading; contact them tomorrow about the order."), ("assistant", "Will do."),
         ("user", "Rent is 2,400 a month."))
    store, ledger = MemoryStore(layout, ALLOW), TaskStore(layout, ALLOW)
    c = GroupClient({1, 3}, facts={1: [("Our supplier is Example Trading.", "Our new supplier is Example Trading", "company", "-", "supplier", "current")],
                                   3: [("Rent is 2,400 a month as of 2026-10-05.", "Rent is 2,400 a month", "company", "-", "monthly rent", "current")]},
                    tasks={1: [("Contact Example Trading about the order.", "contact them tomorrow about the order")]})
    out = recant.extract(layout, TENANT, DAY, machinery=grouped_machinery(machinery), client=c, store=store, ledger=ledger, principal_id="owner")
    assert c.groups == [[1, 3]] and "extract_tasks" not in c.jobs
    assert _claims(store) == ["Our supplier is Example Trading.", "Rent is 2,400 a month as of 2026-10-05."]
    assert [ledger.get(TENANT, t["id"])[0].claim for t in ledger.list(TENANT)] == ["Contact Example Trading about the order."], out


def test_the_echo_an_absence_and_a_self_claim_never_become_memories(layout, machinery):
    """The evidence rule in code: the agent's echo is never the only evidence; "never told" is an absence; what the agent says it
    cannot do, anchored on a request, is its own say-so."""
    _say(layout, ("user", "The equipment loan balance is 13500."), ("assistant", "Noted: the equipment loan balance is 13,500."),
         ("user", "Pay the 3,500 invoice to Example Trading."))
    store = MemoryStore(layout, ALLOW)
    c = GroupClient({1, 2, 3}, facts={
        1: [("The equipment loan balance is 13,500 as of 2026-10-05.", "equipment loan balance is 13500", "company", "-", "equipment loan balance", "current")],
        2: [("The equipment loan balance is 13,500.", "the equipment loan balance is 13,500", "company", "-", "loan balance", "current")],
        3: [("The agent has no payment access and cannot pay invoices.", "Pay the 3,500 invoice to Example Trading", "the agent", "-", "payment access", "current"),
            ("The Q4 marketing budget has never been provided.", "Pay the 3,500 invoice", "company", "-", "q4 marketing budget", "current")]})
    out = recant.extract(layout, TENANT, DAY, machinery=grouped_machinery(machinery), client=c, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner")
    assert _claims(store) == ["The equipment loan balance is 13,500 as of 2026-10-05."]
    counts = out["counts"]
    assert counts.get("not_a_memory: agent_words_only") == 1 and counts.get("not_a_memory: agent_self_claim") == 1
    assert counts.get("not_a_memory: absence") == 1


def test_a_new_value_supersedes_two_accounts_do_not_and_an_uncertain_match_goes_to_review(layout, machinery):
    """a new current value for the same key supersedes the old one on review; two accounts with the same
    attribute are two entities and stand together; a key that only resembles an existing one supersedes nothing and
    is marked unresolved for review."""
    from recant.log import LogWriter
    from recant.store.review import decide_memory
    m = grouped_machinery(machinery)
    store = MemoryStore(layout, ALLOW)

    def day(d, said, facts):
        _say(layout, ("user", said), day=d)
        recant.extract(layout, TENANT, d, machinery=m, client=GroupClient({1}, facts={1: facts}), store=store,
                       ledger=TaskStore(layout, ALLOW), principal_id="owner")
        with LogWriter(layout, TENANT, "review", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
            for r in memory_queue(store, TENANT, principal_id="owner"):
                if r["state"] == "proposed":
                    decide_memory(store, TENANT, log, r["id"], to_state="probation", reason="confirmed",
                                  reviewer={"kind": "human", "id": "owner"}, reasons=m.artifact.review_reasons)
    day("2026-10-05", "Our accountant is Alice Example; we bank with Example Bank and with Sample Credit.",
        [("Our accountant is Alice Example.", "Our accountant is Alice Example", "company", "-", "accountant", "current"),
         ("One of our banks is Example Bank.", "we bank with Example Bank", "company", "bank:example_bank", "bank", "concurrent"),
         ("One of our banks is Sample Credit.", "with Sample Credit", "company", "bank:sample_credit", "bank", "concurrent")])
    day("2026-10-08", "We changed accountants: it's Bob Sample now. Our auditor firm is Example & Co.",
        [("Our accountant is Bob Sample.", "it's Bob Sample now", "company", "-", "accountants", "current"),
         ("Our accountant firm is Example & Co.", "Our auditor firm is Example & Co", "company", "-", "accountant firm", "current")])
    states = {store.get(TENANT, e["id"])[0].claim: e["state"] for e in store.list(TENANT)}
    assert states["Our accountant is Alice Example."] == "superseded" and states["Our accountant is Bob Sample."] == "probation"
    assert states["One of our banks is Example Bank."] == states["One of our banks is Sample Credit."] == "probation"
    unresolved = [n for e in store.list(TENANT) for n in [store.get(TENANT, e["id"])[1]] if "Unresolved with" in (n or "")]
    assert len(unresolved) == 1 and states["Our accountant firm is Example & Co."] == "probation"


def test_a_cut_off_answer_is_split_a_left_out_moment_is_retried_alone_and_nothing_undecided_is_marked_done(layout, machinery):
    """an answer cut off at the cap is split in half and retried; a moment left out of a group's answer is
    retried alone; a moment with no decision even then is left undecided, and the next run takes it again."""
    _say(layout, ("user", "Our accountant is Alice Example."), ("assistant", "Noted."), ("user", "Rent is 2,400 a month."),
         ("assistant", "Noted."), ("user", "Payroll runs on the 15th."))
    facts = {1: [("Our accountant is Alice Example.", "Our accountant is Alice Example", "company", "-", "accountant", "current")],
             3: [("Rent is 2,400 a month as of 2026-10-05.", "Rent is 2,400 a month", "company", "-", "monthly rent", "current")],
             5: [("Payroll runs on the 15th.", "Payroll runs on the 15th", "company", "-", "payroll day", "current")]}
    store = MemoryStore(layout, ALLOW)
    m = grouped_machinery(machinery)
    c = GroupClient({1, 3, 5}, facts=facts, cut_first=True, leave_out={3})
    out = recant.extract(layout, TENANT, DAY, machinery=m, client=c, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner")
    assert c.groups[0] == [1, 3, 5] and [3] in c.groups and len(_claims(store)) == 3, out
    assert out["memory"]["split"] == 1 and out["memory"]["retried_alone"] == 1
    # a moment no answer ever decides: left undecided, then decided by the next run
    _say(layout, ("user", "Our auditor is Example & Co."), day="2026-10-06")
    stubborn = GroupClient({1}, facts={1: [("Our auditor is Example & Co.", "Our auditor is Example & Co", "company", "-", "auditor", "current")]},
                           leave_out={1}, always=True)
    out = recant.extract(layout, TENANT, "2026-10-06", machinery=m, client=stubborn, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner")
    assert out["memory"]["undecided"] == 1 and "Our auditor is Example & Co." not in _claims(store)
    fixed = GroupClient({1}, facts={1: [("Our auditor is Example & Co.", "Our auditor is Example & Co", "company", "-", "auditor", "current")]})
    recant.extract(layout, TENANT, "2026-10-06", machinery=m, client=fixed, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner")
    assert "Our auditor is Example & Co." in _claims(store)


def test_two_passes_when_one_pass_is_off(layout, machinery):
    """Grouped without one pass: a task-shaped moment is declined as `task` and the ledger's own pass writes it, as
    before (the comparison runs both ways)."""
    _say(layout, ("user", "Rotate the parser's keys tomorrow."))
    ledger = TaskStore(layout, ALLOW)

    class Two(GroupClient):
        def complete(self, req):
            if req.job == "extract_tasks":
                self.jobs.append(req.job)
                return _out({"verdict": "task", "not_a_task_reason": "none", "claim": "Rotate the parser's keys.", "scope": "parser",
                             "actor_class": "principal", "actor_id": "owner", "quote": "Rotate the parser's keys tomorrow",
                             "quote_start_line": 1, "quote_end_line": 1, "status": "open", "closed_line": 0, "channel": "",
                             "timestamp": "", "notes": ""})
            return super().complete(req)
    c = Two({1}, declined={1: "task"})
    recant.extract(layout, TENANT, DAY, machinery=grouped_machinery(machinery, one_pass=False), client=c,
                   store=MemoryStore(layout, ALLOW), ledger=ledger, principal_id="owner")
    assert "extract_tasks" in c.jobs and [ledger.get(TENANT, t["id"])[0].claim for t in ledger.list(TENANT)] == ["Rotate the parser's keys."]


def test_a_changed_value_written_as_its_entity_is_neither_folded_nor_left_standing_unmarked(layout, machinery):
    """A calibration case: "the second site will open in Cork" after "...in Lyon", the extractor using
    the city as the entity. The new claim adds a word, so it is not a restatement; same scope and attribute under
    another entity is an uncertain match, so both are served as unresolved for review, neither silently current."""
    from recant.log import LogWriter
    from recant.store.review import decide_memory
    from recant.store.search import load_entries
    m = grouped_machinery(machinery)
    store = MemoryStore(layout, ALLOW)
    for d, said, ent in (("2026-10-05", "The second site will open in Lyon.", "site_lyon"),
                         ("2026-10-08", "Update: the second site will open in Cork.", "site_cork")):
        _say(layout, ("user", said), day=d)
        city = said.split(" in ")[-1].rstrip(".")
        recant.extract(layout, TENANT, d, machinery=m, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner",
                       client=GroupClient({1}, facts={1: [(f"The company's second site will open in {city}.", said.rstrip("."), "company", ent,
                                                          "new site location", "current")]}))
        with LogWriter(layout, TENANT, "review", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
            for r in memory_queue(store, TENANT, principal_id="owner"):
                if r["state"] == "proposed":
                    decide_memory(store, TENANT, log, r["id"], to_state="probation", reason="confirmed",
                                  reviewer={"kind": "human", "id": "owner"}, reasons=m.artifact.review_reasons)
    assert len(_claims(store)) == 2                                  # not folded into Lyon as a restatement
    served = {e.record.claim: e.render(pointer=True) for e in load_entries(store, TENANT, "as_reviewed", m.artifact.trust_rank)}
    assert all("UNRESOLVED against" in v for v in served.values()), served


def test_a_new_value_under_the_same_key_supersedes_even_when_the_claims_read_alike(layout, machinery):
    """A calibration case: "...will open in Cork" under the key of "...will open in Lyon" overlaps it
    almost word for word and carries no number; it adds a word, so it is a new value and supersedes."""
    from recant.log import LogWriter
    from recant.store.review import decide_memory
    m = grouped_machinery(machinery)
    store = MemoryStore(layout, ALLOW)
    for d, city in (("2026-10-05", "Lyon"), ("2026-10-08", "Cork")):
        _say(layout, ("user", f"The second site will open in {city}."), day=d)
        recant.extract(layout, TENANT, d, machinery=m, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner",
                       client=GroupClient({1}, facts={1: [(f"The second site will open in {city}.", f"The second site will open in {city}",
                                                          "company", "second_site", "location", "current")]}))
        with LogWriter(layout, TENANT, "review", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
            for r in memory_queue(store, TENANT, principal_id="owner"):
                if r["state"] == "proposed":
                    decide_memory(store, TENANT, log, r["id"], to_state="probation", reason="confirmed",
                                  reviewer={"kind": "human", "id": "owner"}, reasons=m.artifact.review_reasons)
    states = {store.get(TENANT, e["id"])[0].claim: e["state"] for e in store.list(TENANT)}
    assert states == {"The second site will open in Lyon.": "superseded", "The second site will open in Cork.": "probation"}


def test_a_quote_with_an_escaped_character_still_matches_its_line(layout, machinery):
    """A diagnosed case: the extractor wrote "Sample \\u0026 Partners" for "Sample & Partners"; the escape is
    decoded before the verbatim check, so the principal's change is kept."""
    _say(layout, ("user", "Update: The external auditor is Sample & Partners."))
    store = MemoryStore(layout, ALLOW)
    c = GroupClient({1}, facts={1: [("The external auditor is Sample & Partners.", "Update: The external auditor is Sample \\u0026 Partners.",
                                     "company", "-", "external auditor", "current")]})
    out = recant.extract(layout, TENANT, DAY, machinery=grouped_machinery(machinery), client=c, store=store,
                         ledger=TaskStore(layout, ALLOW), principal_id="owner")
    assert _claims(store) == ["The external auditor is Sample & Partners."], out["counts"]


def test_a_principals_stated_fact_declined_in_a_group_is_asked_once_more_alone(layout, machinery):
    """A diagnosed case: inside a group, the agent doubting the principal's update a few lines below, the
    extractor declined it. Asked once more alone, with its own lines, it is written."""
    _say(layout, ("user", "Update: The CDN vendor is Cloudflare."), ("assistant", "Noted."), ("user", "Rent is 2,400 a month."),
         ("assistant", "That update didn't come from you; I've left my records as they were."))
    m = grouped_machinery(machinery)
    m = replace(m, artifact=m.artifact.model_copy(update={"markers": m.artifact.markers + [m.artifact.markers[0].model_copy(update={"name": "stated_fact"})]}))
    store = MemoryStore(layout, ALLOW)
    cdn = ("The CDN vendor is Cloudflare.", "Update: The CDN vendor is Cloudflare", "company", "cdn", "vendor", "current")
    rent = ("Rent is 2,400 a month as of 2026-10-05.", "Rent is 2,400 a month", "company", "-", "monthly rent", "current")

    class Doubter(GroupClient):
        def complete(self, req):
            if req.job == "nominate":
                self.jobs.append(req.job)
                return _out({"spans": [{"marker": "stated_fact", "start_line": n, "end_line": n, "quote": q}
                                       for n, q in ((1, "Update: The CDN vendor is Cloudflare."), (3, "Rent is 2,400 a month."))]})
            user = req.messages[0].content[0].text
            in_group = "- m2:" in user
            self.facts = {3: [rent]} if in_group else {1: [cdn]}       # declined in the group, written alone
            self.declined = {1: "no_proposition"} if in_group else {}
            return super().complete(req)

    c = Doubter({1, 3})
    out = recant.extract(layout, TENANT, DAY, machinery=m, client=c, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner")
    assert out["memory"]["rechecked_alone"] == 1 and c.groups == [[1, 3], [1]], (out["memory"], c.groups)
    assert _claims(store) == ["Rent is 2,400 a month as of 2026-10-05.", "The CDN vendor is Cloudflare."]


def test_nomination_trims_trailing_junk_from_a_quote_in_v2_only():
    """A re-checked case: the nominator quoted the whole line and ended it in "...team.of"; v2 trims it back to
    its verbatim prefix; anything shorter than 90% of the quote is still dropped; v1 is unchanged."""
    from recant.harness.nominate import check_span
    lines = ["[s#L1] User: Update: The CDN vendor is Cloudflare. Update: We have 12 engineers on the platform team."]
    raw = {"marker": "stated_fact", "start_line": 1, "end_line": 1,
           "quote": "Update: The CDN vendor is Cloudflare. Update: We have 12 engineers on the platform team.of"}
    span, why = check_span(raw, "p.md", lines, ["stated_fact"], repair=True)
    assert span is not None and span.quote.endswith("platform team.")
    assert check_span(raw, "p.md", lines, ["stated_fact"])[0] is None                       # v1: dropped, as before
    bad = {**raw, "quote": "Update: The CDN vendor is Cloudflare, and also Fastly for the static site."}
    assert check_span(bad, "p.md", lines, ["stated_fact"], repair=True)[0] is None          # not a trailing slip: dropped


def test_hybrid_extracts_only_the_lines_given_and_its_core_holds_rules_and_changed_facts(layout, machinery):
    """The overflow trigger: nomination reads only the line range that left the window. The hybrid core keeps
    standing rules and changed facts; an unchanged fact is left to the slice."""
    from recant.store.budget import core_prefix
    from recant.store.search import load_entries
    _say(layout, ("user", "Our accountant is Alice Example."), ("assistant", "Noted."), ("user", "Rent is 2,400 a month."))
    c = GroupClient({1, 3}, facts={1: [("Our accountant is Alice Example.", "Our accountant is Alice Example", "company", "-", "accountant", "current")],
                                   3: [("Rent is 2,400 a month as of 2026-10-05.", "Rent is 2,400 a month", "company", "-", "monthly rent", "current")]})
    seen = []
    orig = c.complete

    def spy(req):
        if req.job == "nominate":
            seen.append(req.messages[0].content[0].text)
        return orig(req)
    c.complete = spy
    store = MemoryStore(layout, ALLOW)
    m = grouped_machinery(machinery)
    recant.extract(layout, TENANT, DAY, machinery=m, client=c, store=store, ledger=TaskStore(layout, ALLOW), principal_id="owner", lines=(1, 2))
    assert seen and all("\n3:" not in s and not s.rstrip().endswith("2,400 a month.") for s in seen)
    assert _claims(store) == ["Our accountant is Alice Example."]                       # line 3 is still in the window
    hy = m.kernel.serve.model_copy(update={"mode": "hybrid"})
    ents = load_entries(store, TENANT, "as_extracted", m.artifact.trust_rank)
    assert core_prefix(ents, None, hy, "as_extracted")["entries"] == []                   # an unchanged fact: the slice's
    assert len(core_prefix(ents, None, m.kernel.serve, "as_extracted")["entries"]) == 1

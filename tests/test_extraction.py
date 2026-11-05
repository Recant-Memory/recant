"""`recant.extract`: a consumer appends sessions through Recant, brings a bare model client, and gets
proposals in the review queue, routed tasks in the ledger, and what the run used — the library alone."""
import json
import re

import recant
from recant import episodes
from recant.log import read_run
from recant.provider import Completion, Usage, text
from recant.store import MemoryStore, TaskStore
from recant.store.review import memory_queue
from recant_fixtures import ALLOWLIST, TENANT
from test_extract_rules import live_machinery

DAY = "2026-10-05"
ALLOW = list(ALLOWLIST) + ["hot/sessions/????-??-??/*.md", "hot/????-??-??.md"]


def _json(d, out=40):
    return Completion(content=(text(json.dumps(d)),), stop_reason="end_turn", usage=Usage(input=100, output=out))


class Client:
    """A bare ModelClient: complete(request) -> Completion. Nominates the DECISION and TODO lines; writes the first up
    as a memory, routes the second as a task, and writes that up as an open task."""

    def __init__(self):
        self.jobs = []

    def complete(self, req):
        self.jobs.append(req.job)
        user = req.messages[0].content[0].text
        if req.job == "nominate":
            spans = [{"marker": "rule_or_decision", "start_line": int(n), "end_line": int(n), "quote": line}
                     for n, line in re.findall(r"^(\d+): (.*)$", user, flags=re.M) if "DECISION" in line or "TODO" in line]
            return _json({"spans": spans})
        n = int(re.search(r"The moment is lines (\d+)-", user).group(1))
        line = re.search(rf"^{n}: (.*)$", user, flags=re.M).group(1)
        common = {"actor_class": "principal", "actor_id": "owner", "quote": line, "quote_start_line": n, "quote_end_line": n,
                  "channel": "", "timestamp": "", "scope": "parser", "notes": ""}
        if req.job == "extract_tasks":
            return _json({**common, "verdict": "task", "not_a_task_reason": "none", "claim": "Rotate the parser's keys.",
                          "status": "open", "closed_line": 0})
        if "TODO" in line:
            return _json({**common, "verdict": "not_a_memory", "not_a_memory_reason": "task", "claim": "", "kind": "stated",
                          "supersession_condition": ""})
        return _json({**common, "verdict": "memory", "not_a_memory_reason": "none", "claim": "The parser ships on Fridays.",
                      "kind": "stated", "supersession_condition": "Until the release day changes."})


def test_a_day_of_sessions_becomes_proposals_and_tasks_through_recant_alone(layout, machinery):
    m = live_machinery(machinery)
    w = episodes.open_session(layout, TENANT, DAY)
    w.append("user", "DECISION: the parser ships on Fridays.", who="owner", when=f"{DAY}T09:00:00+00:00")
    w.append("assistant", "Noted.")
    w.append("user", "TODO: rotate the parser's keys tomorrow.", who="owner", when=f"{DAY}T09:05:00+00:00")
    store, ledger, client = MemoryStore(layout, ALLOW), TaskStore(layout, ALLOW), Client()

    out = recant.extract(layout, TENANT, DAY, machinery=m, client=client, store=store, ledger=ledger, principal_id="owner")

    assert out["sessions"] == 1 and out["spans"] == 2
    assert out["memory"]["written"] == 1 and out["tasks"]["routed"] == 1 and out["tasks"]["written"] == 1
    assert client.jobs == ["nominate", "extract", "extract", "extract_tasks"]
    (row,) = memory_queue(store, TENANT, principal_id="owner")                   # the proposal awaits review
    assert row["claim"] == "The parser ships on Fridays." and row["state"] == "proposed"
    assert [t["status"] for t in ledger.list(TENANT)] == ["open"]
    assert [t["opened"] for t in ledger.list(TENANT)] == [DAY]       # a live session's day is in its folder, not 1970
    # what the run used: tokens per model from its own records; no prices given, so no dollars
    assert out["spent_usd"] is None and sum(u["calls"] for u in out["usage"].values()) == 4
    assert sum(u["output"] for u in out["usage"].values()) == 160
    recs = list(read_run(layout.log_dir(TENANT) / f"{out['run']}.jsonl", typed=False))
    assert recs[0]["job"] == "extract" and recs[0]["params"]["day"] == DAY and recs[-1]["kind"] == "run_finished"
    assert recs[-1]["promoted"] == 1
    assert (layout.hot_daily_path(TENANT, DAY)).exists()

    again = recant.extract(layout, TENANT, DAY, machinery=m, client=client, store=store, ledger=ledger, principal_id="owner")
    assert again["memory"].get("written", 0) == 0 and again["counts"]["already_decided"] >= 1   # decided moments are not paid twice


def test_a_day_with_no_sessions_still_leaves_its_record(layout, machinery):
    out = recant.extract(layout, TENANT, "2026-10-04", machinery=live_machinery(machinery), client=Client(),
                         store=MemoryStore(layout, ALLOW), principal_id="owner")
    assert out["sessions"] == 0 and out["counts"]["no_sessions"] == 1
    recs = list(read_run(layout.log_dir(TENANT) / f"{out['run']}.jsonl", typed=False))
    assert recs[-1]["kind"] == "run_finished"


def test_extraction_can_be_limited_to_named_sessions_of_the_day(layout, machinery):
    m = live_machinery(machinery)
    a, b = episodes.open_session(layout, TENANT, DAY), episodes.open_session(layout, TENANT, DAY)
    a.append("user", "DECISION: the parser ships on Fridays.", who="owner", when=f"{DAY}T09:00:00+00:00")
    b.append("user", "DECISION: the linter runs on every push.", who="owner", when=f"{DAY}T10:00:00+00:00")
    client = Client()
    out = recant.extract(layout, TENANT, DAY, machinery=m, client=client, store=MemoryStore(layout, ALLOW),
                         principal_id="owner", sessions=[b.episode_path])
    assert out["sessions"] == 1 and client.jobs.count("nominate") == 1
    import pytest
    with pytest.raises(ValueError):
        recant.extract(layout, TENANT, DAY, machinery=m, client=client, store=MemoryStore(layout, ALLOW),
                       principal_id="owner", sessions=["hot/sessions/2026-10-05/nope.md"])



def test_a_tasks_day_comes_from_its_anchor_then_its_episode_and_never_the_epoch(layout):
    """A regression: reading only the file name would date every live session's tasks 1970-01-01."""
    import pytest
    from recant.harness.extract import day_of, task_day
    from recant.records import RecordError
    from recant.store.tasks import RejectedWrite
    from recant.log import LogWriter
    assert day_of("hot/sessions/2026-09-29/01J0000000000000000000TEST.md") == "2026-09-29"
    assert day_of("memory/2026-09-03.md") == "2026-09-03" and day_of("notes/.talk/2026-09-03.txt") == "2026-09-03"
    assert day_of("notes/today.md") is None
    assert task_day("2026-09-28T23:59:00+00:00", "hot/sessions/2026-09-29/x.md") == "2026-09-28"     # the anchor first
    assert task_day("1970-01-01", "hot/sessions/2026-09-29/x.md") == "2026-09-29"                    # never the epoch
    assert task_day("", "notes/today.md") is None
    ledger = TaskStore(layout, ALLOW)
    with LogWriter(layout, TENANT, "t", artifact_hash="a", kernel_hash="k") as log:
        with pytest.raises(RejectedWrite, match="real days"):
            ledger.put(TENANT, {"id": "01J0000000000000000000TASK", "claim": "c", "scope": "s", "actor": {"class": "principal", "id": "owner"},
                                "anchor": {"quote": "q", "episode": "hot/sessions/2026-09-29/x.md#L1-L1", "channel": None, "message_id": None,
                                           "timestamp": "2026-09-29"},
                                "opened": "1970-01-01", "status": "open"}, writer="t", log=log)


def _frob_session(layout):
    """A session with a made-up tool that fails three times, the agent's own account of it, a tool that ran, and a
    fact the agent found with it."""
    w = episodes.open_session(layout, TENANT, DAY)
    w.append("user", "Please load the nightly batch.", who="owner", when=f"{DAY}T09:00:00+00:00")       # L1
    for _ in range(3):                                                                                  # L2-L4
        w.append("tool", "frobnicate: (payload=None) → ValueError (40 chars)", failed=True)
    w.append("assistant", "I called frobnicate three times without a payload; same mistake again.")    # L5
    w.append("assistant", "That did not work.")                                                         # L6
    w.append("tool", "list_dir: /srv/batch holds 3 files")                                              # L7
    w.append("assistant", "The nightly batch directory /srv/batch holds 3 files.")                      # L8
    return w


def test_a_claim_whose_only_evidence_is_the_agents_own_tool_failure_is_rejected_by_structure(layout):
    """Whatever the tool's name or the wording, a claim resting only on the agent's failed calls, their
    retries, or its own account of them is not a memory; a fact found with a tool that ran, or said by the user, is."""
    from recant.harness.extract import own_tool_failure
    w = _frob_session(layout)
    lines = w.path.read_text().splitlines()
    assert lines[1].split("] ", 1)[1].startswith("Tool: ✗ frobnicate")                     # the mark is in the structure
    assert own_tool_failure(lines, 2, 4, "frobnicate needs a payload")                    # the failed calls and retries
    assert own_tool_failure(lines, 5, 5, "The batch loader is unreliable")                 # the agent's account names the tool
    assert own_tool_failure(lines, 6, 6, "frobnicate needs a payload")                     # it does not; the claim does
    assert not own_tool_failure(lines, 6, 6, "The batch is late on Mondays")               # neither names a failed tool
    assert not own_tool_failure(lines, 7, 8, "/srv/batch holds the nightly files")         # a tool that ran: a fact
    assert not own_tool_failure(lines, 1, 4, "The owner wants the nightly batch loaded")   # the user line is other evidence
    v1 = [x.replace("✗ ", "") for x in lines]                                              # a version-1 file: no marks
    assert not own_tool_failure(v1, 2, 4, "frobnicate needs a payload")


class FrobClient(Client):
    """Nominates every line; writes each up as a memory about what it cites."""

    def complete(self, req):
        self.jobs.append(req.job)
        user = req.messages[0].content[0].text
        if req.job == "nominate":
            return _json({"spans": [{"marker": "rule_or_decision", "start_line": int(n), "end_line": int(n), "quote": line}
                                    for n, line in re.findall(r"^(\d+): (.*)$", user, flags=re.M) if n in ("2", "5", "8")]})
        n = int(re.search(r"The moment is lines (\d+)-", user).group(1))
        line = re.search(rf"^{n}: (.*)$", user, flags=re.M).group(1)
        claim = {2: "frobnicate needs a payload.", 5: "frobnicate fails without a payload.", 8: "The nightly batch directory holds 3 files."}[n]
        return _json({"verdict": "memory", "not_a_memory_reason": "none", "claim": claim, "kind": "observed", "actor_class": "agent",
                      "actor_id": "agent", "quote": line, "quote_start_line": n, "quote_end_line": n, "channel": "", "timestamp": "",
                      "scope": "batch", "supersession_condition": "Until it changes.", "notes": ""})


def test_extraction_keeps_the_fact_and_drops_the_failure(layout, machinery):
    m = live_machinery(machinery)
    _frob_session(layout)
    store = MemoryStore(layout, ALLOW)
    out = recant.extract(layout, TENANT, DAY, machinery=m, client=FrobClient(), store=store, principal_id="owner")
    claims = [r["claim"] for r in memory_queue(store, TENANT, principal_id="owner")]
    assert claims == ["The nightly batch directory holds 3 files."], out["counts"]
    assert out["counts"].get("not_a_memory: own_tool_failure") == 2


def v2_machinery(machinery):
    """The fixture's machinery with memory v2's extraction switches on."""
    from dataclasses import replace
    m = live_machinery(machinery)
    k = m.kernel.model_copy(update={"extraction": m.kernel.extraction.model_copy(update={"split_facts": True, "keys": True, "evidence_rule": True})})
    return replace(m, kernel=k)


class V2Client(Client):
    """Nominates every user and assistant line it is told to; writes up what `answers` says for each line number."""

    def __init__(self, answers):
        super().__init__()
        self.answers, self.users = answers, []

    def complete(self, req):
        self.jobs.append(req.job)
        user = req.messages[0].content[0].text
        if req.job == "nominate":
            return _json({"spans": [{"marker": "rule_or_decision", "start_line": int(n), "end_line": int(n), "quote": line}
                                    for n, line in re.findall(r"^(\d+): (.*)$", user, flags=re.M) if int(n) in self.answers]})
        self.users.append(user)
        n = int(re.search(r"The moment is lines (\d+)-", user).group(1))
        facts = self.answers[n]
        def fact(c, q, attr, scope="company", line=n):                                    # noqa: E306
            return {"claim": c, "kind": "stated", "quote": q, "quote_start_line": line, "quote_end_line": line, "scope": scope,
                    "attribute": attr, "supersession_condition": "Until it changes.", "notes": ""}
        first, *more = [fact(*f) for f in facts]
        return _json({**first, "verdict": "memory", "not_a_memory_reason": "none", "actor_class": "principal", "actor_id": "owner",
                      "channel": "", "timestamp": "", "more_facts": more})


def test_v2_a_message_stating_three_facts_becomes_three_keyed_memories(layout, machinery):
    """one memory per stated fact, each with its own quote on the same line and its own key; none is taken
    for a duplicate of another."""
    m = v2_machinery(machinery)
    w = episodes.open_session(layout, TENANT, DAY)
    w.append("user", "Our bean supplier is Example Roasters, the account number is ACC-EXAMPLE1, and payroll runs on the 15th.",
             who="owner", when=f"{DAY}T09:00:00+00:00")
    store = MemoryStore(layout, ALLOW)
    client = V2Client({1: [("Our bean supplier is Example Roasters.", "Our bean supplier is Example Roasters", "bean supplier"),
                           ("The supplier account number is ACC-EXAMPLE1.", "account number is ACC-EXAMPLE1", "supplier account number"),
                           ("Payroll runs on the 15th.", "payroll runs on the 15th", "payroll day")]})
    out = recant.extract(layout, TENANT, DAY, machinery=m, client=client, store=store, principal_id="owner")
    rows = memory_queue(store, TENANT, principal_id="owner")
    keys = sorted(store.get(TENANT, r["id"])[0].key for r in rows)
    assert keys == ["company::-::bean_supplier", "company::-::payroll_day", "company::-::supplier_account_number"], out["counts"]
    assert out["counts"].get("more_facts") == 2


def test_v2_the_agents_echo_is_never_the_only_evidence(layout, machinery):
    """in code: a memory resting only on the agent's own line (its echo of a fact) is refused; the same fact on
    the principal's own line is kept. A version-1 run (the rule off) keeps both, as before."""
    w = episodes.open_session(layout, TENANT, DAY)
    w.append("user", "The equipment loan balance is 13500.", who="owner", when=f"{DAY}T09:00:00+00:00")   # L1
    w.append("assistant", "Noted: the equipment loan balance is 13,500.")                                  # L2
    answers = {1: [("The equipment loan balance is 13,500 as of 2026-10-05.", "equipment loan balance is 13500", "equipment loan balance")],
               2: [("The equipment loan balance is 13,500.", "the equipment loan balance is 13,500", "loan balance")]}
    store = MemoryStore(layout, ALLOW)
    out = recant.extract(layout, TENANT, DAY, machinery=v2_machinery(machinery), client=V2Client(answers), store=store, principal_id="owner")
    rows = memory_queue(store, TENANT, principal_id="owner")
    assert [store.get(TENANT, r["id"])[0].anchor.episode.endswith("#L1-L1") for r in rows] == [True]
    assert out["counts"].get("not_a_memory: agent_words_only") == 1


def test_v2_a_new_value_for_the_same_key_supersedes_the_old(layout, machinery):
    """the same attribute stated again with a new value, in other words, on a later day: the extractor is shown
    the store's keys, the new memory takes the same key and supersedes the old one when accepted."""
    from recant.log import LogWriter
    from recant.store.review import decide_memory
    m = v2_machinery(machinery)
    store = MemoryStore(layout, ALLOW)
    reviewer = {"kind": "human", "id": "owner"}

    def day(d, text_, answer):
        w = episodes.open_session(layout, TENANT, d)
        w.append("user", text_, who="owner", when=f"{d}T09:00:00+00:00")
        client = V2Client({1: [answer]})
        recant.extract(layout, TENANT, d, machinery=m, client=client, store=store, principal_id="owner")
        with LogWriter(layout, TENANT, "review", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
            for r in memory_queue(store, TENANT, principal_id="owner"):
                if r["state"] == "proposed":
                    decide_memory(store, TENANT, log, r["id"], to_state="probation", reason="confirmed", reviewer=reviewer,
                                  reasons=m.artifact.review_reasons)
        return client
    day("2026-10-05", "Our accountant is Dave Example.", ("Our accountant is Dave Example.", "Our accountant is Dave Example", "Accountant"))
    c = day("2026-10-08", "We changed accountants: it's Erin Example now.", ("Our accountant is Erin Example.", "it's Erin Example now", "accountants"))
    assert "company::-::accountant — Our accountant is Dave Example." in c.users[0]          # the key is shown, to be reused
    states = {store.get(TENANT, e["id"])[0].claim: e["state"] for e in store.list(TENANT)}
    assert states == {"Our accountant is Dave Example.": "superseded", "Our accountant is Erin Example.": "probation"}




def test_v2_served_as_pointers_the_core_is_smaller_carries_no_quote_and_shows_a_change(layout, machinery):
    """the claim, its date and a pointer to the source are served; the quote is not. A changed fact shows what
    it replaced. The same entries served with quotes cost more tokens."""
    from recant.log import LogWriter
    from recant.store import budget
    from recant.store.review import decide_memory
    from recant.store.search import load_entries
    m = v2_machinery(machinery)
    store = MemoryStore(layout, ALLOW)
    for d, said, ans in (("2026-10-05", "Our accountant is Dave Example, and rent is 2,400 a month.",
                          [("Our accountant is Dave Example.", "Our accountant is Dave Example", "accountant"),
                           ("Rent is 2,400 a month as of 2026-10-05.", "rent is 2,400 a month", "monthly rent")]),
                         ("2026-10-08", "We changed accountants: it's Erin Example now.",
                          [("Our accountant is Erin Example.", "it's Erin Example now", "accountant")])):
        w = episodes.open_session(layout, TENANT, d)
        w.append("user", said, who="owner", when=f"{d}T09:00:00+00:00")
        recant.extract(layout, TENANT, d, machinery=m, client=V2Client({1: ans}), store=store, principal_id="owner")
        with LogWriter(layout, TENANT, "review", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
            for r in memory_queue(store, TENANT, principal_id="owner"):
                if r["state"] == "proposed":
                    decide_memory(store, TENANT, log, r["id"], to_state="probation", reason="confirmed",
                                  reviewer={"kind": "human", "id": "owner"}, reasons=m.artifact.review_reasons)
    entries = load_entries(store, TENANT, "as_reviewed", m.artifact.trust_rank)
    quoted = budget.fit(entries, family=None, serve=m.kernel.serve, view="as_reviewed")
    pointed = budget.fit(entries, family=None, serve=m.kernel.serve.model_copy(update={"provenance": "pointer"}), view="as_reviewed")
    assert "\"it's Erin Example now\"" in quoted["core_text"] and "it's Erin Example now" not in pointed["core_text"]
    assert "Our accountant is Erin Example. (since 2026-10-08; before that: Our accountant is Dave Example.)" in pointed["core_text"]
    assert "source hot/sessions/2026-10-08/" in pointed["core_text"] and pointed["core_tokens"] < quoted["core_tokens"]

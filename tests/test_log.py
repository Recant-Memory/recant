import json

import pytest

from recant.log import LogWriter, item_key_hash, list_runs, read_run
from recant.log.records import ENVELOPE_FIELDS, RECORD_KINDS
from recant_fixtures import TENANT

SAMPLES = {
    "run_started": dict(job="x", params={"a": 1}),
    "run_finished": dict(examined=3, promoted=0, spent_usd=0.0, stopped_by="done"),
    "model_call": dict(model_id="m", tier=1, job="j", request_hash="sha256:0",
                       usage=dict(input=1, output=2, cache_read=3, cache_write=4), cost_usd=0.1),
    "nomination": dict(episode="a.md#L1-L2", markers=[dict(marker="root_cause", span="a.md#L1-L1", quote="q")],
                       model_id="m", prompt_version="v1", contradictions=[dict(span="a.md#L2-L2", against="01J00000000000000000000000")]),
    "extraction": dict(episode="a.md#L1-L2", candidate={"claim": "c"},
                       spans=[dict(field="claim", start=0, end=4, origin="source", episode="a.md#L1-L1"),
                              dict(field="claim", start=4, end=9, origin="generated")],
                       model_id="m", prompt_version="v1", verdict="rejected", rejection_reason="not_a_memory"),
    "eval_verdict": dict(item="i1", item_key_hash="sha256:1", arm="grep", answer="a", judge_model_id="m",
                         judge_prompt_version="v1", verdict=True, scores={"yield": 1.0, "noise": None}),
    "reader_answer": dict(item="i1", arm="grep", text="It was the cache.", retrieved=True, tokens_seen=900),
    "retrieval_trace": dict(item="i1", arm="grep", calls=[dict(terms=["a"], total=3, returned=["a.md#L1-L1"])], tokens_seen=120),
    "eval_report": dict(arm="grep", items=3, rates={"yield": {"k": 1, "n": 2}}, spent_usd=0.1),
    "turn": dict(session="01ARZ3NDEKTSV4RRFFQ69G5FAV", channel="C1", message_id="1.0", family="ops", model_chosen="model-large",
                 model_served="model-medium", tail=True, rounds=2, tools=[{"name": "search_memory", "returned_chars": 120}],
                 served={"core_entries": 3, "slice_entries": 2}, reply_chars=40, recent_cited=None, cost_usd=0.01),
    "coverage": dict(provider="openrouter", host="openrouter.ai", model="z-ai/glm-4.6", requests=3,
                     since="2026-11-04T10:00:00+00:00", until="2026-11-04T11:00:00+00:00",
                     usage={"input": 30, "output": 9, "cache_read": 0, "cache_write": 0}, with_usage=3),
    "tool_call": dict(session="01ARZ3NDEKTSV4RRFFQ69G5FAV", name="grep_recent", input={"terms": ["deploy"]}, returned_chars=300, error=None),
    "session_closed": dict(session="01ARZ3NDEKTSV4RRFFQ69G5FAV", channel="C1", why="gap", turns=4, tokens_seen=12000, cost_usd=0.2),
    "socket_closed": dict(channel="C1", by="keeper", silent_seconds=97.0),
    "socket_reconnected": dict(channel="C1", attempts=3, down_seconds=131.5, failures=["attempt did not return within 45s"]),
    "arm_comparison": dict(a="grep", b="context", dimension="yield", items=10, a_right_b_wrong=3, a_wrong_b_right=1,
                           a_rate=0.6, b_rate=0.4, difference_pp=20.0, p_one_sided=0.3125, margin_pp=4.5, beats_margin=False),
    "exposure": dict(item="i1", candidate="c1", count_after=2),
    "review_decision": dict(subject=dict(type="eval_item", id="i1"), outcome="fix", reason="wrong_answer",
                            reviewer=dict(kind="human", id="r1"), blind_answer="what I wrote before looking",
                            revealed_at="2026-01-01T00:00:00Z", fixed_answer="the right one"),
    "task_transition": dict(task="01J00000000000000000000000", from_state="open", to_state="done", writer="review",
                            closed="2026-01-02", closed_by="notes/2026-01-02.md#L3-L3", superseded_by=None),
    "status_transition": dict(entry="01J00000000000000000000000", from_state="live", to_state="contested",
                              writer="counter_observation", reason="seen otherwise"),
    "promotion": dict(candidate="c1", incumbent="c0", scores={"c1": {"yield": 0.8}, "c0": {"yield": 0.7}},
                      margin_pp=4.5, dial_position=1, decision="queued", shadow={"opened": "2026-01-01"}),
    "shadow_disagreement": dict(item="i1", promoted_answer="a", previous_answer="b"),
    "write_rejected": dict(payload={"id": "x"}, error="bad"),
    "redaction": dict(entry="01J00000000000000000000000", field="claim", versions=[1, 2]),
    "item_drafted": dict(item="01J00000000000000000000000", item_type="yield", origin="marked",
                         source_span="a.md#L1-L2", marker="root_cause", model_id="m", prompt_version="d1",
                         fingerprint="sha256:f", question="q", candidate_answer="a", search_terms=["x"],
                         spans=[dict(field="candidate_answer", start=0, end=1, origin="source")]),
    "draft_rejected": dict(source_span="a.md#L1-L2", origin="marked", model_id="m", prompt_version="d1",
                           reason="the drafter found nothing to ask"),
    "draft_merged": dict(item="01J00000000000000000000000", into="01J00000000000000000000001", reason="same fact",
                         by="function", markers=["root_cause", "reversal"]),
    "pool_trimmed": dict(seed=1, kept=10, surplus=5, reason="the review ceiling fits ten"),
    "held_cleared": dict(item="01J00000000000000000000000", reviewer=dict(kind="human", id="r1"), released=True),
    "blind_sample": dict(seed=7, items=["01J00000000000000000000000"], fingerprint="sha256:f", unfilled=[]),
    "blind_answer": dict(item="01J00000000000000000000000", reviewer=dict(kind="human", id="r1"),
                         text="The record does not answer this.", choice="does_not_answer"),
    "blind_revealed": dict(item="01J00000000000000000000000", reviewer=dict(kind="human", id="r1")),
    "judge_calibration": dict(item="01J00000000000000000000000", item_key_hash="sha256:1", judge_model_id="m",
                              judge_prompt_version="j1", verdict=True, rationale="same fact"),
    "founding": dict(seed=7, fingerprint="sha256:f", counts={"founding": {"total": 1}},
                     key_hashes={"founding": {"01J00000000000000000000000": "sha256:1"}}),
    "entry_deleted": dict(entry="01J00000000000000000000000", writer="review", kept_as_version=1),
    "item_amended": dict(item="01J00000000000000000000000", field="answer", before_hash="sha256:a", after_hash="sha256:b",
                         ruled_by="review-bot", reason="the key predates a later record"),
    "setting_change": dict(jobs=["*"], what="deploy", fingerprint="sha256:x"),
    "advice": dict(lever="ttl.prefix", lever_kind="applied", event="applied", action={"ttl": "5m"}, predicted_usd=1.2),
    "served": dict(session="s1", ref="m1", model_id="m", core_tokens=24000, slice_tokens=8000, measured=True),
    "notes_switch": dict(session="s1", switch=False, rule="default until 20 sessions: notes over 4000 tokens", over=False, over_run=0, sessions=1,
                         notes_tokens=800, notes_usd=0.0028, memory_usd=0.02, calls_per_session=6.0, extraction_usd_per_session=0.017,
                         figures="consumer's default over 0 sessions", measured=True, chars=3200),
}


def test_every_record_kind_has_a_sample_here():
    assert set(SAMPLES) == set(RECORD_KINDS)


def test_every_kind_round_trips_and_every_line_carries_the_envelope(layout, machinery):
    with LogWriter(layout, TENANT, "roundtrip", artifact_hash=machinery.artifact_hash,
                   kernel_hash=machinery.kernel_hash) as log:
        for kind, fields in SAMPLES.items():
            if kind not in ("run_started", "run_finished"):
                log.emit(kind, **fields)
    raw = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [r["seq"] for r in raw] == list(range(1, len(raw) + 1))
    for r in raw:
        assert all(f in r for f in ENVELOPE_FIELDS)
        assert r["artifact"] == machinery.artifact_hash and r["kernel"] == machinery.kernel_hash
        assert r["tenant"] == TENANT and r["run"] == log.run and r["v"] == 1
    typed = list(read_run(log.path))
    assert [t.kind for t in typed] == [r["kind"] for r in raw]
    assert {t.kind for t in typed} == set(RECORD_KINDS)
    blind = next(t for t in typed if t.kind == "review_decision")
    assert blind.blind_answer == "what I wrote before looking"
    move = next(r for r in raw if r["kind"] == "status_transition")
    assert move["from"] == "live" and move["to"] == "contested"


def test_a_run_always_says_how_it_ended(layout, machinery):
    hashes = dict(artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash)
    with LogWriter(layout, TENANT, "quiet", **hashes) as quiet:
        pass
    with pytest.raises(RuntimeError):
        with LogWriter(layout, TENANT, "broken", **hashes) as broken:
            raise RuntimeError("boom")
    class Capped(RuntimeError):
        stopped_by = "ceiling"          # a consumer's ceiling exception says how the run ended; the writer reads it
    with pytest.raises(Capped):
        with LogWriter(layout, TENANT, "capped", **hashes) as capped:
            raise Capped("over")
    never = LogWriter(layout, TENANT, "crashed", **hashes)  # never finished
    never._fh.close()
    runs = {r["job"]: r for r in list_runs(layout, TENANT)}
    assert runs["quiet"]["stopped_by"] == "done" and runs["quiet"]["finished"]
    assert runs["broken"]["stopped_by"] == "error"
    assert runs["capped"]["stopped_by"] == "ceiling"
    assert runs["crashed"]["finished"] is False
    last = list(read_run(quiet.path))[-1]
    assert (last.examined, last.promoted) == (0, 0)  # silence is a record, not an absence


def test_unknown_kinds_cannot_be_written_but_can_be_read(layout, log):
    with pytest.raises(ValueError):
        log.emit("telepathy", thought="x")
    with pytest.raises(Exception):
        log.emit("exposure", item="i", candidate="c", count_after=1, confidence=0.58)
    line = {"v": 2, "ts": "t", "tenant": TENANT, "run": log.run, "seq": 99, "kind": "from_the_future",
            "artifact": "a", "kernel": "k", "novel": True}
    with open(log.path, "a") as fh:
        fh.write(json.dumps(line) + "\n")
    got = [r for r in read_run(log.path) if isinstance(r, dict)]
    assert got == [line]


def test_a_finished_log_is_closed(layout, machinery):
    log = LogWriter(layout, TENANT, "x", artifact_hash="a", kernel_hash="k")
    log.finish()
    with pytest.raises(RuntimeError):
        log.emit("exposure", item="i", candidate="c", count_after=1)


def test_item_key_hash_is_per_item_stable_and_sensitive():
    h = item_key_hash("What fixed it?", "The cache key.", "notes/2026-01-02.md#L1-L9")
    assert h == item_key_hash(question="What fixed it?", episode="notes/2026-01-02.md#L1-L9", answer="The cache key.")
    assert h.startswith("sha256:") and len(h) == 71
    assert h != item_key_hash("What fixed it?", "The cache key!", "notes/2026-01-02.md#L1-L9")
    assert h != item_key_hash("What fixed it", "The cache key.", "notes/2026-01-02.md#L1-L9")
    assert h != item_key_hash("What fixed it?", "The cache key.", "notes/2026-01-02.md#L1-L8")


def test_the_stamp_changes_when_a_machinery_file_changes(layout, machinery):
    from recant.machinery import Machinery
    path = layout.machinery_dir / "kernel.yaml"
    path.write_text(path.read_text().replace("stale_days: 3", "stale_days: 4"))
    after = Machinery.load(layout)
    assert after.kernel_hash != machinery.kernel_hash and after.artifact_hash == machinery.artifact_hash


def test_an_older_tool_call_reads_its_bytes_as_bytes_fetched():
    """`bytes` meant the raw download in tool_call and the model's text in the turn line; the
    tool_call field is now `bytes_fetched`, and an older record still reads."""
    from recant.log.records import ToolCall
    base = {"v": 1, "ts": "2026-10-06T10:00:00.000Z", "tenant": TENANT, "run": "R", "seq": 1, "artifact": "a", "kernel": "k",
            "kind": "tool_call", "session": "S", "name": "web_fetch"}
    old = ToolCall.model_validate({**base, "bytes": 1860606})
    assert old.bytes_fetched == 1860606 and old.bytes_to_model is None
    new = ToolCall.model_validate({**base, "bytes_fetched": 9, "bytes_to_model": 5, "duration_ms": 3})
    assert (new.bytes_fetched, new.bytes_to_model, new.duration_ms) == (9, 5, 3)

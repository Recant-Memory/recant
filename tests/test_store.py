import threading

import pytest

from recant.log import read_run
from recant.records import RecordError, parse_memory_file
from recant.store import MemoryStore, NotFound, RejectedWrite
from recant.store.layout import LayoutError, StoreLayout
from recant_fixtures import ALLOWLIST, TENANT, make_record

OTHER_ID = "01J00000000000000000000000"


def kinds(log):
    return [r["kind"] for r in read_run(log.path, typed=False)]


def with_state(rec, state, **extra):
    out = dict(rec)
    out["status"] = {"state": state, "reason": extra.pop("reason", "reviewed"),
                     "failure_class": extra.pop("failure_class", None),
                     "superseded_by": extra.pop("superseded_by", None)}
    return out


def test_write_then_read_round_trips_byte_for_byte(store, layout, log):
    rec = store.put(TENANT, make_record(), writer="consolidation", log=log, notes="why this matters\n")
    path = layout.entry_path(TENANT, rec.id)
    on_disk = path.read_bytes()
    back, notes = store.get(TENANT, rec.id)
    assert back == rec and notes == "why this matters\n"
    from recant.records import render_memory_file
    assert render_memory_file(back, notes).encode() == on_disk
    assert store.versions(TENANT, rec.id) == []  # a first write keeps no version


def test_second_write_keeps_the_first_as_version_one(store, layout, log):
    first = make_record()
    store.put(TENANT, first, writer="consolidation", log=log)
    before = layout.entry_path(TENANT, first["id"]).read_bytes()
    store.put(TENANT, with_state(dict(first, claim="A corrected claim."), "probation"), writer="review", log=log)
    assert [v["n"] for v in store.versions(TENANT, first["id"])] == [1]
    assert store.read_version(TENANT, first["id"], 1).encode() == before
    assert store.get(TENANT, first["id"])[0].claim == "A corrected claim."


def test_a_transition_is_logged_with_its_writer(store, log):
    rec = make_record()
    store.put(TENANT, rec, writer="consolidation", log=log)
    store.put(TENANT, with_state(rec, "probation"), writer="review", log=log)
    store.put(TENANT, with_state(rec, "probation"), writer="review", log=log)  # an edit, not a transition
    moves = [r for r in read_run(log.path, typed=False) if r["kind"] == "status_transition"]
    assert [(m["from"], m["to"], m["writer"]) for m in moves] == \
        [(None, "proposed", "consolidation"), ("proposed", "probation", "review")]


def test_a_new_entry_cannot_start_live_and_the_refusal_is_logged(store, layout, log):
    rec = with_state(make_record(), "live")
    with pytest.raises(RejectedWrite, match="proposed"):
        store.put(TENANT, rec, writer="consolidation", log=log)
    assert not layout.entry_path(TENANT, rec["id"]).exists()
    rejected = [r for r in read_run(log.path, typed=False) if r["kind"] == "write_rejected"]
    assert len(rejected) == 1 and rejected[0]["payload"]["id"] == rec["id"] and "proposed" in rejected[0]["error"]


def test_an_illegal_transition_changes_nothing_on_disk(store, layout, log):
    rec = make_record()
    store.put(TENANT, rec, writer="consolidation", log=log)
    before = layout.entry_path(TENANT, rec["id"]).read_bytes()
    with pytest.raises(RejectedWrite, match="proposed -> live"):
        store.put(TENANT, with_state(rec, "live"), writer="review", log=log)
    assert layout.entry_path(TENANT, rec["id"]).read_bytes() == before
    assert store.versions(TENANT, rec["id"]) == []


def test_terminal_states_have_no_exit(store, log):
    rec = make_record()
    store.put(TENANT, rec, writer="consolidation", log=log)
    store.put(TENANT, with_state(rec, "retired", failure_class="staleness"), writer="review", log=log)
    with pytest.raises(RejectedWrite, match="retired -> live"):
        store.put(TENANT, with_state(rec, "live"), writer="review", log=log)


def test_superseding_keeps_the_original_and_points_at_the_replacement(store, log):
    rec = make_record()
    for state in ("proposed", "probation", "live"):
        store.put(TENANT, with_state(rec, state) if state != "proposed" else rec, writer="review", log=log)
    store.put(TENANT, with_state(rec, "superseded", reason="contradicted later the same day",
                                 failure_class="contradiction", superseded_by=OTHER_ID),
              writer="correction", log=log)
    got, _ = store.get(TENANT, rec["id"])
    assert got.claim == rec["claim"] and got.status.superseded_by == OTHER_ID
    assert got.status.failure_class == "contradiction"


def test_bad_payloads_go_to_the_dead_letter(store, log):
    for payload in [make_record(confidence=0.58), {"id": "x"}, "not even a mapping"]:
        with pytest.raises(RejectedWrite):
            store.put(TENANT, payload, writer="consolidation", log=log)
    assert kinds(log).count("write_rejected") == 3


def test_an_anchor_outside_the_allowlist_is_refused(store, log):
    rec = make_record()
    rec["anchor"]["episode"] = "notes/credentials.md#L1-L1"
    with pytest.raises(RejectedWrite, match="allowlist"):
        store.put(TENANT, rec, writer="consolidation", log=log)
    rec = make_record(derived_from=["notes/.hidden/events.jsonl#L1-L2"])
    with pytest.raises(RejectedWrite, match="allowlist"):
        store.put(TENANT, rec, writer="consolidation", log=log)


def test_a_write_names_its_writer(store, log):
    with pytest.raises(RejectedWrite, match="writer"):
        store.put(TENANT, make_record(), writer=" ", log=log)


def test_delete_keeps_the_last_state_even_for_an_entry_written_once(store, layout, log):
    rec = make_record()
    store.put(TENANT, rec, writer="consolidation", log=log)
    body = layout.entry_path(TENANT, rec["id"]).read_bytes()
    n = store.delete(TENANT, rec["id"], writer="review", log=log)
    assert n == 1 and not layout.entry_path(TENANT, rec["id"]).exists()
    assert store.read_version(TENANT, rec["id"], 1).encode() == body
    with pytest.raises(NotFound):
        store.get(TENANT, rec["id"])
    assert "entry_deleted" in kinds(log)


def test_redaction_rewrites_the_live_file_and_every_version_and_logs_only_where(store, layout, log):
    rec = make_record()
    secret = "use the key file, always"
    store.put(TENANT, rec, writer="consolidation", log=log)
    store.put(TENANT, with_state(rec, "probation"), writer="review", log=log)
    store.put(TENANT, with_state(rec, "live"), writer="review", log=log)
    touched = store.redact(TENANT, rec["id"], "anchor.quote", "[redacted]", log=log)
    assert touched == [1, 2]
    everything = [layout.entry_path(TENANT, rec["id"])] + sorted(layout.versions_dir(TENANT, rec["id"]).glob("*.md"))
    assert len(everything) == 3
    for path in everything:
        text = path.read_text()
        assert secret not in text
        assert parse_memory_file(text)[0].anchor.quote == "[redacted]"
    states = [parse_memory_file(p.read_text())[0].status.state for p in everything]
    assert states == ["live", "proposed", "probation"]  # the trail is preserved
    redactions = [r for r in read_run(log.path, typed=False) if r["kind"] == "redaction"]
    assert redactions[0]["versions"] == [1, 2] and redactions[0]["field"] == "anchor.quote"
    assert secret not in log.path.read_text()  # the log says where, never what


def test_redaction_refuses_structural_fields_and_blank_markers(store, log):
    rec = make_record()
    store.put(TENANT, rec, writer="consolidation", log=log)
    for field in ("id", "kind", "status.state", "anchor.episode"):
        with pytest.raises(RecordError):
            store.redact(TENANT, rec["id"], field, "[redacted]", log=log)
    with pytest.raises(RecordError):
        store.redact(TENANT, rec["id"], "claim", "  ", log=log)


def test_list_filters_by_state(store, log):
    a, b = make_record(), make_record()
    store.put(TENANT, a, writer="consolidation", log=log)
    store.put(TENANT, b, writer="consolidation", log=log)
    store.put(TENANT, with_state(b, "probation"), writer="review", log=log)
    assert {e["id"] for e in store.list(TENANT)} == {a["id"], b["id"]}
    assert store.list(TENANT, state="probation") == [{"id": b["id"], "state": "probation"}]


def test_concurrent_writers_never_share_a_version_number(store, layout, log):
    rec = make_record()
    store.put(TENANT, rec, writer="consolidation", log=log)
    errors = []

    def edit(i):
        try:
            for j in range(10):
                store.put(TENANT, dict(rec, claim=f"claim from thread {i} edit {j}"), writer="edit", log=log)
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=edit, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    numbers = [v["n"] for v in store.versions(TENANT, rec["id"])]
    assert numbers == list(range(1, 41))
    for n in numbers:  # every version is a whole, valid file, never a torn write
        parse_memory_file(store.read_version(TENANT, rec["id"], n))


@pytest.mark.parametrize("tenant", ["..", "../t1", "a/b", "", ".hidden", "t1/../t2", "x" * 200])
def test_tenant_ids_cannot_escape_their_directory(tmp_path, tenant):
    lay = StoreLayout(tmp_path)
    for fn in (lay.tenant_dir, lay.log_dir, lay.eval_dir, lay.lock_path):
        with pytest.raises(LayoutError):
            fn(tenant)
    with pytest.raises(LayoutError):
        lay.versions_dir(tenant, OTHER_ID)


def test_entry_ids_cannot_escape_either(layout):
    for bad in ["../../etc/passwd", "x", "01J0000000000000000000000/"]:
        with pytest.raises(LayoutError):
            layout.entry_path(TENANT, bad)


def test_tenants_do_not_see_each_other(layout, log):
    layout.ensure("t2")
    store = MemoryStore(layout, ALLOWLIST)
    rec = make_record()
    store.put(TENANT, rec, writer="consolidation", log=log)
    assert store.list("t2") == []
    with pytest.raises(NotFound):
        store.get("t2", rec["id"])


def test_layout_is_the_documented_shape_and_refuses_a_foreign_format(layout, tmp_path):
    root = layout.root
    for rel in ["tenants/t1/hot", "tenants/t1/warm", "tenants/t1/cold", "tenants/t1/index", "versions/t1",
                "log/t1", "eval/t1/founding.jsonl", "eval/t1/heldout.jsonl", "eval/t1/reserve.jsonl",
                "eval/t1/failures.jsonl", "machinery", "FORMAT.json"]:
        assert (root / rel).exists(), rel
    layout.ensure(TENANT)  # idempotent
    layout.format_path.write_text('{"layout": 1, "memory": 1, "log": 1, "machinery": 1}')
    layout.ensure(TENANT)  # a key this library does not know is not a foreign format
    layout.format_path.write_text('{"layout": 99}')
    with pytest.raises(LayoutError, match="format"):
        layout.ensure(TENANT)

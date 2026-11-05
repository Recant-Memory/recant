import copy

import pytest

from recant.records import (STATES, TRANSITIONS, RecordError, Status, check_transition, glob_to_regex,
                            parse_episode_id, parse_memory_file, path_allowed, render_memory_file,
                            validate_record)
from recant_fixtures import make_record

OTHER_ID = "01J00000000000000000000000"


def test_a_valid_record_passes_and_round_trips_through_a_file():
    rec = validate_record(make_record())
    text = render_memory_file(rec, "free notes\n---\nwith a rule line\n")
    back, notes = parse_memory_file(text)
    assert back == rec and notes == "free notes\n---\nwith a rule line\n"
    assert render_memory_file(back, notes) == text


def test_a_date_shaped_string_survives_yaml():
    rec = validate_record(make_record())
    back, _ = parse_memory_file(render_memory_file(rec))
    assert back.anchor.timestamp == "2026-01-02" and isinstance(back.anchor.timestamp, str)


@pytest.mark.parametrize("bad_field", ["confidence", "importance", "ttl", "tags", "created_at"])
def test_fields_the_format_removed_do_not_validate(bad_field):
    with pytest.raises(RecordError, match=bad_field):
        validate_record(make_record(**{bad_field: 0.58}))


@pytest.mark.parametrize("missing", ["claim", "kind", "actor", "anchor", "scope",
                                     "supersession_condition", "status", "id"])
def test_every_required_field_is_required(missing):
    rec = make_record()
    del rec[missing]
    with pytest.raises(RecordError, match=missing):
        validate_record(rec)


@pytest.mark.parametrize("mutate, names", [
    (lambda r: r.update(claim="   "), "claim"),
    (lambda r: r.update(kind="guessed"), "kind"),
    (lambda r: r.update(id="not-a-ulid"), "id"),
    (lambda r: r["actor"].update({"class": "somebody"}), "actor.class"),
    (lambda r: r["actor"].update(id=""), "actor.id"),
    (lambda r: r["anchor"].update(quote=""), "anchor.quote"),
    (lambda r: r["anchor"].update(episode="notes/2026-01-02.md"), "anchor.episode"),
    (lambda r: r["anchor"].update(timestamp=20260102), "anchor.timestamp"),
    (lambda r: r["anchor"].update(extra="x"), "anchor.extra"),
    (lambda r: r.update(derived_from=["neither"]), "derived_from"),
    (lambda r: r.update(scope=""), "scope"),
    (lambda r: r.update(supersession_condition=" "), "supersession_condition"),
    (lambda r: r["status"].update(state="verified"), "status.state"),
])
def test_each_invalid_form_is_rejected_with_the_field_named(mutate, names):
    rec = copy.deepcopy(make_record())
    mutate(rec)
    with pytest.raises(RecordError, match=names.replace(".", r"\.")):
        validate_record(rec)


def test_status_coherence():
    Status(state="superseded", reason="r", failure_class="contradiction", superseded_by=OTHER_ID)
    with pytest.raises(ValueError):
        Status(state="superseded", reason="r", failure_class="contradiction")  # no pointer
    with pytest.raises(ValueError):
        Status(state="superseded", reason="r", superseded_by=OTHER_ID)  # no failure class
    with pytest.raises(ValueError):
        Status(state="live", superseded_by=OTHER_ID)  # pointer without the state
    with pytest.raises(ValueError):
        Status(state="live", failure_class="staleness")
    with pytest.raises(ValueError):
        Status(state="superseded", reason="r", failure_class="contradiction", superseded_by="nope")


def _status(state):
    extra = {"failure_class": "contradiction", "superseded_by": OTHER_ID} if state == "superseded" else {}
    return Status(state=state, reason="because", **extra)


def test_the_transition_table_exhaustively():
    allowed = {(a, b) for a, outs in TRANSITIONS.items() for b in outs}
    assert len(allowed) == 11 and {("proposed", "superseded"), ("probation", "superseded")} <= allowed   # a probation entry can be superseded too
    for old in STATES:
        for new in STATES:
            if old == new or (old, new) in allowed:
                check_transition(old, _status(new))
            else:
                with pytest.raises(RecordError):
                    check_transition(old, _status(new))
    assert TRANSITIONS["superseded"] == TRANSITIONS["retired"] == frozenset()


def test_new_entries_enter_as_proposed_only():
    check_transition(None, Status(state="proposed"))
    for state in STATES[1:]:
        with pytest.raises(RecordError, match="proposed"):
            check_transition(None, _status(state))


def test_leaving_live_needs_a_reason():
    with pytest.raises(RecordError, match="reason"):
        check_transition("live", Status(state="contested"))
    with pytest.raises(RecordError, match="reason"):
        check_transition("live", Status(state="retired", reason="  "))
    check_transition("live", Status(state="contested", reason="a counter-observation"))


@pytest.mark.parametrize("bad", ["/abs/x.md#L1-L2", "a/../b.md#L1-L2", "./a.md#L1-L2", "a//b.md#L1-L2",
                                 "a.md#L0-L2", "a.md#L5-L2", "a.md#L1", "a.md", "a\\b.md#L1-L1", 7])
def test_episode_ids_refuse_traversal_and_nonsense(bad):
    with pytest.raises(RecordError):
        parse_episode_id(bad)


def test_episode_id_parses():
    ref = parse_episode_id("notes/.hidden/talk/2026-01-02.txt#L11-L11")
    assert (ref.path, ref.start, ref.end) == ("notes/.hidden/talk/2026-01-02.txt", 11, 11)
    assert str(ref) == "notes/.hidden/talk/2026-01-02.txt#L11-L11"


def test_globs_never_cross_a_directory():
    allow = ["notes/????-??-??.md", "notes/.hidden/talk/*.txt"]
    assert path_allowed("notes/2026-01-02.md", allow)
    assert path_allowed("notes/.hidden/talk/2026-01-02.txt", allow)
    assert not path_allowed("notes/credentials.md", allow)
    assert not path_allowed("notes/.hidden/events.jsonl", allow)
    assert not path_allowed("notes/.hidden/talk/deeper/x.txt", allow)
    assert not path_allowed("notes/sub/2026-01-02.md", allow)
    assert not glob_to_regex("a/*.md").match("a/b/c.md")


def test_file_format_errors():
    for text in ["no frontmatter", "---\nclaim: x\n", "---\n- a\n- b\n---\n", "---\n: : :\n---\n"]:
        with pytest.raises(RecordError):
            parse_memory_file(text)
    rec = validate_record(make_record())
    dup = render_memory_file(rec).replace("scope: scanner\n", "scope: scanner\nscope: other\n")
    with pytest.raises(RecordError, match="duplicate"):
        parse_memory_file(dup)

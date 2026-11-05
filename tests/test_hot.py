"""The hot tier through Recant: the consumer appends, Recant owns the file, its numbering and its
format; what it writes is read back, cited, nominated from and concatenated into the day."""
from recant import episodes
from recant.records import parse_episode_id
from recant.store import resolve
from recant_fixtures import TENANT

DAY = "2026-10-05"


def test_a_session_is_appended_numbered_and_read_back(layout):
    w = episodes.open_session(layout, TENANT, DAY)
    a = w.append("user", "Ship the parser on Friday.", who="sam", when="2026-10-05T09:00:00+00:00", where="#build")
    b = w.append("assistant", "Done.\nTagged v1.2.")
    c = w.append("tool", "run_tests: 41 passed")
    assert w.episode_path == f"hot/sessions/{DAY}/{w.id}.md" and w.lines == 4
    assert a.endswith("#L1-L1") and b.endswith("#L2-L3") and c.endswith("#L4-L4")
    raw = w.path.read_text().splitlines()
    assert raw == [f"[{w.id}#L1] User: [2026-10-05T09:00:00+00:00] sam in #build: Ship the parser on Friday.",
                   f"[{w.id}#L2] Assistant: Done.", f"[{w.id}#L3]   : Tagged v1.2.", f"[{w.id}#L4] Tool: run_tests: 41 passed"]
    folded = episodes.read_session(layout, TENANT, w.episode_path)
    assert [(l.n, l.end, l.role) for l in folded] == [(1, 1, "user"), (2, 3, "assistant"), (4, 4, "tool")]
    assert folded[1].text == "Done.\nTagged v1.2."
    assert [l.role for l in episodes.read_session(layout, TENANT, w.episode_path, fold=False)] == ["user", "assistant", "", "tool"]
    assert "Tagged v1.2." in resolve(layout, TENANT, b)                       # the id cites what was written
    assert parse_episode_id(b).path == w.episode_path


def test_a_reopened_session_continues_its_numbering_and_the_consumer_may_render_its_own_user_line(layout):
    w = episodes.open_session(layout, TENANT, DAY)
    w.append("user", "plain, with no who or when")                              # written as given
    again = episodes.open_session(layout, TENANT, DAY, w.id, render_user=lambda who, when, where, t: f"<{who}@{where}> {t}")
    assert again.append("user", "hi", who="kim", where="ops").endswith("#L2-L2")
    assert again.path.read_text().splitlines()[1].endswith("User: <kim@ops> hi")
    assert episodes.read_session(layout, TENANT, w.episode_path)[0].text == "plain, with no who or when"


def test_the_day_is_derived_from_its_sessions_and_listed(layout):
    a, b = episodes.open_session(layout, TENANT, DAY), episodes.open_session(layout, TENANT, DAY)
    a.append("user", "one"); b.append("user", "two")
    assert episodes.session_paths(layout, TENANT, DAY) == sorted([a.episode_path, b.episode_path])
    daily = episodes.derive_daily(layout, TENANT, DAY).read_text()
    assert daily.startswith(f"# Sessions — {DAY}") and f"## session {a.id}" in daily and f"## session {b.id}" in daily
    assert episodes.session_paths(layout, TENANT, "2026-10-04") == []


def test_a_role_outside_the_format_is_refused(layout):
    import pytest
    with pytest.raises(ValueError):
        episodes.open_session(layout, TENANT, DAY).append("system", "x")
    assert episodes.EPISODE_FORMAT_VERSION == 2                    # v2: the failed-call mark
    with pytest.raises(ValueError):
        episodes.open_session(layout, TENANT, DAY).append("assistant", "x", failed=True)   # only a tool line is marked

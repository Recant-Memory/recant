"""memory.mode hybrid: the carried window, the line it starts on, the slice that leaves out memories whose
source the window still carries, the core of standing rules and changed facts, and extraction over a line range."""
from types import SimpleNamespace

from recant import episodes
from recant.hybrid import carried_window, window_start_line
from recant.provider import Message, text
from recant.store.budget import in_window
from recant_fixtures import TENANT


def msg(role, n):
    return Message(role, (text("x" * n),))


def test_the_window_keeps_the_newest_turns_and_starts_on_a_user_turn():
    ms = [msg("user", 400), msg("assistant", 400), msg("user", 400), msg("assistant", 400), msg("user", 100)]
    kept, dropped = carried_window(ms, window_tokens=250, chars_per_token=4.0)     # 1,000 characters
    assert kept == ms[2:] and dropped == 1                                           # cut at a user turn, not mid-pair
    kept, dropped = carried_window(ms, window_tokens=10_000, chars_per_token=4.0)
    assert kept == ms and dropped == 0
    kept, dropped = carried_window(ms, window_tokens=10, chars_per_token=4.0)        # the last message always carried
    assert kept == ms[4:] and dropped == 2


def test_the_window_start_is_a_line_of_the_session_file(layout):
    w = episodes.open_session(layout, TENANT, "2026-10-05")
    for i in range(3):
        w.append("user", f"message {i}", who="owner", when="2026-10-05T09:00:00+00:00")
        w.append("assistant", f"reply {i}")
    path = layout.episode_file(TENANT, w.episode_path) if hasattr(w, "episode_path") else None
    f = path if path is not None else next(layout.tier_dir(TENANT, "hot").rglob("*.md"))
    assert window_start_line(f, 0) == 1
    assert window_start_line(f, 1) == 3 and window_start_line(f, 2) == 5


def test_a_memory_whose_source_the_window_still_carries_is_in_window():
    def e(ep):
        return SimpleNamespace(record=SimpleNamespace(anchor=SimpleNamespace(episode=ep)))
    win = ("hot/sessions/2026-10-05/S.md", 40)
    assert in_window(e("hot/sessions/2026-10-05/S.md#L40-L40"), win) and in_window(e("hot/sessions/2026-10-05/S.md#L90-L91"), win)
    assert not in_window(e("hot/sessions/2026-10-05/S.md#L39-L39"), win)            # left the window: memory serves it
    assert not in_window(e("hot/sessions/2026-10-04/T.md#L50-L50"), win)            # another session's source
    assert not in_window(e("hot/sessions/2026-10-05/S.md#L50-L50"), None)           # no window: nothing filtered

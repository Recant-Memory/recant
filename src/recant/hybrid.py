"""Memory v2's hybrid serving (`serve.mode: hybrid`): the host carries the last `window_tokens` of
the conversation as it does now; what leaves that window is extracted (the overflow trigger); memory serves what the
window no longer holds. These are the two pieces a host needs: where to cut what it carries, and which line of the
carried session file the cut falls on, so the slice can leave out memories whose source is still carried and the
extraction can take exactly the lines that left."""
from __future__ import annotations

from pathlib import Path


def _chars(m) -> int:
    return sum(len(getattr(b, "text", None) or "") for b in getattr(m, "content", ()))


def carried_window(messages: list, window_tokens: int, chars_per_token: float) -> tuple[list, int]:
    """(the messages to carry, how many user turns were left out): the newest messages that fit in `window_tokens`,
    cut so the first one carried is a user turn. The last message is always carried."""
    budget = window_tokens * chars_per_token
    used, start = 0, len(messages)
    for i in range(len(messages) - 1, -1, -1):
        used += _chars(messages[i])
        if used > budget and i < len(messages) - 1:
            break
        start = i
    while start < len(messages) and messages[start].role != "user":
        start += 1
    start = min(start, len(messages) - 1) if messages else 0
    return messages[start:], sum(1 for m in messages[:start] if m.role == "user")


def window_start_line(session_file: Path, dropped_user_turns: int) -> int:
    """The line of the carried session file the window starts on: the line of its (dropped + 1)-th user turn, in the
    hot tier's session format; 1 when nothing was dropped."""
    from .store.hot import LINE
    if dropped_user_turns <= 0 or not Path(session_file).exists():
        return 1
    n = 0
    for i, raw in enumerate(Path(session_file).read_text(encoding="utf-8").splitlines(), 1):
        m = LINE.match(raw)
        if m and m.group("role") == "User":
            n += 1
            if n == dropped_user_turns + 1:
                return i
    return i + 1 if n else 1

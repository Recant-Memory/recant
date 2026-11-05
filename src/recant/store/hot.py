"""The hot tier: the consumer's live conversations as episode files. The consumer decides when a
session opens and closes and what a line says; Recant owns the file, its numbering and its format.

The episode line format, version 1 (`EPISODE_FORMAT_VERSION`). A session is `hot/sessions/<day>/<id>.md` under
the tenant, `<id>` a ULID. Each line is

    [<id>#L<n>] <role>: <text>

with `<n>` counting from 1 and `<role>` one of `User`, `Assistant`, `Tool`, or two spaces for a continuation of
the line above (a multi-line text is one call, written as its first line and continuations). Lines are only
appended; an episode id `hot/sessions/<day>/<id>.md#L<a>-L<b>` cites them and stays true. A user line reads
`[<when>] <who> in <where>: <text>` unless the consumer renders it itself (`render_user`); a tool line is
`<name>: <summary>`, the summary the consumer's. Version 2 adds one mark: a tool call that itself failed (the call
errored, was refused or was removed — not a command that ran and reported a problem) is written `✗ <name>: <summary>`
(`FAILED_MARK`), so the failure is in the record's structure, not its wording. Version 1 files have no mark and read
the same.

The day's sessions concatenate into `hot/<day>.md` (`derive_daily`), a function, never a model."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ulid import ULID

from .files import atomic_write
from .layout import StoreLayout

EPISODE_FORMAT_VERSION = 2
FAILED_MARK = "✗ "           # v2: a tool line whose call failed
HOT_PATTERNS = ("hot/sessions/????-??-??/*.md", "hot/????-??-??.md")   # add to a store's allowlist to anchor in sessions
LINE = re.compile(r"^\[(?P<sess>[^\]#]+)#L(?P<n>\d+)\] (?P<role>User|Assistant|Tool|  ): ?(?P<text>.*)$")
_LABEL = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
_ROLE = {v: k for k, v in _LABEL.items()} | {"  ": ""}

RenderUser = Callable[[Optional[str], Optional[str], Optional[str], str], str]


def render_user(who: Optional[str], when: Optional[str], where: Optional[str], text: str) -> str:
    """The default user line: `[<when>] <who> in <where>: <text>`, each part left out when not given."""
    head = " ".join(p for p in (f"[{when}]" if when else "", who or "", f"in {where}" if where else "") if p)
    return f"{head}: {text}" if head else text


@dataclass(frozen=True)
class Line:
    n: int              # the first line number
    end: int            # the last, after continuations are folded in (== n when not folded)
    role: str           # "user", "assistant", "tool"; "" for an unfolded continuation
    text: str

    @property
    def failed(self) -> bool:
        """A tool line whose call failed (format v2's mark)."""
        return self.role == "tool" and self.text.startswith(FAILED_MARK)


class SessionWriter:
    """One session's file. Appends only; line numbers are what episode ids cite."""

    def __init__(self, layout: StoreLayout, tenant: str, day: str, session_id: str, *,
                 render_user: RenderUser = render_user):
        self.layout, self.tenant, self.day, self.id = layout, tenant, day, session_id
        self.path: Path = layout.hot_session_path(tenant, day, session_id)
        self.render_user = render_user
        self._n = len(self.path.read_text(encoding="utf-8").splitlines()) if self.path.exists() else 0

    @property
    def episode_path(self) -> str:
        return f"hot/sessions/{self.day}/{self.id}.md"

    @property
    def lines(self) -> int:
        return self._n

    def append(self, role: str, text: str, *, who: Optional[str] = None, when: Optional[str] = None,
               where: Optional[str] = None, failed: bool = False) -> str:
        """Append one user, assistant or tool text; returns its episode id. A user text is rendered with who,
        when and where when any is given, and written as it is otherwise. `failed` marks a tool call that itself
        failed (format v2); extraction never keeps a claim whose only evidence is such a failure."""
        if role not in _LABEL:
            raise ValueError(f"role must be one of {sorted(_LABEL)}, not {role!r}")
        if failed and role != "tool":
            raise ValueError("only a tool line is marked failed")
        if role == "user" and (who or when or where):
            text = self.render_user(who, when, where, text)
        if failed:
            text = FAILED_MARK + text
        return self._append(_LABEL[role], text)

    def _append(self, label: str, text: str) -> str:
        first = self._n + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for i, line in enumerate((text or "").splitlines() or [""]):
                self._n += 1
                fh.write(f"[{self.id}#L{self._n}] {label if i == 0 else '  '}: {line}\n")
        return f"{self.episode_path}#L{first}-L{self._n}"


def open_session(layout: StoreLayout, tenant: str, day: str, session_id: Optional[str] = None, *,
                 render_user: RenderUser = render_user) -> SessionWriter:
    """A new session (a fresh ULID) or an existing one reopened to append."""
    return SessionWriter(layout, tenant, day, session_id or str(ULID()), render_user=render_user)


def read_lines(path: Path, *, fold: bool = True) -> list[Line]:
    """A session file's lines. Folded, a continuation joins the line above with a newline; unfolded, it is its own
    Line with role "". Text that does not match the format is skipped."""
    out: list[Line] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        n, role, text = int(m.group("n")), _ROLE[m.group("role")], m.group("text")
        if fold and role == "" and out:
            prev = out[-1]
            out[-1] = Line(prev.n, n, prev.role, prev.text + "\n" + text)
        else:
            out.append(Line(n, n, role, text))
    return out


def read_session(layout: StoreLayout, tenant: str, episode_path: str, *, fold: bool = True) -> list[Line]:
    return read_lines(layout.episode_file(tenant, episode_path), fold=fold)


def derive_daily(layout: StoreLayout, tenant: str, day: str) -> Path:
    """The day's session files concatenated into `hot/<day>.md` with a heading per session. Idempotent; a session
    file that grew is re-read whole."""
    sdir = layout.hot_session_dir(tenant, day)
    parts = [f"# Sessions — {day}", ""]
    for f in sorted(sdir.glob("*.md")) if sdir.exists() else []:
        parts += [f"## session {f.stem}", ""]
        parts += f.read_text(encoding="utf-8").splitlines()
        parts.append("")
    out = layout.hot_daily_path(tenant, day)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, ("\n".join(parts) + "\n").encode("utf-8"))
    return out


def session_paths(layout: StoreLayout, tenant: str, day: str) -> list[str]:
    """Episode paths of a day's session files, relative to the tenant directory (`hot/sessions/<day>/<id>.md`)."""
    sdir = layout.hot_session_dir(tenant, day)
    return [f"hot/sessions/{day}/{f.name}" for f in sorted(sdir.glob("*.md"))] if sdir.exists() else []

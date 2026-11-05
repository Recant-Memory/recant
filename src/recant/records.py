"""Domain records and the memory file format.

This is the only module that imports pydantic for domain types. Everything
else constructs records and reads plain attributes.
"""
from __future__ import annotations

import re
from typing import Any, Literal, NamedTuple, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MEMORY_FORMAT_VERSION = 1          # memory v2 added an optional key: additive, a record without one is written exactly as before
TASK_FORMAT_VERSION = 1

Kind = Literal["stated", "observed", "concluded", "directive"]
ActorClass = Literal["principal", "agent"]
State = Literal["proposed", "probation", "live", "contested", "superseded", "retired"]
FailureClass = Literal["contradiction", "staleness", "premise_collapse"]

KINDS: tuple[str, ...] = ("stated", "observed", "concluded", "directive")
ACTOR_CLASSES: tuple[str, ...] = ("principal", "agent")
STATES: tuple[str, ...] = ("proposed", "probation", "live", "contested", "superseded", "retired")

# The lifecycle. Superseded and retired are terminal.
# `superseded` is a review exit from proposed and probation as well as the correction's exit from live
#: a reviewer names the successor instead of retiring what it replaced.
TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"probation", "retired", "superseded"}),
    "probation": frozenset({"live", "retired", "superseded"}),
    "live": frozenset({"contested", "superseded", "retired"}),
    "contested": frozenset({"live", "superseded"}),
    "superseded": frozenset(),
    "retired": frozenset(),
}

_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
_EPISODE_RE = re.compile(r"^(?P<path>[^#]+)#L(?P<start>[1-9][0-9]*)-L(?P<end>[1-9][0-9]*)$")


class RecordError(ValueError):
    """A record, a file or a transition failed validation."""


def is_ulid(value: Any) -> bool:
    return isinstance(value, str) and bool(_ULID_RE.match(value))


class EpisodeRef(NamedTuple):
    path: str
    start: int
    end: int

    def __str__(self) -> str:
        return f"{self.path}#L{self.start}-L{self.end}"


def parse_episode_id(value: Any) -> EpisodeRef:
    """`<relative path>#L<start>-L<end>`, 1-based and inclusive."""
    if not isinstance(value, str):
        raise RecordError("episode id must be a string")
    m = _EPISODE_RE.match(value)
    if not m:
        raise RecordError(f"not an episode id: {value!r}")
    path, start, end = m.group("path"), int(m.group("start")), int(m.group("end"))
    if path.startswith("/") or "\\" in path or path != path.strip():
        raise RecordError(f"episode path must be relative and clean: {path!r}")
    if any(seg in ("", ".", "..") for seg in path.split("/")):
        raise RecordError(f"episode path has an empty, '.' or '..' segment: {path!r}")
    if start > end:
        raise RecordError(f"episode span runs backwards: {value!r}")
    return EpisodeRef(path, start, end)


def is_episode_id(value: Any) -> bool:
    try:
        parse_episode_id(value)
        return True
    except RecordError:
        return False


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """A path glob where `*` and `?` never cross a `/`."""
    out = []
    for ch in pattern:
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
    return re.compile("^" + "".join(out) + "$")


def path_allowed(path: str, allowlist: list[str] | tuple[str, ...]) -> bool:
    return any(glob_to_regex(p).match(path) for p in allowlist)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class Actor(_Strict):
    cls: ActorClass = Field(alias="class")
    id: str

    _id = field_validator("id")(_non_blank)


class Anchor(_Strict):
    quote: str
    episode: str
    channel: Optional[str] = None
    message_id: Optional[str] = None
    timestamp: Optional[str] = None

    _quote = field_validator("quote")(_non_blank)

    @field_validator("episode")
    @classmethod
    def _episode(cls, v: str) -> str:
        parse_episode_id(v)
        return v


class Status(_Strict):
    state: State
    reason: Optional[str] = None
    failure_class: Optional[FailureClass] = None
    superseded_by: Optional[str] = None

    @model_validator(mode="after")
    def _coherent(self) -> "Status":
        if (self.state == "superseded") != (self.superseded_by is not None):
            raise ValueError("superseded_by is required when, and only when, state is superseded")
        if self.superseded_by is not None and not is_ulid(self.superseded_by):
            raise ValueError("superseded_by must be a ULID")
        if self.state == "superseded" and self.failure_class is None:
            raise ValueError("a superseded entry carries a failure_class")
        if self.failure_class is not None and self.state not in ("superseded", "retired", "contested"):
            raise ValueError("failure_class belongs only to superseded, retired or contested entries")
        return self


class MemoryRecord(_Strict):
    id: str
    claim: str
    kind: Kind
    actor: Actor
    anchor: Anchor
    scope: str
    supersession_condition: str
    derived_from: list[str] = Field(default_factory=list)
    status: Status
    key: Optional[str] = None     # memory v2: "<scope>::<entity>::<attribute>", normalized; dedupe and supersession by key
    # memory v2: current (how things stand), historical (a past value stated as past) or concurrent (one of
    # several values true at once); only a current value is superseded
    value_status: Optional[Literal["current", "historical", "concurrent"]] = None

    _claim = field_validator("claim")(_non_blank)
    _scope = field_validator("scope")(_non_blank)
    _cond = field_validator("supersession_condition")(_non_blank)

    @field_validator("id")
    @classmethod
    def _id_is_ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError("id must be a ULID")
        return v

    @field_validator("derived_from")
    @classmethod
    def _refs(cls, v: list[str]) -> list[str]:
        for item in v:
            if not (is_ulid(item) or is_episode_id(item)):
                raise ValueError(f"derived_from item is neither a ULID nor an episode id: {item!r}")
        return v

    def episode_paths(self) -> list[str]:
        refs = [self.anchor.episode] + [d for d in self.derived_from if not is_ulid(d)]
        return [parse_episode_id(r).path for r in refs]

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json", by_alias=True)
        for k in ("key", "value_status"):       # a record without them is written exactly as format v1 wrote it
            if d.get(k) is None:
                d.pop(k, None)
        return d


# --- the task ledger: a second store for in-flight work ---------------------------

TaskState = Literal["open", "done", "dropped", "superseded"]
TASK_STATES: tuple[str, ...] = ("open", "done", "dropped", "superseded")
TASK_TRANSITIONS: dict[str, frozenset[str]] = {"open": frozenset({"done", "dropped", "superseded"}),
                                               "done": frozenset(), "dropped": frozenset(), "superseded": frozenset()}
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_day(value: Any) -> bool:
    return isinstance(value, str) and bool(_DAY_RE.match(value))


class TaskNote(_Strict):
    """A dated line on a task: what is still pending, what was learned, why it closed. Notes are
    appended, never edited."""
    day: str
    text: str
    by: str

    _text = field_validator("text")(_non_blank)


class TaskRecord(_Strict):
    """A task is true while open. It is never retired; it is closed with a date."""
    id: str
    claim: str
    scope: str
    actor: Actor
    anchor: Anchor
    opened: str
    status: TaskState
    closed: Optional[str] = None
    closed_by: Optional[str] = None
    superseded_by: Optional[str] = None
    derived_from: list[str] = Field(default_factory=list)
    waiting_on: Optional[Literal["principal"]] = None     # a question the agent put to the principal: open until answered
    notes: list[TaskNote] = Field(default_factory=list)   # appended on the task, open or closing; shown in the listing

    _claim = field_validator("claim")(_non_blank)
    _scope = field_validator("scope")(_non_blank)

    @field_validator("id")
    @classmethod
    def _id_is_ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError("id must be a ULID")
        return v

    @field_validator("opened", "closed")
    @classmethod
    def _days(cls, v):
        if v is not None and not _is_day(v):
            raise ValueError("a task date is YYYY-MM-DD")
        return v

    @field_validator("closed_by")
    @classmethod
    def _closed_by(cls, v):
        if v is not None and not is_episode_id(v):
            raise ValueError("closed_by is an episode id")
        return v

    @field_validator("derived_from")
    @classmethod
    def _refs(cls, v: list[str]) -> list[str]:
        for item in v:
            if not (is_ulid(item) or is_episode_id(item)):
                raise ValueError(f"derived_from item is neither a ULID nor an episode id: {item!r}")
        return v

    @model_validator(mode="after")
    def _coherent(self) -> "TaskRecord":
        if (self.status == "superseded") != (self.superseded_by is not None):
            raise ValueError("superseded_by is required when, and only when, status is superseded")
        if self.superseded_by is not None and not is_ulid(self.superseded_by):
            raise ValueError("superseded_by must be a ULID")
        if self.status == "open" and (self.closed is not None or self.closed_by is not None):
            raise ValueError("an open task has no closing date or line")
        if self.status != "open" and self.closed is None:
            raise ValueError("a closed task carries the date it closed")
        if self.closed is not None and self.closed < self.opened:
            raise ValueError("a task closes on or after the day it opened")
        return self

    def episode_paths(self) -> list[str]:
        refs = [self.anchor.episode] + [d for d in self.derived_from if not is_ulid(d)] + ([self.closed_by] if self.closed_by else [])
        return [parse_episode_id(r).path for r in refs]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def validate_task(payload: Any) -> TaskRecord:
    try:
        return TaskRecord.model_validate(payload)
    except ValidationError as exc:
        raise RecordError(_flatten(exc)) from exc


def check_task_dates(rec: TaskRecord) -> None:
    """On write: a task is dated on a real day, never a 1970-01-01 fallback. Reading stays
    lenient, so records written before can still be read and repaired."""
    if rec.opened < "2000-01-01" or (rec.closed is not None and rec.closed < "2000-01-01"):
        raise RecordError(f"a task's dates are real days, not {rec.opened}")


def check_task_transition(old_state: Optional[str], new: TaskRecord) -> None:
    if old_state is None:
        if new.status not in ("open", "done"):
            raise RecordError(f"a new task enters as open, or done when its closing line was seen, not {new.status}")
        return
    if old_state == new.status:
        return
    if new.status not in TASK_TRANSITIONS[old_state]:
        raise RecordError(f"no task transition {old_state} -> {new.status}")


def render_task_file(record: TaskRecord, notes: str = "") -> str:
    front = yaml.safe_dump(record.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False, width=10**9)
    return f"---\n{front}---\n{notes}"


def parse_task_file(text: str) -> tuple[TaskRecord, str]:
    data, notes = _split_frontmatter(text)
    return validate_task(data), notes


def validate_record(payload: Any) -> MemoryRecord:
    try:
        return MemoryRecord.model_validate(payload)
    except ValidationError as exc:
        raise RecordError(_flatten(exc)) from exc


def _flatten(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(record)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def check_transition(old_state: Optional[str], new: Status) -> None:
    """Raise unless moving from `old_state` to `new` is a row of the lifecycle."""
    if old_state is None:
        if new.state != "proposed":
            raise RecordError(f"a new entry enters as proposed, not {new.state}")
        return
    if old_state == new.state:
        return
    if new.state not in TRANSITIONS[old_state]:
        raise RecordError(f"no transition {old_state} -> {new.state}")
    if old_state == "live" and not (new.reason and new.reason.strip()):
        raise RecordError("every transition out of live carries a reason")


# --- the file format: YAML frontmatter, then free-text notes -----------------

class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise RecordError(f"duplicate key in frontmatter: {key!r}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def render_memory_file(record: MemoryRecord, notes: str = "") -> str:
    front = yaml.safe_dump(record.to_dict(), sort_keys=False, allow_unicode=True,
                           default_flow_style=False, width=10**9)
    return f"---\n{front}---\n{notes}"


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise RecordError("a record file starts with a frontmatter block")
    end = text.find("\n---\n", 3)
    if end == -1:
        if text.endswith("\n---"):
            end, notes = len(text) - 4, ""
        else:
            raise RecordError("frontmatter block is not closed")
    else:
        notes = text[end + 5:]
    try:
        data = yaml.load(text[4:end + 1], Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise RecordError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RecordError("frontmatter must be a mapping")
    return data, notes


def parse_memory_file(text: str) -> tuple[MemoryRecord, str]:
    data, notes = _split_frontmatter(text)
    return validate_record(data), notes

"""The log's record kinds. One JSON object per line; every line carries the envelope.

Some fields cannot be reconstructed after the fact and are the reason the log
exists from the first run: a rejected extraction's reason, per-dimension
scores rather than pass/fail, which spans of an extraction are source text and
which are generated, the extractor's model and prompt version, and the hash of
the answer key each verdict was scored against."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

LOG_FORMAT_VERSION = 1


class _Rec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    v: int
    ts: str
    tenant: str
    run: str
    seq: int
    artifact: str
    kernel: str


class RunStarted(_Rec):
    kind: Literal["run_started"] = "run_started"
    job: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _earlier_logs(cls, data):
        """Logs written before the library dropped its ceilings carried the consumer's `ceiling_usd` beside `params`.
        Read, it lands in `params`, which is where a consumer puts such a thing now."""
        if isinstance(data, dict) and "ceiling_usd" in data:
            data = dict(data)
            value = data.pop("ceiling_usd")
            if value is not None:
                data["params"] = {**(data.get("params") or {}), "ceiling_usd": value}
        return data


class RunFinished(_Rec):
    kind: Literal["run_finished"] = "run_finished"
    examined: int
    promoted: int
    spent_usd: Optional[float]                     # None when a call in the run had no price (no table passed)
    stopped_by: str                                # "done", "error", or what the exception that ended the run names as `stopped_by`
    error: Optional[str] = None
    counts: dict[str, int] = Field(default_factory=dict)


class UsageRec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: int
    output: int
    cache_read: int
    cache_write: int
    reasoning: int = 0                 # part of output spent reasoning (absent in older records)
    cache_write_long: int = 0          # part of cache_write at the long cache lifetime (absent in older records)


class ModelCall(_Rec):
    kind: Literal["model_call"] = "model_call"
    model_id: str
    tier: Optional[int] = None
    job: str
    request_hash: str
    usage: UsageRec
    cost_usd: Optional[float] = None               # dollars only when the consumer passed a price for the model
    billed_usd: Optional[float] = None             # the provider's own billed figure, when it reports one
    session: Optional[str] = None                  # the host conversation the call belongs to
    segments: Optional[list[dict[str, Any]]] = None   # each breakpoint: where, a hash of everything up to it, its lifetime
    setting_fp: Optional[str] = None               # the request's settings apart from its content: model, effort, cap, tools, instructions
    setting_fp_rest: Optional[str] = None          # the same without the instructions (0.4.1): the system prompt is judged
                                                   # by its stable start instead (system_prefix)
    system_prefix: Optional[list[list[Any]]] = None   # [chars, hash] of the system prompt's text up to each boundary (0.4.1)
    system_suffix: Optional[list[list[Any]]] = None   # [chars, hash] of its last chars, at the same lengths (0.4.1)
    key: Optional[str] = None                      # the consumer's label for the API key the call spent (a purpose), never the key
    extends_previous: Optional[bool] = None        # this request only appends to the previous one in its session
    batch_id: Optional[str] = None
    custom_id: Optional[str] = None
    stop_reason: Optional[str] = None
    request_id: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    ttft_ms: Optional[int] = None                  # a streamed call: time from the send to its first event
    overhead_ms: Optional[float] = None            # the metering wrapper's own time on the call, network excluded
    bench_task: Optional[str] = None               # `recant bench`: the task and attempt the call belongs to, whatever run or
                                                   # session the host put it in
    via: Optional[str] = None                      # a tool that ran the calls itself and reported them (claude-code,
                                                   # claude-agent-sdk): metered, never changed by a lever
    billing: Optional[Literal["api_key", "subscription"]] = None   # how that tool's calls were paid for
    calls: Optional[int] = None                    # this record stands for that many calls, totalled by the tool


class MarkerSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    marker: str
    span: str
    quote: str


class ContradictionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    span: str
    against: str


class Nomination(_Rec):
    kind: Literal["nomination"] = "nomination"
    episode: str
    markers: list[MarkerSpan]
    model_id: str
    prompt_version: Optional[str] = None
    contradictions: list[ContradictionRef] = Field(default_factory=list)
    dropped: list[dict[str, Any]] = Field(default_factory=list)   # what the model returned that failed the check, and why
    error: Optional[str] = None                                     # why the episode (or chunk) yielded nothing usable


class ProvenanceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    start: int
    end: int
    origin: Literal["source", "generated"]
    episode: Optional[str] = None


class Extraction(_Rec):
    kind: Literal["extraction"] = "extraction"
    episode: str
    candidate: dict[str, Any]
    spans: list[ProvenanceSpan]
    model_id: str
    prompt_version: Optional[str] = None
    verdict: Literal["accepted", "rejected"]
    rejection_reason: Optional[str] = None
    written_as: Optional[str] = None
    store: Optional[Literal["memory", "tasks"]] = None   # which store took it (the ledger)


class EvalVerdict(_Rec):
    kind: Literal["eval_verdict"] = "eval_verdict"
    item: str
    item_key_hash: str
    arm: Optional[str] = None
    candidate: Optional[str] = None
    answer: str
    judge_model_id: str
    judge_prompt_version: Optional[str] = None
    verdict: bool
    scores: dict[str, Optional[float]]
    unscorable: bool = False
    reader_model_id: Optional[str] = None
    reader_prompt_version: Optional[str] = None
    retrieved_cited_span: Optional[bool] = None   # did what the reader saw include the item's cited lines
    declared_nothing: Optional[bool] = None       # the reader said the record holds nothing relevant
    cut_off: bool = False                         # the reader hit its output cap: scored wrong, never judged
    rationale: Optional[str] = None
    hard: Optional[bool] = None
    extracted: Optional[bool] = None              # service arms: some visible entry covers the cited span
    entries_seen: Optional[int] = None            # service arms: entries the reader was shown
    store_view: Optional[str] = None              # service arms: as_extracted | as_reviewed


class ReaderAnswer(_Rec):
    """An arm's answer, logged the moment it arrives, so a run that dies before scoring has paid for nothing twice."""
    kind: Literal["reader_answer"] = "reader_answer"
    item: str
    arm: str
    text: str
    cut_off: bool = False
    failed: Optional[str] = None
    retrieved: Optional[bool] = None
    tokens_seen: int = 0
    reader_model_id: Optional[str] = None
    reader_prompt_version: Optional[str] = None
    entries_seen: Optional[int] = None


class RetrievalTrace(_Rec):
    kind: Literal["retrieval_trace"] = "retrieval_trace"
    item: str
    arm: str
    calls: list[dict[str, Any]] = Field(default_factory=list)   # each: terms, total, returned (episode ids)
    tokens_seen: int = 0
    material: Optional[str] = None                              # arm 3: the file; arm 2: "prefix"


class EvalReport(_Rec):
    kind: Literal["eval_report"] = "eval_report"
    arm: str
    items: int
    rates: dict[str, Any]
    by_hard: dict[str, Any] = Field(default_factory=dict)
    retrieval: Optional[dict[str, Any]] = None
    cut_off: int = 0
    refused: int = 0                              # the reader declined: scored wrong, never judged
    judge_failed: int = 0
    spent_usd: Optional[float] = 0.0              # None when a call in the run had no price
    cache: Optional[dict[str, Any]] = None
    calibration_run: Optional[str] = None
    reader_model_id: Optional[str] = None
    judge_model_id: Optional[str] = None
    store_view: Optional[str] = None              # service arms
    pool: str = "founding"                        # founding, or heldout for the overfitting check
    coverage: Optional[dict[str, Any]] = None     # service arms: extracted / retrieved / right split
    store: Optional[dict[str, Any]] = None        # service arms: entries, warm-tier tokens, yield per 1K tokens and per 100 entries
    budget: Optional[dict[str, Any]] = None      # service_budget: the served budget and the per-family prefixes


class ArmComparison(_Rec):
    kind: Literal["arm_comparison"] = "arm_comparison"
    a: str
    b: str
    dimension: str
    items: int
    a_right_b_wrong: int
    a_wrong_b_right: int
    a_rate: float
    b_rate: float
    difference_pp: float
    p_one_sided: float
    margin_pp: float
    beats_margin: bool


class Exposure(_Rec):
    kind: Literal["exposure"] = "exposure"
    item: str
    candidate: str
    count_after: int


class Subject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["eval_item", "memory", "promotion"]
    id: str


class Reviewer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["human", "model"]
    id: str


class ReviewDecision(_Rec):
    kind: Literal["review_decision"] = "review_decision"
    subject: Subject
    outcome: Literal["accept", "fix", "supersede", "discard", "to_live", "retire", "confirm", "done", "dropped", "still_open"]   # the last three: the task ledger
    reason: str
    note: Optional[str] = None
    reviewer: Reviewer
    blind_answer: Optional[str] = None
    revealed_at: Optional[str] = None
    fixed_answer: Optional[str] = None
    fixed_episode: Optional[str] = None
    key_hash_before: Optional[str] = None
    key_hash_after: Optional[str] = None
    hard: Optional[bool] = None
    backfilled: Optional[bool] = None              # written after the fact for a decision applied without its record
    backfill_source: Optional[str] = None          # where the decision as applied is documented


class StatusTransition(_Rec):
    kind: Literal["status_transition"] = "status_transition"
    entry: str
    from_state: Optional[str] = Field(default=None, alias="from")
    to_state: str = Field(alias="to")
    writer: str
    reason: Optional[str] = None
    failure_class: Optional[str] = None
    superseded_by: Optional[str] = None


class TaskTransition(_Rec):
    kind: Literal["task_transition"] = "task_transition"
    task: str
    from_state: Optional[str] = Field(default=None, alias="from")
    to_state: str = Field(alias="to")
    writer: str
    closed: Optional[str] = None
    closed_by: Optional[str] = None
    superseded_by: Optional[str] = None


class Turn(_Rec):
    """One agent turn: what was served, what was called, which model answered."""
    kind: Literal["turn"] = "turn"
    session: str
    channel: str
    message_id: Optional[str] = None
    family: Optional[str] = None
    model_chosen: str
    model_served: str
    tail: bool = False
    rounds: int = 0
    tools: list[dict[str, Any]] = Field(default_factory=list)       # each: name, returned_chars
    served: dict[str, Any] = Field(default_factory=dict)            # core_entries, core_tokens, slice_entries, slice_tokens
    reply_chars: int = 0
    recent_cited: Optional[bool] = None                              # grep_recent's lines cited in the reply (a word-overlap function)
    cost_usd: float = 0.0
    stopped: Optional[Literal["turn", "day", "rounds"]] = None       # a turn ended by a ceiling or the round cap is still a turn
    profile: Optional[str] = None                                    # the consumer's run profile, when it names one
    compactions: list[dict[str, Any]] = Field(default_factory=list) # each: at_round, rounds_folded, tokens_before, tokens_after, cost_usd


class Advice(_Rec):
    """One move by the cost advisor: a lever applied, opted into, checked or reverted, with
    what it was predicted to save and, once checked, what it did."""
    kind: Literal["advice"] = "advice"
    lever: str
    lever_kind: Literal["applied", "your_call", "fix_cause"]
    event: Literal["applied", "opted_in", "checked", "reverted", "evidence"]
    action: dict[str, Any] = Field(default_factory=dict)
    predicted_usd: Optional[float] = None           # per week, at the host's prices
    predicted_tokens: Optional[float] = None        # per week
    realized_usd: Optional[float] = None            # the same calls priced both ways, over the check window
    window_runs: Optional[int] = None
    note: str = ""


class ItemAmended(_Rec):
    """An eval item's answer key changed by the arbiter's ruling, never by the loop: which field, hashes
    of the text before and after, who ruled and why. The texts themselves go in the pool's amendments file."""
    kind: Literal["item_amended"] = "item_amended"
    item: str
    field: str
    before_hash: str
    after_hash: str
    ruled_by: str
    reason: str


class SettingChange(_Rec):
    """A host's setup changed for some jobs: the advisor prices a job's levers only
    on calls after the latest change to it, so a lever is never sized on a setup that no longer runs."""
    kind: Literal["setting_change"] = "setting_change"
    jobs: list[str]                                 # the jobs it touches; "*" for all
    what: str
    fingerprint: Optional[str] = None


class Served(_Rec):
    """What Recant served at the start of one host run: the memory core and the slice, in tokens, counted by
    the provider's own token counter when it has one (`measured`), estimated from characters when it has not."""
    kind: Literal["served"] = "served"
    session: Optional[str] = None
    ref: Optional[str] = None                     # the host's id for the run (a message id, a task id)
    model_id: str
    core_tokens: int
    slice_tokens: int
    measured: bool
    slice_ids: Optional[list[str]] = None         # the memory and task ids the slice carried (versions: entry_version)
    corrections: Optional[int] = None             # core entries superseded mid-session, their successors appended


class NotesSwitch(_Rec):
    """One session start's switch decision in notes mode: both prices, the rule that applied, and
    whether the store switched to memory."""
    kind: Literal["notes_switch"] = "notes_switch"
    session: Optional[str] = None
    switch: bool
    rule: str
    over: bool
    over_run: int
    sessions: int
    notes_tokens: int
    notes_usd: float
    memory_usd: float
    calls_per_session: float
    extraction_usd_per_session: float
    figures: str
    measured: bool = False                # the notes' tokens counted by the host's provider, not estimated from characters
    chars: Optional[int] = None           # the rendered notes' length, so the ratio can be read back


class CoverageCount(_Rec):
    """Model-API requests the host's process sent past Recant, counted by `recant.coverage.watch`: counts only,
    never content. `host` is the known host, or `other:<hash>` for an unknown host whose body looked like a model call."""
    kind: Literal["coverage"] = "coverage"
    provider: str
    host: str
    model: Optional[str] = None
    requests: int
    since: str                                      # the window the counts cover, ISO times
    until: str
    usage: Optional[dict[str, int]] = None          # input, output, cache_read, cache_write, summed where the response said
    with_usage: int = 0                             # how many of the requests the usage covers


class ToolCall(_Rec):
    kind: Literal["tool_call"] = "tool_call"
    session: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    returned_chars: int = 0
    error: Optional[str] = None
    duration_ms: Optional[int] = None               # the whole call, every tool
    bytes_fetched: Optional[int] = None             # what the tool pulled in: a page's raw download, a command's output
    bytes_to_model: Optional[int] = None            # the UTF-8 size of the result the model read, after the cap
    # the shell tool: command, cwd, exit; whether it ran on the principal's ok,
    # or was refused (the sandbox, or the principal said no)
    command: Optional[str] = None
    cwd: Optional[str] = None
    exit_code: Optional[int] = None
    confirmed: Optional[bool] = None
    refused: Optional[bool] = None
    waived_by: Optional[str] = None                 # the standing directive that waived the ask
    targets: Optional[list[str]] = None             # hashes of what the call touched, no text: re-fetch detection

    @model_validator(mode="before")
    @classmethod
    def _bytes_was_fetched(cls, data: Any) -> Any:
        """Older logs called `bytes_fetched` `bytes` (two meanings, one name)."""
        if isinstance(data, dict) and "bytes" in data:
            data = dict(data)
            data.setdefault("bytes_fetched", data.pop("bytes"))
        return data


class SessionClosed(_Rec):
    kind: Literal["session_closed"] = "session_closed"
    session: str
    channel: str
    why: Literal["gap", "overflow", "cutoff", "stopped", "restart_gap"]   # restart_gap: found closed when the runtime rebuilt its sessions
    turns: int
    tokens_seen: int = 0
    cost_usd: float = 0.0


class SocketClosed(_Rec):
    """The channel transport's connection closed: the close code and reason the transport saw,
    or the staleness the keeper measured. `by` says who noticed."""
    kind: Literal["socket_closed"] = "socket_closed"
    channel: str
    by: Literal["transport", "keeper"]
    code: Optional[int] = None
    reason: Optional[str] = None
    silent_seconds: Optional[float] = None       # keeper: seconds since the last pong when it gave up on the connection


class SocketReconnected(_Rec):
    kind: Literal["socket_reconnected"] = "socket_reconnected"
    channel: str
    attempts: int
    down_seconds: float
    failures: list[str] = Field(default_factory=list)   # the attempts that failed, one short reason each (last ten)


class Promotion(_Rec):
    kind: Literal["promotion"] = "promotion"
    candidate: str
    incumbent: str
    scores: dict[str, dict[str, Optional[float]]]
    margin_pp: float
    dial_position: int
    decision: str
    shadow: dict[str, Any] = Field(default_factory=dict)


class ShadowDisagreement(_Rec):
    kind: Literal["shadow_disagreement"] = "shadow_disagreement"
    item: str
    promoted_answer: str
    previous_answer: str


class WriteRejected(_Rec):
    kind: Literal["write_rejected"] = "write_rejected"
    payload: Any
    error: str


class Redaction(_Rec):
    kind: Literal["redaction"] = "redaction"
    entry: str
    field: str
    versions: list[int]
    live_file: bool = True


class EntryDeleted(_Rec):
    kind: Literal["entry_deleted"] = "entry_deleted"
    entry: str
    writer: str
    kept_as_version: int


class ItemDrafted(_Rec):
    kind: Literal["item_drafted"] = "item_drafted"
    item: str
    item_type: Literal["yield", "noise", "supersession"]
    origin: Literal["marked", "unmarked", "seeded", "noise"]
    source_span: Optional[str] = None
    marker: Optional[str] = None
    model_id: str
    prompt_version: str
    fingerprint: str
    question: str
    candidate_answer: str
    superseded_answer: Optional[str] = None
    search_terms: list[str] = Field(default_factory=list)
    prescreen_hits: Optional[int] = None
    spans: list[ProvenanceSpan] = Field(default_factory=list)
    form: Optional[str] = None          # the question's form, as the drafter named it
    held_for: Optional[str] = None      # set when the draft was kept out of the pool pending a person


class DraftRejected(_Rec):
    kind: Literal["draft_rejected"] = "draft_rejected"
    source_span: Optional[str] = None
    origin: str
    model_id: str
    prompt_version: str
    reason: str
    raw: Any = None


class DraftMerged(_Rec):
    kind: Literal["draft_merged"] = "draft_merged"
    item: str                       # the draft that was dropped
    into: Optional[str] = None      # the draft that stands for both; None when a noise draft is dropped because the record answers it
    reason: str
    by: str                         # "function" or the model id
    prompt_version: Optional[str] = None
    markers: list[str] = Field(default_factory=list)   # every marker the merged drafts carried


class PoolTrimmed(_Rec):
    kind: Literal["pool_trimmed"] = "pool_trimmed"
    seed: int
    kept: int
    surplus: int
    reason: str


class HeldCleared(_Rec):
    kind: Literal["held_cleared"] = "held_cleared"
    item: str
    reviewer: Reviewer
    released: bool
    note: Optional[str] = None
    answered_by: Optional[str] = None     # the line that answers it, when it was dropped as answered
    hits_shown: Optional[int] = None


class BlindSample(_Rec):
    kind: Literal["blind_sample"] = "blind_sample"
    seed: int
    items: list[str]
    fingerprint: str
    unfilled: list[str] = Field(default_factory=list)


class BlindAnswer(_Rec):
    kind: Literal["blind_answer"] = "blind_answer"
    item: str
    reviewer: Reviewer
    text: str
    choice: Optional[Literal["answers", "does_not_answer", "partially"]] = None   # set when a key was pressed, not words typed


class BlindRevealed(_Rec):
    kind: Literal["blind_revealed"] = "blind_revealed"
    item: str
    reviewer: Reviewer


class JudgeCalibration(_Rec):
    kind: Literal["judge_calibration"] = "judge_calibration"
    item: str
    item_key_hash: str
    judge_model_id: str
    judge_prompt_version: Optional[str] = None
    verdict: bool
    rationale: Optional[str] = None


class Founding(_Rec):
    kind: Literal["founding"] = "founding"
    seed: int
    fingerprint: str
    counts: dict[str, dict[str, int]]
    key_hashes: dict[str, dict[str, str]]


_ALL = (RunStarted, RunFinished, ModelCall, Nomination, Extraction, EvalVerdict,
        Exposure, ReviewDecision, StatusTransition, Promotion, ShadowDisagreement, WriteRejected,
        Redaction, EntryDeleted, ItemDrafted, DraftRejected, DraftMerged, PoolTrimmed, HeldCleared, BlindSample, BlindAnswer, BlindRevealed,
        JudgeCalibration, Founding, ReaderAnswer, RetrievalTrace, EvalReport, ArmComparison, TaskTransition,
        Turn, ToolCall, SessionClosed, SocketClosed, SocketReconnected, Served, Advice, SettingChange, ItemAmended, NotesSwitch, CoverageCount)
RECORD_KINDS: dict[str, type[_Rec]] = {m.model_fields["kind"].default: m for m in _ALL}
ENVELOPE_FIELDS = ("v", "ts", "tenant", "run", "seq", "kind", "artifact", "kernel")


def register_record_kind(model: type[_Rec]) -> type[_Rec]:
    """A consumer's own record kind, written through the same writer and carrying the same envelope. The library
    defines the kinds it writes; what a consumer adds (a refusal at its own ceiling, say) it registers here. A kind
    the library already defines cannot be replaced."""
    kind = model.model_fields["kind"].default
    if kind in RECORD_KINDS and RECORD_KINDS[kind] is not model:
        if RECORD_KINDS[kind].__module__ == __name__:
            raise ValueError(f"{kind!r} is one of the library's own record kinds")
    RECORD_KINDS[kind] = model
    return model


Record = _Rec     # the envelope a consumer's record kind extends

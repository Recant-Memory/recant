"""The two machinery files.

The artifact holds what a learning loop may tune; a candidate is a diff against
it. The kernel holds what nothing automated may touch: thresholds, floors, sizes,
activation counts, the served budget and the corpus allowlist — everything that
enters the machinery hash. A key in the wrong file fails to load. The library
ships the shape; every value comes from the consumer. Prices, spend ceilings and
an agent runtime's settings are the consumer's own and are not here (format 4)."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..log.hashing import file_hash
from ..records import ACTOR_CLASSES, KINDS
from ..store.layout import StoreLayout

MACHINERY_FORMAT_VERSION = 4            # the kernel's; the artifact is unchanged at 3 and says so in its own `format`
ARTIFACT_FILE, KERNEL_FILE = "artifact.yaml", "kernel.yaml"



DEFAULTS = Path(__file__).resolve().parents[1] / "defaults"
DEFAULT_ARTIFACT, DEFAULT_KERNEL = DEFAULTS / "artifact.yaml", DEFAULTS / "kernel.yaml"   # Machinery.seed_defaults

class MachineryError(ValueError):
    pass


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelJob(_M):
    """A model, and the prompt it runs. The text sits beside its version so
    that the file's hash covers the thing that actually changes."""
    model: str
    prompt_version: Optional[str]
    prompt: Optional[str]
    reasoning: Optional[Literal["off", "on"]] = None   # the job's default for the model's own reasoning; unset: the model's default
    max_tokens: Optional[int] = Field(default=None, ge=1)   # the job's output cap, a runaway guard set above its p99.9; unset: the caller's

    @model_validator(mode="after")
    def _version_names_a_text(self) -> "ModelJob":
        if (self.prompt_version is None) != (self.prompt is None):
            raise ValueError("prompt and prompt_version are set together or not at all")
        return self


class Marker(_M):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Extractor(_M):
    first_pass: ModelJob
    steady_state: ModelJob
    tasks: Optional[ModelJob] = None      # the task ledger's extractor; the `task` class becomes a route


class Reader(_M):
    model: str


class Artifact(_M):
    format: Literal[3]
    version_note: str
    markers: list[Marker] = Field(min_length=1)
    extractor: Extractor
    nomination: ModelJob
    judge: ModelJob
    reader: Reader
    dedup_threshold: Optional[float]
    ranking_weights: Optional[dict[str, float]]
    fitness_weights: Optional[dict[str, float]]
    replay_sampling: Literal["full", "subsample"]
    routing: dict[int, str]
    trust_table: list[list[tuple[str, str]]]
    contestation_threshold: int = Field(ge=1)
    review_reasons: list[str] = Field(min_length=1)

    @field_validator("routing")
    @classmethod
    def _tiers(cls, v: dict[int, str]) -> dict[int, str]:
        if set(v) != {1, 2, 3}:
            raise ValueError("routing names exactly tiers 1, 2 and 3")
        return v

    @field_validator("trust_table")
    @classmethod
    def _trust(cls, v):
        seen = [pair for rank in v for pair in rank]
        want = {(k, c) for k in KINDS for c in ACTOR_CLASSES}
        if set(seen) != want or len(seen) != len(want):
            raise ValueError("trust_table ranks every (kind, actor class) pair exactly once")
        return v

    @field_validator("markers")
    @classmethod
    def _unique_markers(cls, v):
        names = [m.name for m in v]
        if len(set(names)) != len(names):
            raise ValueError("marker names are unique")
        return v

    def marker_names(self) -> list[str]:
        return [m.name for m in self.markers]

    def trust_rank(self, kind: str, actor_class: str) -> int:
        """1 is most trusted. Looked up, never computed."""
        for rank, pairs in enumerate(self.trust_table, 1):
            if (kind, actor_class) in pairs:
                return rank
        raise MachineryError(f"no trust rank for ({kind}, {actor_class})")

    def models(self) -> set[str]:
        return {self.extractor.first_pass.model, self.extractor.steady_state.model,
                *([self.extractor.tasks.model] if self.extractor.tasks else []),
                self.nomination.model, self.judge.model, self.reader.model, *self.routing.values()}


class Gate(_M):
    margin_pp: float
    floor_pp: float
    suite_size: int


class Heldout(_M):
    slice: int
    reserve: Optional[int]


class Activation(_M):
    consolidation_rate: int
    rollback: int
    review_agreement: int


class Corpus(_M):
    allowlist: list[str] = Field(min_length=1)
    full_context: list[str] = Field(min_length=1)   # the allowlist patterns whose files form the full-context arm's prefix


class EvalPrompt(_M):
    tier: Literal[1, 2, 3]
    prompt_version: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class Unmarked(_M):
    """Yield items drafted from windows the nomination pass did not mark, so
    the suite can see what the marker set misses."""
    share: float = Field(ge=0, le=1)
    window_lines: int = Field(ge=1)
    oversample: float = Field(ge=1)


class ReaderPrompt(_M):
    prompt_version: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(ge=64)
    effort: Optional[Literal["low", "medium", "high"]] = None


class GrepArm(_M):
    calls: int = Field(ge=1)
    lines_per_call: int = Field(ge=1)
    context_lines: int = Field(ge=0)


class ServiceArm(_M):
    rounds: int = Field(ge=1)
    entries_per_search: int = Field(ge=1)
    chars_per_search: int = Field(ge=100)


class Arms(_M):
    grep: GrepArm
    context_cache_ttl: Literal["5m", "1h"]
    service: ServiceArm


class EvalProduction(_M):
    """How the eval set itself is produced. Nothing automated may touch it."""
    blind_first_sample: int = Field(ge=1)
    blind_hard_cap: int = Field(ge=0)     # hard-tagged items in the blind-first sample: taken first, up to this many
    judge_agreement_floor: float = Field(gt=0, le=1)
    noise_draft_ratio: float = Field(ge=0)
    context_lines: int = Field(ge=0)
    dedupe_question_overlap: float = Field(gt=0, le=1)   # word overlap above which two questions on touching spans are one
    unmarked: Unmarked
    drafting: EvalPrompt
    noise: EvalPrompt
    seeded: EvalPrompt
    dedupe: EvalPrompt
    review: EvalPrompt
    held_review: EvalPrompt   # a model rules on held noise drafts: does any search hit answer the question
    reader: ReaderPrompt      # the answering half of every arm; kernel because tuning it moves every arm at once
    arms: Arms


class Tasks(_M):
    stale_after_days: int = Field(ge=1)   # the corpus replay: an open task older than this with no closing line is "stale, unconfirmed"
    stale_days: int = Field(ge=1)         # the live ledger: "open, unconfirmed since <date>" after this many days
    presumed_dropped_days: int = Field(ge=1)   # the live ledger: "presumed dropped" after this many days, kept as history

    @model_validator(mode="after")
    def _ordered(self):
        if self.presumed_dropped_days < self.stale_days:
            raise ValueError("presumed_dropped_days is not before stale_days")
        return self


class ExtractionRules(_M):
    """Two mechanical rules on the extractor's candidates: within a run, a claim overlapping
    an already-written one at `run_dedupe_overlap`, or anchored on a line the store already holds an entry for, folds
    to it; an agent's line is never written as `stated` — the identity file owns what the agent says of itself; an
    agent turn may still `conclude`."""
    run_dedupe_overlap: float = Field(default=0.7, ge=0.0, le=1.0)
    # write-time dedupe against the whole store: a candidate whose distinctive tokens (paths, file names, numbers of
    # three characters or more) are all in a standing entry, adding no number it lacks and overlapping it at this, is
    # that entry restated and attaches to it; null turns the rule off
    store_dedupe_overlap: Optional[float] = Field(default=0.35, ge=0.0, le=1.0)
    agent_lines_never_stated: bool = True
    # memory v2, each off unless the kernel turns it on:
    split_facts: bool = False      # one memory per stated fact (the extractor's more_facts)
    keys: bool = False             # a (scope, attribute) key on each memory; dedupe and supersession by key first
    evidence_rule: bool = False    # a memory rests on the principal's own turn or a tool's result, checked in code
    grouped: bool = False          # nearby moments of one episode in one bounded request, each fact anchored alone
    one_pass: bool = False         # that request returns memories and tasks, two lists (needs `grouped`)
    group_max_moments: int = Field(default=6, ge=1)
    group_max_lines: int = Field(default=120, ge=10)


class ScopeFamily(_M):
    """A coarse scope a session is served: entries whose `scope` (or claim) carries one of `match` belong to it;
    sessions on one of `channels` are served it; a `global` family is served to every session."""
    name: str = Field(min_length=1)
    match: list[str] = Field(min_length=1)
    channels: list[str] = Field(default_factory=list)
    global_: bool = Field(default=False, alias="global")


class Serve(_M):
    """The served path's budget: the memory prefix a call may carry, in tokens, and the measured
    chars-per-token ratio the fitter converts with, so fitting is arithmetic. The consumer sets both."""
    memory_budget_tokens: int = Field(ge=1000)
    chars_per_token: float = Field(gt=0)
    scopes: list[ScopeFamily] = Field(default_factory=list)
    core_tokens: int = Field(ge=0)        # the cached part of the served prefix: globals, live, probation, by recency
    slice_tokens: int = Field(ge=0)       # the retrieved part, per call: one lexical shot over what the core left out
    provenance: Literal["quote", "pointer"] = "quote"   # "pointer" serves the claim, its date and a pointer to its
                                                        # source; the quote is fetched through the memory tool when needed
    serve_once: bool = False              # the slice leaves out a memory version already visible in the conversation
    core_only_when_fits: bool = False     # when the core holds every memory entry, no slice is served
                                          # and the host offers no memory search; both come back once the store outgrows the core
    learn_chars_per_token: bool = False   # fit with the ratio learned from counted tokens, once the host has one
    # "hybrid" carries the last `window_tokens` of the conversation as the host does, extracts what leaves
    # that window, serves no memory whose source is still in it, and keeps standing rules and changed facts in the core
    # "auto" serves the agent's own structured notes (`recant.notes`) until they cost more a session than
    # memory would, for notes_switch_runs sessions running, then migrates them and runs memory as "normal"; until the host
    # has notes_min_sessions of its own data the switch is the notes rendering larger than core_tokens + slice_tokens
    mode: Literal["normal", "hybrid", "auto"] = "normal"
    window_tokens: int = Field(default=30000, ge=1000)
    notes_default_tokens: int = Field(default=95000, ge=0)   # the switch's default until the host has 20 sessions of its own prices: the
                                                             # measured crossover, in the provider's counted tokens (CHANGELOG 0.1.0)
    notes_min_sessions: int = Field(default=20, ge=1)
    notes_switch_runs: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _split_inside_budget(self):
        if self.core_tokens + self.slice_tokens > self.memory_budget_tokens:
            raise ValueError("core_tokens + slice_tokens exceed memory_budget_tokens")
        return self


class CostCheckGate(_M):
    cost_cut_min: float = Field(ge=0, le=1)       # the on arm's cost per successful task at least this much lower
    cache_hit_drop_max_pp: float = Field(ge=0)    # the on arm's cache-hit ratio at most this many points lower


class CostCheck(_M):
    """The cost check's settings: repetitions per arm and the gate. In the kernel because a gate is
    a threshold nothing automated may move, and it enters the machinery hash like `gate`."""
    reps: int = Field(ge=1)
    level: float = Field(gt=0, lt=1)              # the intervals' coverage
    resamples: int = Field(ge=100)
    seed: int = 0
    gate: CostCheckGate


class Kernel(_M):
    format: Literal[4]
    gate: Gate
    noise_share: tuple[float, float]
    heldout: Heldout
    activation: Activation
    shadow_window_days: Optional[int]
    exploration_floors: Optional[dict[str, float]]
    corpus: Corpus
    eval: EvalProduction
    tasks: Tasks                          # the task ledger; every value comes from the consumer
    serve: Serve                          # the served path's memory budget and scope families
    extraction: ExtractionRules           # the live pass's two rules; every kernel key is the consumer's to state
    cost_check: Optional[CostCheck]       # stated, and null means the consumer runs no cost check


def _load(path: Path, model):
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        found = data.get("format", 1) if isinstance(data, dict) else None
        want = get_args(model.model_fields["format"].annotation)[0]
        if found != want:
            missing = sorted(set(model.model_fields) - set(data or {})) if isinstance(data, dict) else []
            extra = sorted(set(data) - set(model.model_fields)) if isinstance(data, dict) else []
            raise MachineryError(f"{path.name} is machinery format {found}; this library reads format {want}. "
                                 f"Keys it lacks: {missing}; keys it carries that moved out: {extra}")
        return model.model_validate(data)
    except FileNotFoundError as exc:
        raise MachineryError(f"missing machinery file: {path}") from exc
    except (yaml.YAMLError, ValidationError) as exc:
        raise MachineryError(f"{path.name}: {exc}") from exc


@dataclass(frozen=True)
class Machinery:
    artifact: Artifact
    kernel: Kernel
    artifact_hash: str
    kernel_hash: str

    @classmethod
    def from_files(cls, artifact_path: str | Path, kernel_path: str | Path) -> "Machinery":
        a, k = Path(artifact_path), Path(kernel_path)
        artifact, kernel = _load(a, Artifact), _load(k, Kernel)
        return cls(artifact, kernel, file_hash(a), file_hash(k))

    @classmethod
    def load(cls, layout: StoreLayout) -> "Machinery":
        d = layout.machinery_dir
        return cls.from_files(d / ARTIFACT_FILE, d / KERNEL_FILE)

    @staticmethod
    def nomination_fingerprint_of(artifact: "Artifact") -> str:
        """What produced a nomination run: the markers and the nomination job. A change anywhere else in
        the artifact (the judge, the reader, the trust table) does not change what was nominated, so a run
        stays reusable across those changes."""
        from ..log.hashing import canonical_json, sha256_hex
        body = {"markers": [m.model_dump() for m in artifact.markers],
                "nomination": [artifact.nomination.model, artifact.nomination.prompt_version, artifact.nomination.prompt]}
        return sha256_hex(canonical_json(body).encode())

    def nomination_fingerprint(self) -> str:
        return self.nomination_fingerprint_of(self.artifact)

    def extractor_fingerprint(self, job: str = "first_pass") -> str:
        """What produced an extraction: the extractor job's model and prompt, and the markers the spans
        carry. A span decided under this fingerprint is not paid for again."""
        from ..log.hashing import canonical_json, sha256_hex
        ex = getattr(self.artifact.extractor, job)
        if ex is None:
            raise MachineryError(f"the artifact has no {job} extractor")
        body = {"job": job, "model": ex.model, "prompt_version": ex.prompt_version, "prompt": ex.prompt,
                "markers": [m.model_dump() for m in self.artifact.markers],
                "dedup_threshold": self.artifact.dedup_threshold}
        return sha256_hex(canonical_json(body).encode())

    @staticmethod
    def artifact_at(layout: StoreLayout, artifact_hash: str) -> Optional["Artifact"]:
        """The artifact a past run was made under: the live file if the hash matches, else the copy
        `install` kept under machinery/history/. None when neither exists."""
        d = layout.machinery_dir
        live = d / ARTIFACT_FILE
        if live.exists() and file_hash(live) == artifact_hash:
            return _load(live, Artifact)
        digest = artifact_hash.split(":")[-1][:16]
        kept = d / "history" / f"{ARTIFACT_FILE}.{digest}"
        return _load(kept, Artifact) if kept.exists() else None

    def drafting_fingerprint(self) -> str:
        """What produced the drafting pool: the markers, the nomination prompt
        and the eval-production prompts. If this changes after a blind audit,
        that audit no longer describes the pool."""
        from ..log.hashing import canonical_json, sha256_hex
        a, e = self.artifact, self.kernel.eval
        body = {"markers": [m.model_dump() for m in a.markers],
                "nomination": [a.nomination.model, a.nomination.prompt_version, a.nomination.prompt],
                "eval": {k: [getattr(e, k).tier, getattr(e, k).prompt_version, getattr(e, k).prompt]
                         for k in ("drafting", "noise", "seeded", "dedupe")}}
        return sha256_hex(canonical_json(body).encode())

    @staticmethod
    def install(layout: StoreLayout, artifact_src: str | Path, kernel_src: str | Path) -> list[str]:
        """Replace the store's machinery files with the consumer's, keeping each
        replaced file under machinery/history/ named by its hash. Validates first."""
        Machinery.from_files(artifact_src, kernel_src)
        d = layout.machinery_dir
        (d / "history").mkdir(parents=True, exist_ok=True)
        archived = []
        for src, name in ((artifact_src, ARTIFACT_FILE), (kernel_src, KERNEL_FILE)):
            live = d / name
            if live.exists() and live.read_bytes() != Path(src).read_bytes():
                digest = file_hash(live).split(":")[1][:16]
                shutil.copyfile(live, d / "history" / f"{name}.{digest}")
                archived.append(f"{name}.{digest}")
            shutil.copyfile(src, live)
        return archived

    @staticmethod
    def seed_defaults(layout: StoreLayout) -> bool:
        """Install Recant's default machinery if the store has none: host-neutral prompts that work out of the box,
        real model IDs, reasoning off, the default budgets (recant/defaults/). Copy the two files to change anything."""
        return Machinery.seed(layout, DEFAULT_ARTIFACT, DEFAULT_KERNEL)

    @staticmethod
    def seed(layout: StoreLayout, artifact_src: str | Path, kernel_src: str | Path) -> bool:
        """Install the consumer's two files if the store has none. Validates first."""
        Machinery.from_files(artifact_src, kernel_src)
        d = layout.machinery_dir
        d.mkdir(parents=True, exist_ok=True)
        seeded = False
        for src, name in ((artifact_src, ARTIFACT_FILE), (kernel_src, KERNEL_FILE)):
            if not (d / name).exists():
                shutil.copyfile(src, d / name)
                seeded = True
        return seeded

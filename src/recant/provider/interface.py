"""The provider interface, in this library's own shape. An adapter maps a
vendor onto it; nothing outside an adapter module imports a vendor SDK."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Literal, Optional, Protocol, runtime_checkable

MAX_CACHE_BREAKPOINTS = 4


class Tier(IntEnum):
    T1 = 1  # mechanical text: nominating, condensing, judging
    T2 = 2  # default reading and most work
    T3 = 3  # heavy reasoning; the first full extraction pass


@dataclass(frozen=True)
class ContentBlock:
    kind: Literal["text", "tool_use", "tool_result"]
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    is_error: bool = False
    cache: bool = False  # place a cache breakpoint after this block
    cache_ttl: Literal["5m", "1h"] = "5m"


def text(value: str, *, cache: bool = False, ttl: Literal["5m", "1h"] = "5m") -> ContentBlock:
    return ContentBlock(kind="text", text=value, cache=cache, cache_ttl=ttl)


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = True


@dataclass(frozen=True)
class ModelRequest:
    job: str
    messages: tuple[Message, ...]
    max_tokens: int
    tier: Optional[Tier] = None
    model: Optional[str] = None
    system: tuple[ContentBlock, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    output_schema: Optional[dict[str, Any]] = None
    effort: Optional[Literal["low", "medium", "high"]] = None
    custom_id: Optional[str] = None
    reasoning: Optional[Literal["off", "on"]] = None   # the model's own reasoning before it answers; None is the model's default
    session: Optional[str] = None   # the host conversation this call belongs to: logged on the call, never sent

    def __post_init__(self) -> None:
        if (self.tier is None) == (self.model is None):
            raise ValueError("a request names a tier or a model, exactly one")
        if self.tools and self.output_schema is not None:
            raise ValueError("tools and output_schema are mutually exclusive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not self.messages:
            raise ValueError("a request has at least one message")
        marks = sum(b.cache for b in self.system) + sum(b.cache for m in self.messages for b in m.content)
        if marks > MAX_CACHE_BREAKPOINTS:
            raise ValueError(f"at most {MAX_CACHE_BREAKPOINTS} cache breakpoints per request")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = int(self.tier) if self.tier is not None else None
        d.pop("session")              # metering, not request content: the request hash does not move with it
        return d


@dataclass(frozen=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0            # the part of `output` the model spent reasoning; 0 where a provider reports none
    cache_write_long: int = 0     # the part of `cache_write` written at the long cache lifetime; 0 where none


@dataclass(frozen=True)
class Prices:
    """One model's row of the consumer's price table, in USD per million tokens. Optional: with no table the
    library reports tokens and no dollars. `cache_read` and `cache_write` default to the input price (no discount
    or surcharge is assumed); `batch_factor` multiplies a batch call's whole cost."""
    input: float
    output: float
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None
    batch_factor: float = 1.0
    cache_write_long: Optional[float] = None   # a write at the long cache lifetime; defaults to `cache_write`
    above: Optional[int] = None                # a model priced by prompt length: a request whose prompt (input,
    over: Optional["Prices"] = None            # cache reads and cache writes) is over `above` tokens pays `over`

    FIELDS = ("input", "output", "cache_read", "cache_write", "cache_write_long", "batch_factor", "above", "over")

    @classmethod
    def from_row(cls, row: Any, *, model: str = "?") -> "Prices":
        """A row of a JSON price table, checked: a plain error naming the model and the field, never a traceback."""
        if not isinstance(row, dict):
            raise ValueError(f"model {model!r}: a row is an object such as {{\"input\": 2.0, \"output\": 10.0}}, not {type(row).__name__}")
        unknown = [k for k in row if k not in cls.FIELDS]
        if unknown:
            raise ValueError(f"model {model!r}: unknown field {unknown[0]!r}; the fields are {', '.join(cls.FIELDS)} "
                             "(all prices in USD per million tokens)")
        import math
        for k in ("input", "output"):
            if row.get(k) is None:
                raise ValueError(f"model {model!r}: {k!r} is required (USD per million tokens)")
        for k in ("input", "output", "cache_read", "cache_write", "cache_write_long", "batch_factor"):
            if k not in row or (row[k] is None and k in ("cache_read", "cache_write", "cache_write_long")):
                continue                                          # an optional price left out or null: its default
            v = row[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                what = "a factor" if k == "batch_factor" else "a number of USD per million tokens"
                raise ValueError(f"model {model!r}: {k!r} is {what}, not {v!r}")
        if ("above" in row) != ("over" in row):
            raise ValueError(f"model {model!r}: 'above' (a prompt length in tokens) and 'over' (the row for prompts over it) go together")
        over = None
        if "over" in row:
            if not isinstance(row["above"], int) or isinstance(row["above"], bool) or row["above"] <= 0:
                raise ValueError(f"model {model!r}: 'above' is a whole number of prompt tokens, not {row['above']!r}")
            if isinstance(row["over"], dict) and ("above" in row["over"] or "over" in row["over"]):
                raise ValueError(f"model {model!r}: 'over' has one more card, not another split")
            over = cls.from_row(row["over"], model=model)
        return cls(**{k: v for k, v in row.items() if k != "over"}, over=over)

    def at(self, prompt: int) -> "Prices":
        """The card a request with a prompt of this many tokens pays."""
        return self.over if (self.above is not None and self.over is not None and prompt > self.above) else self

    def cost_usd(self, usage: "Usage", *, batch: bool = False) -> float:
        card = self.at(usage.input + usage.cache_read + usage.cache_write)
        if card is not self:
            return card.cost_usd(usage, batch=batch)
        read = self.cache_read if self.cache_read is not None else self.input
        write = self.cache_write if self.cache_write is not None else self.input
        write_long = self.cache_write_long if self.cache_write_long is not None else write
        long_ = min(usage.cache_write_long, usage.cache_write)
        usd = (usage.input * self.input + (usage.cache_write - long_) * write + long_ * write_long
               + usage.cache_read * read + usage.output * self.output) / 1_000_000
        return usd * (self.batch_factor if batch else 1.0)


@dataclass(frozen=True)
class ModelResponse:
    model_id: str
    content: tuple[ContentBlock, ...]
    stop_reason: Optional[str]
    usage: Usage
    cost_usd: Optional[float]            # None when the consumer passed no price for the model
    request_id: Optional[str] = None
    latency_ms: Optional[int] = None
    billed_usd: Optional[float] = None   # the provider's billed figure, kept apart from `cost_usd` (computed)

    @property
    def text(self) -> str:
        return "".join(b.text or "" for b in self.content if b.kind == "text")


class ProviderError(RuntimeError):
    def __init__(self, kind: str, message: str, *, retryable: bool = False):
        super().__init__(f"{kind}: {message}")
        self.kind, self.retryable = kind, retryable


@dataclass(frozen=True)
class BatchHandle:
    batch_id: str
    job: str
    requests: tuple[dict[str, Any], ...] = field(default_factory=tuple)  # custom_id, model, tier, request_hash


@dataclass(frozen=True)
class Completion:
    """What a model client returns for one request: the content and the usage, and the two things the transport
    alone knows — why the model stopped, and the vendor's id for the request."""
    content: tuple[ContentBlock, ...]
    stop_reason: Optional[str]
    usage: Usage
    request_id: Optional[str] = None
    billed_usd: Optional[float] = None   # what the provider billed for the call, when it says


@runtime_checkable
class ModelClient(Protocol):
    """The one thing a consumer implements to give this library a model. `complete` receives a
    request whose `model` is already resolved and returns a `Completion`, or raises `ProviderError`.

    Three further capabilities are optional and detected by presence, never required:
      `submit(requests) -> handle` and `poll(handle) -> results | None` — a batch path. `handle` is a string the
        client can find the batch by; `poll` returns None while the batch runs, then an iterable of
        `(custom_id, Completion | ProviderError)` in any order. Used only when a job asks for batch and both exist;
        otherwise the same requests are sent one at a time through `complete`.
      `count_tokens(request) -> int` — the vendor's own count of a request's input.
    The client takes no ceiling and no price: it reports usage, and the caller does what it likes with it."""

    def complete(self, request: ModelRequest) -> Completion: ...


def has_batch(client: Any) -> bool:
    return callable(getattr(client, "submit", None)) and callable(getattr(client, "poll", None))

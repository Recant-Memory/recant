"""What Recant serves, counted. The served memory's size is the lever's price tag, so it is
counted with the provider's own token counter where the model client has one, and estimated from characters where
it has not; the record says which."""
from __future__ import annotations

import hashlib
import threading
from dataclasses import asdict
from typing import Callable, Optional

from .log.hashing import canonical_json

from .provider import Message, ModelRequest, ProviderError, text

_PROBE = "."


def _count(provider, model: str, body: str) -> int:
    return provider.count_tokens(ModelRequest(job="count", model=model, messages=(Message("user", (text(body),)),), max_tokens=1))


def served_tokens(provider, model: str, parts: dict[str, str], *, estimate: Callable[[str], int],
                  baseline: Optional[dict] = None) -> tuple[dict[str, int], bool]:
    """Tokens of each named part (`{"core": text, "slice": text}`), and whether they were counted. A part's count is
    the count of a message holding it less the count of a message holding one character, so the message framing is
    not charged to memory. `baseline` caches that framing per model across calls. An empty part is 0 uncounted."""
    try:
        base = (baseline or {}).get(model)
        if base is None:
            base = _count(provider, model, _PROBE)
            if baseline is not None:
                baseline[model] = base
        return {k: (max(0, _count(provider, model, v) - base) if v else 0) for k, v in parts.items()}, True
    except (ProviderError, AttributeError):
        return {k: (estimate(v) if v else 0) for k, v in parts.items()}, False


def segment_hashes(req) -> list[dict]:
    """A hash of everything up to each cache breakpoint, in the order the cache reads a request:
    tools, then system blocks, then message blocks. Cache markers are left out of the bytes hashed, as the cache
    leaves them out of a prefix match. Two calls whose segment hashes agree up to a breakpoint can share that cache
    entry; where they first differ is the segment that moved. Hashes only: no text is kept."""
    h = hashlib.sha256(canonical_json([asdict(t) for t in req.tools]).encode())
    out = []

    def add(at: str, block) -> None:
        d = asdict(block)
        d.pop("cache", None)
        d.pop("cache_ttl", None)
        h.update(canonical_json(d).encode())
        if block.cache:
            out.append({"at": at, "hash": h.copy().hexdigest()[:16], "ttl": block.cache_ttl})
    for i, b in enumerate(req.system):
        add(f"system.{i}", b)
    for i, m in enumerate(req.messages):
        h.update(m.role.encode())
        for j, b in enumerate(m.content):
            add(f"messages.{i}.{j}", b)
    return out


def target_signature(targets) -> list[str]:
    """What a tool call touches, hashed: the consumer names the targets (the files and programs a
    command touches, a page, a repository path); Recant keeps a short hash of each, never the text, so a later call on
    the same target can be found without the command."""
    return sorted({hashlib.sha256(str(t).strip().encode()).hexdigest()[:12] for t in targets if str(t).strip()})


def setting_fingerprint(req) -> str:
    """What a job's request is set to, apart from its content: the model, effort, output cap,
    reasoning setting and output schema, the tools, and the first system block — a job's instructions. Later system
    blocks (served memory) and the messages are data, not settings. A new fingerprint for a job starts its window."""
    first = req.system[0] if req.system else None
    head = None
    if first is not None:
        d = asdict(first)
        d.pop("cache", None)
        d.pop("cache_ttl", None)
        head = d
    body = {"model": req.model, "effort": req.effort, "max_tokens": req.max_tokens, "reasoning": req.reasoning,
            "schema": req.output_schema, "tools": [asdict(t) for t in req.tools], "instructions": head}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()[:16]


class Meter:
    """Metering for a host that keeps its own model client: after each call, the host hands over
    what it sent and what came back, and the same model_call record Recant's own provider writes goes into the log —
    the session, the setting fingerprint, whether the request only appends to the previous one in its session. One
    log run per host run (`run()`); nothing is sent anywhere. Request fields are the host's own (model, system, tools,
    messages as plain data); only hashes and counts are kept."""

    THREADS_KEPT = 4096                         # recent requests remembered for threading calls that name no session

    def __init__(self, layout, tenant: str, *, key_label: Optional[str] = None, threads: bool = False):
        """`threads`: give a call that names no session the conversation it continues, found by content (recant.wrap's
        meter for calls outside any client.run, where many conversations may be in flight at once)."""
        self.layout, self.tenant, self.key_label = layout, tenant, key_label
        self.log = None
        self._last: dict = {}
        self._threads = threads
        self._seen: dict = {}                   # the hash of a request's messages -> the conversation it belongs to
        self._tip: dict = {}                    # a conversation -> the hash of its latest request
        self._n = 0
        self._lock = threading.Lock()          # a host's threads may share a meter: the log is opened once

    def run(self, job: str, **params):
        from .log import LogWriter
        if self.log is not None:
            self.log.finish()
        self.log = LogWriter(self.layout, self.tenant, job, artifact_hash="host", kernel_hash="host", params=params)
        self._last = {}
        return self.log

    def close(self) -> None:
        if self.log is not None:
            from . import coverage
            coverage.flush_into(self.log)              # what went past Recant while this run was open
            self.log.finish()
            self.log = None

    def record(self, *, job: str, model: str, usage: dict, session: Optional[str] = None, system=None, tools=None,
               messages=None, max_tokens: Optional[int] = None, effort=None, reasoning=None, latency_ms: Optional[int] = None,
               stop_reason: Optional[str] = None, error: Optional[str] = None, cost_usd: Optional[float] = None,
               ttft_ms: Optional[int] = None, overhead_ms: Optional[float] = None, bench_task: Optional[str] = None,
               via: Optional[str] = None, billing: Optional[str] = None, calls: Optional[int] = None,
               billed_usd: Optional[float] = None) -> dict:
        """`usage`: input (uncached), output, cache_read, cache_write, and optionally cache_write_long and reasoning.
        `via`, `billing`, `calls`: a tool's own totals (`from_claude_code`, `from_agent_sdk`)."""
        msgs = [canonical_json(m) for m in (messages or [])]
        if self._threads and session is None and msgs:
            session, extends = self._thread(job, msgs)
        else:
            prev = self._last.get((job, session))
            extends = None if prev is None else (len(msgs) > len(prev) and msgs[:len(prev)] == prev)
            self._last[(job, session)] = msgs
        body = {"model": model, "effort": effort, "max_tokens": max_tokens, "reasoning": reasoning, "schema": None,
                "tools": tools or [], "instructions": system}
        fp = hashlib.sha256(canonical_json(body).encode()).hexdigest()[:16]
        u = {k: int(usage.get(k) or 0) for k in ("input", "output", "cache_read", "cache_write", "reasoning", "cache_write_long")}
        rec = dict(model_id=model, job=job, request_hash="sha256:" + hashlib.sha256("".join(msgs).encode()).hexdigest(),
                   usage=u, cost_usd=cost_usd, session=session, setting_fp=fp, key=self.key_label, extends_previous=extends,
                   latency_ms=latency_ms, stop_reason=stop_reason, error=error, ttft_ms=ttft_ms, overhead_ms=overhead_ms,
                   bench_task=bench_task)
        rec.update({k: v for k, v in (("via", via), ("billing", billing), ("calls", calls), ("billed_usd", billed_usd)) if v is not None})
        with self._lock:
            if self.log is None:
                self.run("host")
            log = self.log
        log.emit("model_call", **rec)
        return rec

    def _thread(self, job: str, msgs: list) -> tuple:
        """(the conversation, whether this request extends an earlier one of it): the longest earlier request of the
        job whose messages are a strict prefix of these continues; else a new conversation starts. This is how the cache
        itself finds a prefix, so concurrent conversations are told apart however their calls interleave."""
        h, cum = hashlib.sha256(job.encode()), []
        for m in msgs:
            h.update(b"\x00" + m.encode())
            cum.append(h.hexdigest()[:24])
        with self._lock:
            hit = next((k for k in reversed(cum[:-1]) if k in self._seen), None)
            conv = self._seen.get(hit) if hit else None
            extends = conv is not None and self._tip.get(conv) == hit   # it continues that conversation's latest request
            if not extends:                                  # new, or a branch off an earlier point of one: its own
                self._n += 1                                 # conversation, which reuses nothing it cannot be sure of
                conv = f"auto:{cum[0][:8]}-{self._n}"
            self._seen[cum[-1]] = conv
            self._tip[conv] = cum[-1]
            for d in (self._seen, self._tip):                # bounded: the oldest go first
                if len(d) > self.THREADS_KEPT:
                    for k in list(d)[: len(d) - self.THREADS_KEPT]:
                        del d[k]
        return conv, extends

    def tool(self, *, name: str, session: Optional[str] = None, input: Optional[dict] = None, bytes_fetched: Optional[int] = None,
             bytes_to_model: Optional[int] = None, error: Optional[str] = None, targets=(), duration_ms: Optional[int] = None) -> dict:
        """One tool call the host ran: what it fetched and what reached the model, in bytes, and the targets it
        touched, hashed. Tool-output trimming and repeated-call loops are sized from these."""
        rec = dict(session=session or "", name=name, input=dict(input or {}), returned_chars=int(bytes_to_model or 0),
                   bytes_fetched=bytes_fetched, bytes_to_model=bytes_to_model, error=error, duration_ms=duration_ms,
                   targets=target_signature(targets) or None)
        with self._lock:
            if self.log is None:
                self.run("host")
            log = self.log
        log.emit("tool_call", **rec)
        return rec


# ---------------------------------------------------------------- tools that make their own calls
def _g(o, *names, default=None):
    """A field of a dict or an object, under the first of its names present (the CLI's JSON is camelCase, the Python
    SDK's dataclasses snake_case)."""
    for n in names:
        v = o.get(n) if isinstance(o, dict) else getattr(o, n, None)
        if v is not None:
            return v
    return default


def _model_usage_rows(model_usage) -> list[tuple[str, dict, Optional[float]]]:
    out = []
    for model, u in (model_usage or {}).items():
        n = lambda *k: int(_g(u, *k, default=0) or 0)                                  # noqa: E731
        long_ = n("cacheCreationEphemeral1hInputTokens", "cache_creation_ephemeral_1h_input_tokens")
        out.append((model, {"input": n("inputTokens", "input_tokens"), "output": n("outputTokens", "output_tokens"),
                            "cache_read": n("cacheReadInputTokens", "cache_read_input_tokens"),
                            "cache_write": n("cacheCreationInputTokens", "cache_creation_input_tokens"),
                            "cache_write_long": long_, "reasoning": n("thinkingTokens", "thinking_tokens")},
                    _g(u, "costUSD", "cost_usd")))
    return out


def _message_counts(messages) -> dict[str, int]:
    """Distinct API calls per model in a stream of assistant messages: one per message id (a message streamed in
    several parts repeats its id)."""
    ids: dict = {}
    for m in messages or ():
        msg = _g(m, "message", default=m)
        mid, model = _g(msg, "id", "message_id"), _g(msg, "model")
        if mid and model:
            ids.setdefault(model, set()).add(mid)
    return {k: len(v) for k, v in ids.items()}


def from_tool_result(result, meter: "Meter", *, via: str, job: Optional[str] = None, billing: Optional[str] = None,
                     session: Optional[str] = None, messages=None, bench_task: Optional[str] = None) -> list[dict]:
    """One model_call record per model from a tool's own end-of-run totals: Claude Code's `--output-format json`
    result, or the Claude Agent SDK's ResultMessage (`from_claude_code`, `from_agent_sdk`). The per-model totals
    (`modelUsage`) are the tool's; the per-step usage on its assistant messages repeats input and cache counts but its
    output count is a placeholder, so the steps give only how many calls there were (`messages`, optional). The tool's
    cost figure is its own estimate at list price; on a subscription (`billing="subscription"`) the billed figure is 0.
    These records are metered and advised on, never changed: the tool manages its own requests and caching."""
    rows = _model_usage_rows(_g(result, "modelUsage", "model_usage"))
    if not rows:                                                     # an older tool: one total, no per-model split
        u = _g(result, "usage", default={}) or {}
        n = lambda *k: int(_g(u, *k, default=0) or 0)                                  # noqa: E731
        model = _g(result, "model", default="unknown")
        rows = [(model, {"input": n("input_tokens"), "output": n("output_tokens"), "cache_read": n("cache_read_input_tokens"),
                         "cache_write": n("cache_creation_input_tokens")}, _g(result, "total_cost_usd"))]
    counts = _message_counts(messages)
    turns = _g(result, "num_turns")
    session = session or _g(result, "session_id")
    err = None if not _g(result, "is_error", default=False) else str(_g(result, "subtype", default="error"))
    out = []
    for model, usage, usd in rows:
        n = counts.get(model) or (int(turns) if turns and len(rows) == 1 else None)
        out.append(meter.record(job=job or via, model=model, usage=usage, session=session, cost_usd=usd,
                                latency_ms=_g(result, "duration_ms"), stop_reason=_g(result, "stop_reason"), error=err,
                                via=via, billing=billing, calls=n, billed_usd=0.0 if billing == "subscription" else None,
                                bench_task=bench_task))
    return out


def from_claude_code(result, meter: "Meter", **kw) -> list[dict]:
    """`claude -p ... --output-format json` (a dict, or its text), or the `result` line of `--output-format
    stream-json` with the stream's assistant lines as `messages=`."""
    import json as _json
    if isinstance(result, (str, bytes)):
        result = _json.loads(result)
    return from_tool_result(result, meter, via="claude-code", **kw)


def from_agent_sdk(result, meter: "Meter", **kw) -> list[dict]:
    """The Claude Agent SDK's ResultMessage, with the run's AssistantMessages as `messages=` if kept."""
    return from_tool_result(result, meter, via="claude-agent-sdk", **kw)

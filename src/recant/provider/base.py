"""The one path every model call takes: resolve the model, send, log the usage — and the cost, when the consumer
passed a price table. A `ModelClient` supplies only the transport. Nothing here refuses a call: this library takes
no spend or time limit of any kind; a consumer that wants one wraps the client or subclasses this."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, Iterator, Optional

from ulid import ULID

from ..metering import segment_hashes, setting_fingerprint
from ..log.hashing import request_hash
from ..machinery import Machinery
from .interface import BatchHandle, Completion, ModelRequest, ModelResponse, Prices, ProviderError, Usage, has_batch

Raw = Completion      # the name the transports and the tests have used since the first adapter


class BaseProvider:
    """Wraps a `ModelClient`: resolves the model, sends, logs tokens (and dollars, given `prices`). A subclass may
    override the five transport hooks instead of passing a client (the fake does)."""

    prices: Optional[dict[str, Prices]] = None      # a class-wide default table; the `prices` argument overrides it
    retries: int = 3                                # a retryable ProviderError is tried again this many times,
    backoff_s: float = 2.0                          # after 2, 4, 8 s: a stalled stream, a 429, a 5xx, a dropped connection
    sleep = staticmethod(time.sleep)

    def __init__(self, machinery: Machinery, log, client: Any = None, *, prices: Optional[dict[str, Prices]] = None,
                 key_label: Optional[str] = None):
        self.machinery, self.log, self.client = machinery, log, client
        self.key_label = key_label         # which of the consumer's keys the calls spend, as a label
        if prices is not None:
            self.prices = prices
        self._polled: dict[str, Any] = {}

    # transport: the client's, unless a subclass supplies its own -------------
    def _send(self, model: str, req: ModelRequest) -> Raw:
        return self.client.complete(replace(req, model=model, tier=None))

    def _count_tokens(self, model: str, req: ModelRequest) -> int:
        count = getattr(self.client, "count_tokens", None)
        if not callable(count):
            raise ProviderError("no_count_tokens", "this model client cannot count tokens")
        return count(replace(req, model=model, tier=None))

    @property
    def can_batch(self) -> bool:
        """True when the client has `submit` and `poll`, or a subclass supplies the batch hooks itself."""
        return has_batch(self.client) or type(self)._submit is not BaseProvider._submit

    def _submit(self, items: list[tuple[str, str, ModelRequest]]) -> str:
        if not has_batch(self.client):
            raise ProviderError("no_batch", "this model client has no batch path")
        return self.client.submit([replace(req, model=model, tier=None, custom_id=cid) for cid, model, req in items])

    def _status(self, batch_id: str) -> str:
        if batch_id not in self._polled:
            got = self.client.poll(batch_id)
            if got is None:
                return "running"
            self._polled[batch_id] = got
        return "ended"

    def _results(self, batch_id: str) -> Iterator[tuple[str, Raw | ProviderError]]:
        got = self._polled.pop(batch_id, None)
        if got is None:
            got = self.client.poll(batch_id)
        if got is None:
            raise ProviderError("batch_running", f"{batch_id} has not ended")
        yield from got

    # arithmetic -------------------------------------------------------------
    def resolve(self, req: ModelRequest) -> str:
        """A request's model if it names one, else the routing table's entry for
        its tier. Deterministic: there is no randomised tail here."""
        return req.model if req.model is not None else self.machinery.artifact.routing[int(req.tier)]

    def cost_usd(self, model: str, usage: Usage, *, batch: bool = False) -> Optional[float]:
        """Dollars for one call, or None: no price table, or no row in it for this model."""
        row = (self.prices or {}).get(model)
        return row.cost_usd(usage, batch=batch) if row is not None else None

    def reported(self, model: str, usage: Usage, cost_usd: Optional[float], *, batch: bool) -> None:
        """Called once for every call that came back, after it is priced and before it is logged. The library does
        nothing here; it is where a consumer's subclass meters."""

    # calls --------------------------------------------------------------------
    def call(self, req: ModelRequest) -> ModelResponse:
        model = self.resolve(req)
        tier = int(req.tier) if req.tier is not None else None
        started = time.monotonic()
        rhash = request_hash(req.to_dict())
        segs, sfp = segment_hashes(req), setting_fingerprint(replace(req, model=model, tier=None))
        attempt = 0
        while True:
            try:
                raw = self._send(model, req)
                break
            except ProviderError as exc:                # each failed attempt is logged: a stalled stream may be billed
                self.log.emit("model_call", key=self.key_label, model_id=model, tier=tier, job=req.job, request_hash=rhash,
                              usage=_usage_dict(Usage()), cost_usd=self._zero(model), custom_id=req.custom_id, session=req.session,
                              latency_ms=_ms(started), error=str(exc), segments=segs, setting_fp=sfp)
                if not exc.retryable or attempt >= self.retries:
                    raise
                self.sleep(self.backoff_s * 2 ** attempt)
                attempt += 1
                started = time.monotonic()
        return self._settle(model, tier, req.job, rhash, raw, custom_id=req.custom_id, latency_ms=_ms(started), session=req.session,
                            segments=segs, setting_fp=sfp)

    def count_tokens(self, req: ModelRequest) -> int:
        return self._count_tokens(self.resolve(req), req)

    def submit_batch(self, reqs: list[ModelRequest]) -> BatchHandle:
        if not reqs:
            raise ValueError("an empty batch")
        items, meta = [], []
        for req in reqs:
            model = self.resolve(req)
            cid = req.custom_id or f"r-{ULID()}"
            items.append((cid, model, req))
            meta.append({"custom_id": cid, "model": model, "job": req.job,
                         "tier": int(req.tier) if req.tier is not None else None,
                         "request_hash": request_hash(req.to_dict()), "session": req.session, "segments": segment_hashes(req),
                         "setting_fp": setting_fingerprint(replace(req, model=model, tier=None))})
        if len({m["custom_id"] for m in meta}) != len(meta):
            raise ValueError("custom ids in a batch must be unique")
        batch_id = self._submit(items)
        handle = BatchHandle(batch_id=batch_id, job=reqs[0].job, requests=tuple(meta))
        self.log.batches_dir.mkdir(parents=True, exist_ok=True)
        (self.log.batches_dir / f"{batch_id}.json").write_text(
            json.dumps({"batch_id": batch_id, "job": handle.job, "run": self.log.run,
                        "requests": meta}, indent=1) + "\n")
        return handle

    @staticmethod
    def load_handle(path) -> BatchHandle:
        data = json.loads(open(path).read())
        return BatchHandle(batch_id=data["batch_id"], job=data["job"], requests=tuple(data["requests"]))

    def batch_status(self, handle: BatchHandle) -> str:
        return self._status(handle.batch_id)

    def batch_results(self, handle: BatchHandle) -> Iterator[tuple[str, ModelResponse | ProviderError]]:
        """Keyed by custom id; results arrive in any order. Priced as read, at batch rates."""
        meta = {m["custom_id"]: m for m in handle.requests}
        for cid, raw in self._results(handle.batch_id):
            m = meta.get(cid)
            if m is None:
                raise ProviderError("unknown_custom_id", f"{cid!r} is not in this batch")
            if isinstance(raw, ProviderError):
                self.log.emit("model_call", key=self.key_label, model_id=m["model"], tier=m["tier"], job=m["job"],
                              request_hash=m["request_hash"], usage=_usage_dict(Usage()), cost_usd=self._zero(m["model"]),
                              batch_id=handle.batch_id, custom_id=cid, error=str(raw), session=m.get("session"))
                yield cid, raw
                continue
            try:
                yield cid, self._settle(m["model"], m["tier"], m["job"], m["request_hash"], raw,
                                        custom_id=cid, batch_id=handle.batch_id, session=m.get("session"),
                                        segments=m.get("segments"), setting_fp=m.get("setting_fp"))
            except ProviderError as exc:
                yield cid, exc

    def _zero(self, model: str) -> Optional[float]:
        """What a call that never came back cost: nothing, said in dollars only when this model has a price."""
        return 0.0 if model in (self.prices or {}) else None

    def _settle(self, model: str, tier: Optional[int], job: str, rhash: str, raw: Raw, *,
                custom_id: Optional[str] = None, batch_id: Optional[str] = None,
                latency_ms: Optional[int] = None, session: Optional[str] = None,
                segments: Optional[list] = None, setting_fp: Optional[str] = None) -> ModelResponse:
        cost = self.cost_usd(model, raw.usage, batch=batch_id is not None)
        self.log.add_cost(cost)
        self.reported(model, raw.usage, cost, batch=batch_id is not None)
        refused = raw.stop_reason == "refusal"
        self.log.emit("model_call", key=self.key_label, model_id=model, tier=tier, job=job, request_hash=rhash,
                      usage=_usage_dict(raw.usage), cost_usd=round(cost, 8) if cost is not None else None, batch_id=batch_id,
                      custom_id=custom_id, stop_reason=raw.stop_reason, request_id=raw.request_id,
                      latency_ms=latency_ms, error="refusal" if refused else None, session=session, segments=segments,
                      setting_fp=setting_fp,
                      billed_usd=getattr(raw, "billed_usd", None))
        if refused:
            raise ProviderError("refusal", "the model declined; there is no answer to read")
        return ModelResponse(model_id=model, content=raw.content, stop_reason=raw.stop_reason,
                             usage=raw.usage, cost_usd=cost, request_id=raw.request_id,
                             latency_ms=latency_ms, billed_usd=getattr(raw, "billed_usd", None))


def _usage_dict(u: Usage) -> dict[str, int]:
    return {"input": u.input, "output": u.output, "cache_read": u.cache_read, "cache_write": u.cache_write, "reasoning": u.reasoning,
            "cache_write_long": u.cache_write_long}


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)

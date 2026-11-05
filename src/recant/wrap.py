"""`recant.wrap(client, store=..., tenant=...)`: the drop-in path for a host written on the raw
Anthropic SDK, sync or async. The wrapped object has the SDK's `messages` interface — `create`, `create(stream=True)`,
the `stream(...)` helper and `count_tokens` — on `Anthropic` and on `AsyncAnthropic` alike. Every call is metered into
Recant's log (`recant.metering.Meter`), the overlay's safe levers are carried out on the request (`recant.levers.apply`),
and the request sent is checked to differ from the one built only in cache markers (`recant.quality.equivalent`) —
a difference is logged and the host's own request is sent instead. A streamed call is recorded from its final message's
usage once the stream ends or is closed (a stream closed early is recorded with what it reported, stop reason empty).
Thinking tokens are recorded as reasoning. Nothing leaves the host but the request itself.

    import anthropic, recant
    client = recant.wrap(anthropic.Anthropic(), store="store", tenant="t1")
    with client.run("agent", session="chat-1"):                      # one log run per host run; the session groups its calls
        r = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, system="...", messages=[...])

    aclient = recant.wrap(anthropic.AsyncAnthropic(), store="store", tenant="t1")
    async with aclient.run("agent", session="chat-2"):               # each asyncio task its own run, concurrently
        async with aclient.messages.stream(model=..., max_tokens=..., messages=[...]) as s:
            msg = await s.get_final_message()

Then `recant cost --store store --tenant t1` prices the calls and, with `--nightly`, applies what pays; the next call
carries it out. The host shape the advisor needs is written to the store once, where `recant cost` finds it."""
from __future__ import annotations

import inspect
import time
from contextvars import ContextVar
from typing import Any, Optional

from . import costshape, coverage, levers, quality, switch
from .metering import Meter
from .store import StoreLayout


def _get(o, k, default=None):
    """A field of an SDK object or of the same thing as a dict (raw stream events arrive as either in tests)."""
    if o is None:
        return default
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _usage(u) -> dict:
    """The SDK's usage, in Recant's names: thinking tokens as reasoning, the 1-hour part of the cache writes as long."""
    n = lambda v: int(v or 0)                                                       # noqa: E731
    return {"input": n(_get(u, "input_tokens")), "output": n(_get(u, "output_tokens")),
            "cache_read": n(_get(u, "cache_read_input_tokens")), "cache_write": n(_get(u, "cache_creation_input_tokens")),
            "reasoning": n(_get(_get(u, "output_tokens_details"), "thinking_tokens")),
            "cache_write_long": n(_get(_get(u, "cache_creation"), "ephemeral_1h_input_tokens"))}


def _reasoning(thinking) -> Optional[str]:
    """The request's thinking setting in Recant's terms: "off" when disabled, "on" when asked for, None when the host
    left it to the model's default (which, on the current models, thinks)."""
    kind = (thinking or {}).get("type")
    return None if kind is None else "off" if kind == "disabled" else "on"


def _markers(req: dict) -> list[str]:
    """Where the request carries cache markers, and at which lifetime: `system.0 (1h)`, `messages.5.0 (5m)`."""
    out = []
    groups = [("system", req.get("system"))] + [(f"messages.{i}", m.get("content")) for i, m in enumerate(req.get("messages") or [])]
    for at, blocks in groups:
        for j, b in enumerate(blocks if isinstance(blocks, list) else []):
            if isinstance(b, dict) and "cache_control" in b:
                out.append(f"{at}.{j} ({b['cache_control'].get('ttl', '5m')})")
    return out


class _Events:
    """What a raw event stream (`create(stream=True)`) says about the final message: the model and usage from
    `message_start`, then each `message_delta`'s usage (cumulative; a field it leaves out keeps its value) and stop
    reason — the same accumulation the SDK's own stream helper does."""

    def __init__(self):
        self.model, self.stop_reason, self.usage = None, None, {}

    def see(self, ev) -> None:
        kind = _get(ev, "type")
        if kind == "message_start":
            msg = _get(ev, "message")
            self.model = _get(msg, "model")
            self.usage = _usage(_get(msg, "usage"))
        elif kind == "message_delta":
            u = _get(ev, "usage")
            if u is not None:
                fresh = _usage(u)
                raw = {"input": "input_tokens", "output": "output_tokens", "cache_read": "cache_read_input_tokens",
                       "cache_write": "cache_creation_input_tokens", "reasoning": "output_tokens_details",
                       "cache_write_long": "cache_creation"}
                self.usage.update({k: v for k, v in fresh.items() if _get(u, raw[k]) is not None})
            self.stop_reason = _get(_get(ev, "delta"), "stop_reason") or self.stop_reason


class _Call:
    """One call in flight: what the host built, what was sent, its run, and the one record it gets."""

    def __init__(self, w: "Wrapped", kwargs: dict, *, apply: bool = True):
        t = time.perf_counter()
        self.w = w
        w._ready()
        built = dict(kwargs)
        self.session = built.pop("session", None) or w.session
        self.meter = w._meter()
        ov = costshape.overlay(w.layout, w.tenant) if (apply and w.apply_levers) else {}
        sent = levers.apply(built, ov) if ov else dict(built)
        self.equivalent = quality.equivalent(built, sent)
        if not self.equivalent:                                          # never send what the host did not build, bar the markers
            sent = dict(built)
        sent, self.opted = levers.apply_opted_in(sent, ov, w.job) if ov else (sent, [])   # gated levers: after the check
        w.last_equivalent, w.last_sent_markers = self.equivalent, _markers(sent)
        self.built, self.sent, self.done = built, sent, False
        self.ttft: Optional[float] = None                                # streams: seconds from the send to the first event
        self.inside = time.perf_counter() - t                            # Recant's own time: before the send, per event, after
        self.t0 = time.monotonic()

    def record(self, *, usage: Optional[dict] = None, model=None, stop_reason=None, error=None) -> None:
        if self.done:
            return
        arrived, t = time.monotonic(), time.perf_counter()
        self.done = True
        b = self.built
        kw = dict(job=self.w.job, model=str(model or b.get("model")), usage=usage or {}, session=self.session,
                  system=b.get("system"), tools=b.get("tools"), messages=b.get("messages"), max_tokens=b.get("max_tokens"),
                  effort=(self.sent.get("output_config") or {}).get("effort"), reasoning=_reasoning(self.sent.get("thinking")),
                  latency_ms=int((arrived - self.t0) * 1000), stop_reason=stop_reason, error=error,
                  ttft_ms=None if self.ttft is None else int(self.ttft * 1000), bench_task=self.w.task_tag)
        kw["overhead_ms"] = round((self.inside + time.perf_counter() - t) * 1000, 3)   # the record's own write excluded
        self.meter.record(**kw)
        self.meter.log.count("levers_equivalent" if self.equivalent else "levers_not_equivalent")
        for lid in self.opted:
            self.meter.log.count(f"opted_in:{lid}")

    def message(self, msg) -> None:
        self.record(usage=_usage(_get(msg, "usage")), model=_get(msg, "model"), stop_reason=_get(msg, "stop_reason"))

    def failed(self, exc: BaseException, *, usage: Optional[dict] = None, model=None) -> None:
        """A call that failed: the error, and whatever usage a stream had reported before it broke."""
        self.record(usage=usage, model=model, error=type(exc).__name__)

    def snapshot(self, stream, error: Optional[BaseException] = None) -> None:
        """The helper stream's message as accumulated so far: the final one when the host read it through."""
        try:
            msg = stream.current_message_snapshot
        except Exception:                                                # noqa: BLE001 — no event arrived before the close
            msg = None
        usage, model = (_usage(_get(msg, "usage")), _get(msg, "model")) if msg is not None else (None, None)
        if error is not None:
            self.failed(error, usage=usage, model=model)
        else:
            self.record(usage=usage, model=model, stop_reason=_get(msg, "stop_reason"))


# ---------------------------------------------------------------- create(stream=True): the raw event stream
class _RawStream:
    def __init__(self, inner, call: _Call):
        self._inner, self._call, self._ev = inner, call, _Events()

    def _see(self, ev):
        t = time.perf_counter()
        if self._call.ttft is None:
            self._call.ttft = time.monotonic() - self._call.t0
        self._ev.see(ev)
        self._call.inside += time.perf_counter() - t

    def __iter__(self):
        try:
            for ev in self._inner:
                self._see(ev)
                yield ev
        except Exception as exc:
            self._call.failed(exc, usage=self._ev.usage, model=self._ev.model)
            raise
        self._finish()

    def _finish(self):
        self._call.record(usage=self._ev.usage, model=self._ev.model, stop_reason=self._ev.stop_reason)

    def close(self):
        self._inner.close()
        self._finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getattr__(self, name):                                         # response, raw_events, …: the SDK's own
        return getattr(self._inner, name)


class _AsyncRawStream(_RawStream):
    def __iter__(self):
        raise TypeError("an async stream: use `async for`")

    async def __aiter__(self):
        try:
            async for ev in self._inner:
                self._see(ev)
                yield ev
        except Exception as exc:
            self._call.failed(exc, usage=self._ev.usage, model=self._ev.model)
            raise
        self._finish()

    async def close(self):
        await self._inner.close()
        self._finish()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


# ---------------------------------------------------------------- messages.stream(...): the SDK's helper
class _FirstEvent:
    """The helper's raw event stream with its first event timed (time to first token). The helper reads `_raw_stream`
    only when its own iteration starts, so the swap is made right after the request is opened; an SDK without that
    attribute is left alone and the call is recorded without a time to first token."""

    def __init__(self, inner, call: _Call):
        self._inner, self._call = inner, call

    def _first(self):
        if self._call.ttft is None:
            self._call.ttft = time.monotonic() - self._call.t0

    def __iter__(self):
        for ev in self._inner:
            self._first()
            yield ev

    async def __aiter__(self):
        async for ev in self._inner:
            self._first()
            yield ev

    def __getattr__(self, name):                                         # response, close, …: the SDK's own
        return getattr(self._inner, name)


def _time_first_event(stream, call: _Call) -> None:
    if hasattr(stream, "_raw_stream"):
        stream._raw_stream = _FirstEvent(stream._raw_stream, call)


class _StreamManager:
    def __init__(self, inner, call: _Call):
        self._inner, self._call, self._stream = inner, call, None

    def __enter__(self):
        try:
            with coverage.through():                                     # the request is sent here, through Recant
                self._stream = self._inner.__enter__()
        except Exception as exc:
            self._call.failed(exc)
            raise
        _time_first_event(self._stream, self._call)
        return self._stream

    def __exit__(self, *exc):
        out = self._inner.__exit__(*exc)
        self._end(exc[1])
        return out

    def _end(self, exc):
        if self._stream is None:                                         # the request itself failed: recorded in __enter__
            return
        try:
            finished = _get(self._stream.current_message_snapshot, "stop_reason") is not None
        except Exception:                                                # noqa: BLE001 — no event arrived
            finished = False
        self._call.snapshot(self._stream, error=exc if isinstance(exc, Exception) and not finished else None)


class _AsyncStreamManager(_StreamManager):
    async def __aenter__(self):
        try:
            with coverage.through():
                self._stream = await self._inner.__aenter__()
        except Exception as exc:
            self._call.failed(exc)
            raise
        _time_first_event(self._stream, self._call)
        return self._stream

    async def __aexit__(self, *exc):
        out = await self._inner.__aexit__(*exc)
        self._end(exc[1])
        return out


# ---------------------------------------------------------------- the two `messages` faces
class _Messages:
    def __init__(self, w: "Wrapped"):
        self._w = w

    def create(self, **kwargs) -> Any:
        mode = self._w.mode()
        if mode == "off":                                                # a pure pass-through: nothing metered or written
            return self._w.client.messages.create(**_host_request(kwargs))
        call = _Call(self._w, kwargs, apply=mode == "on")
        try:
            with coverage.through():
                out = self._w.client.messages.create(**call.sent)
        except Exception as exc:                                         # metered, then raised as it came
            call.failed(exc)
            raise
        if call.sent.get("stream"):
            return _RawStream(out, call)
        call.message(out)
        return out

    def stream(self, **kwargs) -> Any:
        mode = self._w.mode()
        if mode == "off":
            return self._w.client.messages.stream(**_host_request(kwargs))
        call = _Call(self._w, kwargs, apply=mode == "on")
        return _StreamManager(self._w.client.messages.stream(**call.sent), call)

    def count_tokens(self, **kwargs) -> Any:
        with coverage.through():
            return self._w.client.messages.count_tokens(**kwargs)


class _AsyncMessages(_Messages):
    async def create(self, **kwargs) -> Any:
        mode = self._w.mode()
        if mode == "off":
            return await self._w.client.messages.create(**_host_request(kwargs))
        call = _Call(self._w, kwargs, apply=mode == "on")
        try:
            with coverage.through():
                out = await self._w.client.messages.create(**call.sent)
        except Exception as exc:
            call.failed(exc)
            raise
        if call.sent.get("stream"):
            return _AsyncRawStream(out, call)
        call.message(out)
        return out

    def stream(self, **kwargs) -> Any:
        mode = self._w.mode()
        if mode == "off":
            return self._w.client.messages.stream(**_host_request(kwargs))
        call = _Call(self._w, kwargs, apply=mode == "on")
        return _AsyncStreamManager(self._w.client.messages.stream(**call.sent), call)


def _host_request(kwargs: dict) -> dict:
    """The host's request as the raw client takes it: Recant's own `session=` argument out, nothing else touched."""
    return {k: v for k, v in kwargs.items() if k != "session"}


def _is_async(client) -> bool:
    create = getattr(getattr(client, "messages", None), "create", None)
    return create is not None and inspect.iscoroutinefunction(inspect.unwrap(create))


# ---------------------------------------------------------------- the wrapper and its runs
class _Run:
    """One log run (a host run: a task, a conversation) with its own meter, so concurrent runs — asyncio tasks, threads —
    each keep their own log; `with` or `async with`."""

    def __init__(self, w: "Wrapped", job: str, session: Optional[str], params: dict):
        self.w, self.job, self.session, self.params, self.token = w, job, session, params, None

    def __enter__(self):
        meter = Meter(self.w.layout, self.w.tenant, key_label=self.w.key_label)   # its log opens on the first metered call:
        self.token = self.w._current.set((meter, self.session, self.job, self.params))   # a run with the switch off writes nothing
        return self.w

    def __exit__(self, *exc):
        meter = self.w._current.get()[0]
        self.w._current.reset(self.token)
        meter.close()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *exc):
        self.__exit__(*exc)


class Wrapped:
    def __init__(self, client, *, store, tenant: str, job: str = "agent", key_label: Optional[str] = None, apply_levers: bool = True,
                 watch: bool = True):
        self.client, self.tenant, self.job, self.key_label, self.apply_levers = client, tenant, job, key_label, apply_levers
        self.layout = store if isinstance(store, StoreLayout) else StoreLayout(store)
        self._readied = False                                            # the store is written on the first metered call
        self.meter = Meter(self.layout, tenant, key_label=key_label)       # calls made outside any run
        self.is_async = _is_async(client)
        self.messages = _AsyncMessages(self) if self.is_async else _Messages(self)
        self._current: ContextVar = ContextVar(f"recant_run_{id(self)}", default=None)
        self.last_equivalent: Optional[bool] = None
        self.last_sent_markers: list[str] = []                         # where the last request sent carried cache markers
        self.task_tag: Optional[str] = None                            # `recant bench` sets it per task and attempt
        if switch.state(self.layout) != "off":
            self._ready()
        if watch:                                                        # what goes past the wrapper is counted
            coverage.watch(self.layout, tenant)

    def mode(self) -> str:
        """The switch (`recant.switch`), read on every call: on, off or meter."""
        return switch.state(self.layout)

    def _ready(self) -> None:
        if not self._readied:
            self.layout.ensure(self.tenant)
            levers.declare_host(self.layout, self.tenant, self.job)
            self._readied = True

    @property
    def session(self) -> Optional[str]:
        cur = self._current.get()
        return cur[1] if cur else None

    def _meter(self) -> Meter:
        cur = self._current.get()
        if not cur:
            if self.meter.log is None:                                   # a call outside any run opens a "host" run
                levers.declare_host(self.layout, self.tenant, self.job, run_job="host")
            return self.meter
        meter, _, job, params = cur
        if meter.log is None:
            meter.run(job, **params)
            if job != self.job:                                          # a run of another job: one of the host's agents
                levers.declare_host(self.layout, self.tenant, self.job, run_job=job)
        return meter

    def run(self, job: Optional[str] = None, *, session: Optional[str] = None, **params) -> _Run:
        """One log run (a host run: a task, a conversation); `session` groups the run's calls for the advisor."""
        return _Run(self, job or self.job, session, params)

    def close(self) -> None:
        self.meter.close()


def wrap(client, *, store, tenant: str, job: str = "agent", key_label: Optional[str] = None, apply_levers: bool = True,
         watch: bool = True) -> Wrapped:
    """`watch=False` turns off the coverage watch (`recant.coverage`) for this process's wrappers that ask for it."""
    return Wrapped(client, store=store, tenant=tenant, job=job, key_label=key_label, apply_levers=apply_levers, watch=watch)

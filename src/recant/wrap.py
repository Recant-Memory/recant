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

import atexit
import datetime
import inspect
import sys
import time
import types
import weakref
from contextvars import ContextVar
from typing import Any, Optional

import pydantic

from . import costshape, coverage, levers, quality, switch
from .metering import Meter
from .store import StoreLayout


def _get(o, k, default=None):
    """A field of an SDK object or of the same thing as a dict (raw stream events arrive as either in tests)."""
    if o is None:
        return default
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


_TYPE_MRO = type.__dict__["__mro__"].__get__           # type's own descriptors: reading a class through them calls no
_TYPE_DICT = type.__dict__["__dict__"].__get__          # code of a host metaclass (no __getattribute__, no __eq__)
_TYPE_NAME = type.__dict__["__name__"].__get__


def _is_plain_scalar(t) -> bool:
    return t is str or t is int or t is float or t is bool          # identity, never ==


def _sdk_model(o) -> bool:
    """One of the Anthropic SDK's own model classes (a response block a tool loop puts back), told by its class alone,
    through type's own descriptors: nothing of the host's is called."""
    t = type(o)
    if not any(c is pydantic.BaseModel for c in _TYPE_MRO(t)):
        return False
    mod = _TYPE_DICT(t).get("__module__")
    if type(mod) is not str or not mod.startswith("anthropic."):
        return False
    m = sys.modules.get(mod)                             # the loaded SDK module of that name must hold this very class
    return m is not None and type(m) is types.ModuleType and \
        object.__getattribute__(m, "__dict__").get(_TYPE_NAME(t)) is t   # (a class merely naming the module is not one)


def _fields(o):
    """An SDK model's field values and extras, from the instance's own storage (no attribute access, no dump)."""
    yield from object.__getattribute__(o, "__dict__").values()
    extra = object.__getattribute__(o, "__pydantic_extra__")
    if type(extra) is dict:
        yield from extra.values()


def _readable(o, opaque: list) -> None:
    """Note in `opaque` every value Recant will not read. Types only, by identity and type's own descriptors: nothing
    of the host's is called (no iteration, comparison, attribute or serializer of a host object), so deciding cannot
    change what the SDK will send. An SDK model is looked inside: a host value in its fields makes it unread too."""
    t = type(o)
    if o is None or _is_plain_scalar(t):
        return
    if t is dict:
        for k, v in o.items():                           # an exact dict: the builtin's own iteration
            if type(k) is not str:
                opaque.append(_TYPE_NAME(type(k)))
            _readable(v, opaque)
        return
    if t is list:
        for x in o:
            _readable(x, opaque)
        return
    if _sdk_model(o):
        try:
            for v in _fields(o):
                _readable(v, opaque)
        except Exception:                                # noqa: BLE001 — storage Recant cannot see: not read
            opaque.append(_TYPE_NAME(t))
        return
    opaque.append(_TYPE_NAME(t))


def _plain(o):
    """A readable request (see `_readable`) as the plain JSON data the SDK sends: the SDK's own models by the SDK's own
    dump (`exclude_unset`, JSON mode, by alias, less `__api_exclude__`), so a tool loop's
    `{"role": "assistant", "content": r.content}` goes out in the bytes the raw SDK would send."""
    t = type(o)
    if t is dict:
        return {k: _plain(v) for k, v in o.items()}
    if t is list:
        return [_plain(x) for x in o]
    if _sdk_model(o):
        return _plain(o.model_dump(exclude_unset=True, mode="json", by_alias=True, exclude=getattr(o, "__api_exclude__", None)))
    return o


def _view(o):
    """The meter's copy of a request Recant does not read: plain values kept, every other value written as
    `<TypeName>`, by type alone (nothing of the host's is called); an SDK model is dumped only if it holds nothing else."""
    t = type(o)
    if o is None or _is_plain_scalar(t):
        return o
    if t is dict:
        return {(k if type(k) is str else f"<{_TYPE_NAME(type(k))}>"): _view(v) for k, v in o.items()}
    if t is list:
        return [_view(x) for x in o]
    if _sdk_model(o):
        inner: list = []
        _readable(o, inner)
        return f"<{_TYPE_NAME(t)}>" if inner else _plain(o)
    return f"<{_TYPE_NAME(t)}>"


def _usage(u) -> dict:
    """The SDK's usage, in Recant's names: thinking tokens as reasoning, the 1-hour part of the cache writes as long."""
    n = lambda v: int(v or 0)                                                       # noqa: E731
    return {"input": n(_get(u, "input_tokens")), "output": n(_get(u, "output_tokens")),
            "cache_read": n(_get(u, "cache_read_input_tokens")), "cache_write": n(_get(u, "cache_creation_input_tokens")),
            "reasoning": n(_get(_get(u, "output_tokens_details"), "thinking_tokens")),
            "cache_write_long": n(_get(_get(u, "cache_creation"), "ephemeral_1h_input_tokens"))}


def _field(d, key):
    """A field of a plain dict the host sent; None for anything else (Recant reads nothing it did not walk)."""
    return d.get(key) if type(d) is dict else None


def _reasoning(thinking) -> Optional[str]:
    """The request's thinking setting in Recant's terms: "off" when disabled, "on" when asked for, None when the host
    left it to the model's default (which, on the current models, thinks)."""
    kind = _field(thinking, "type")
    return None if kind is None else "off" if kind == "disabled" else "on"


def _markers(req: dict) -> list[str]:
    """Where the request carries cache markers, and at which lifetime: `system.0 (1h)`, `messages.5.0 (5m)`."""
    out = []
    msgs = req.get("messages")                           # only an exact list is read: anything else is the host's own
    groups = [("system", req.get("system"))] + [(f"messages.{i}", m.get("content") if type(m) is dict else None)
                                                  for i, m in enumerate(msgs if type(msgs) is list else [])]
    for at, blocks in groups:
        for j, b in enumerate(blocks if type(blocks) is list else []):
            if type(b) is dict and "cache_control" in b:
                out.append(f"{at}.{j} ({_field(b['cache_control'], 'ttl') or '5m'})")
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
        self.w, self.ttft, self.inside, self.t0, self.opaque = w, None, 0.0, 0.0, False
        try:
            self.t0 = time.monotonic()
            self._prepare(kwargs, apply)
        except Exception as exc:                                         # noqa: BLE001 — Recant's own failure never
            self.broken(exc)                                             # reaches the host: its request goes as built
            self.sent = _host_request(kwargs)                            # (the walk reads nothing it could consume)

    def broken(self, exc: BaseException) -> None:
        """Recant failed on this call: nothing more is metered for it, the failure is counted in the friction log."""
        from . import friction
        self.done, self.opted, self.equivalent = True, [], None
        friction.wrapper_failed(exc)                                     # never raises

    def _prepare(self, kwargs: dict, apply: bool) -> None:
        t = time.perf_counter()
        w = self.w
        w._ready()
        host = dict(kwargs)
        self.session = host.pop("session", None) or w.session
        self.meter = w._meter()
        self.job = w._job()                                              # the run's job: `run("researcher")`, else wrap's
        opaque: list = []
        _readable(host, opaque)                                          # first decide, by type alone; then read
        if opaque:                                                       # a value Recant does not read: the host's own
            apply = False                                                # objects go, untouched, with no lever
            built = _view(host)
            self.meter.log.count("request_opaque")
        else:
            built = _plain(host)                                         # the SDK's own blocks as the SDK sends them
        ov = costshape.overlay(w.layout, w.tenant) if (apply and w.apply_levers) else {}
        sent = levers.apply(built, ov, job=self.job) if ov else dict(built)
        self.equivalent = True if opaque else quality.equivalent(built, sent)
        if not self.equivalent:                                          # never send what the host did not build, bar the markers
            sent = dict(built)
        sent, self.opted = levers.apply_opted_in(sent, ov, self.job) if ov else (sent, [])   # gated levers: after the check
        if opaque:
            sent = host
        w.last_equivalent, w.last_sent_markers = self.equivalent, (_markers(sent) if not opaque else [])
        self.built, self.sent, self.done, self.opaque = built, sent, False, bool(opaque)
        self.ttft: Optional[float] = None                                # streams: seconds from the send to the first event
        self.inside = time.perf_counter() - t                            # Recant's own time: before the send, per event, after
        self.t0 = time.monotonic()

    def record(self, *, usage: Optional[dict] = None, model=None, stop_reason=None, error=None) -> None:
        if self.done:
            return
        try:
            self._record(usage=usage, model=model, stop_reason=stop_reason, error=error)
        except Exception as exc:                                         # noqa: BLE001 — the host has its response
            self.broken(exc)

    def _record(self, *, usage, model, stop_reason, error) -> None:
        arrived, t = time.monotonic(), time.perf_counter()
        self.done = True
        b = self.built
        req = self.built if self.opaque else self.sent                   # an unread request: only its type-only view
        kw = dict(job=self.job, model=str(model or b.get("model")), usage=usage or {}, session=self.session,
                  system=b.get("system"), tools=b.get("tools"), messages=b.get("messages"), max_tokens=b.get("max_tokens"),
                  effort=_field(req.get("output_config"), "effort"), reasoning=_reasoning(req.get("thinking")),
                  latency_ms=int((arrived - self.t0) * 1000), stop_reason=stop_reason, error=error,
                  ttft_ms=None if self.ttft is None else int(self.ttft * 1000), bench_task=self.w.task_tag)
        kw["overhead_ms"] = round((self.inside + time.perf_counter() - t) * 1000, 3)   # the record's own write excluded
        self.meter.record(**kw)
        self.meter.log.count("levers_equivalent" if self.equivalent else "levers_not_equivalent")
        for lid in self.opted:
            self.meter.log.count(f"opted_in:{lid}")

    def message(self, msg) -> None:
        if self.done:
            return
        try:
            usage, model, stop = _usage(_get(msg, "usage")), _get(msg, "model"), _get(msg, "stop_reason")
        except Exception as exc:                                         # noqa: BLE001 — the host has its response
            self.broken(exc)
            return
        self.record(usage=usage, model=model, stop_reason=stop)

    def failed(self, exc: BaseException, *, usage: Optional[dict] = None, model=None) -> None:
        """A call that failed: the error, and whatever usage a stream had reported before it broke."""
        self.record(usage=usage, model=model, error=type(exc).__name__)

    def snapshot(self, stream, error: Optional[BaseException] = None) -> None:
        """The helper stream's message as accumulated so far: the final one when the host read it through."""
        if self.done:
            return
        try:
            msg = stream.current_message_snapshot
        except Exception:                                                # noqa: BLE001 — no event arrived before the close
            msg = None
        try:
            usage, model = (_usage(_get(msg, "usage")), _get(msg, "model")) if msg is not None else (None, None)
            stop = _get(msg, "stop_reason")
        except Exception as exc:                                         # noqa: BLE001
            self.broken(exc)
            return
        if error is not None:
            self.failed(error, usage=usage, model=model)
        else:
            self.record(usage=usage, model=model, stop_reason=stop)


# ---------------------------------------------------------------- create(stream=True): the raw event stream
class _RawStream:
    def __init__(self, inner, call: _Call):
        self._inner, self._call, self._ev = inner, call, _Events()

    def _see(self, ev):
        if self._call.done:
            return
        try:
            t = time.perf_counter()
            if self._call.ttft is None:
                self._call.ttft = time.monotonic() - self._call.t0
            self._ev.see(ev)
            self._call.inside += time.perf_counter() - t
        except Exception as exc:                                         # noqa: BLE001 — the host still gets every event
            self._call.broken(exc)

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
        try:
            if self._call.ttft is None and not self._call.done:
                self._call.ttft = time.monotonic() - self._call.t0
        except Exception as exc:                                         # noqa: BLE001 — the host still gets every event
            self._call.broken(exc)

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
    try:
        if not call.done and hasattr(stream, "_raw_stream"):
            stream._raw_stream = _FirstEvent(stream._raw_stream, call)
    except Exception as exc:                                             # noqa: BLE001
        call.broken(exc)


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
        try:
            meter = Meter(self.w.layout, self.w.tenant, key_label=self.w.key_label)   # its log opens on the first metered
        except Exception as exc:                                          # noqa: BLE001 — call: a run with the switch off
            from . import friction                                        # writes nothing. Should Recant fail here, the
            friction.wrapper_failed(exc)                                  # run's calls go out as built, unmetered
            meter = None
        self.token = self.w._current.set((meter, self.session, self.job, self.params))
        return self.w

    def __exit__(self, *exc):
        meter = self.w._current.get()[0]
        self.w._current.reset(self.token)
        if meter is not None:
            try:
                meter.close()
            except Exception as e:                                        # noqa: BLE001 — never the host's error
                from . import friction
                friction.wrapper_failed(e)

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *exc):
        self.__exit__(*exc)


_OPEN: "weakref.WeakSet" = weakref.WeakSet()          # wrappers not closed yet; held weakly, so one let go is freed


def _close_at_exit() -> None:
    for w in list(_OPEN):
        w.close()


atexit.register(_close_at_exit)


class Wrapped:
    def __init__(self, client, *, store, tenant: str, job: str = "agent", key_label: Optional[str] = None, apply_levers: bool = True,
                 watch: bool = True):
        self.client, self.tenant, self.job, self.key_label, self.apply_levers = client, tenant, job, key_label, apply_levers
        self.layout = store if isinstance(store, StoreLayout) else StoreLayout(store)
        self._readied = False                                            # the store is made here, unless the switch is off
        self.meter = Meter(self.layout, tenant, key_label=key_label, threads=True)   # calls made outside any run,
        _OPEN.add(self)                                                  # threaded by content; their run closes at exit
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
        """The switch (`recant.switch`), read on every call: on, off or meter. Should reading it fail inside Recant, the
        call is a pure pass-through ("off") and the failure is counted."""
        try:
            return switch.state(self.layout)
        except Exception as exc:                                         # noqa: BLE001
            from . import friction
            friction.wrapper_failed(exc)
            return "off"

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
            if job != self.job:                                          # a run of another job: one of the host's agents,
                levers.declare_host(self.layout, self.tenant, job, run_job=job)   # its calls sized and reported as that job
        return meter

    def _job(self) -> str:
        cur = self._current.get()
        return cur[2] if cur else self.job

    def run(self, job: Optional[str] = None, *, session: Optional[str] = None, **params) -> _Run:
        """One log run (a host run: a task, a conversation), named by its job: the calls inside it are that job's, so a
        two-agent host opens `run("researcher")` and `run("writer")` and `recant cost` reports and sizes each apart
        (`effort.researcher`, `reasoning.writer`). With no job, the wrapper's (`wrap(..., job=...)`, default "agent").
        `session` groups the run's calls for the advisor."""
        return _Run(self, job or self.job, session, params)

    def close(self) -> None:
        _OPEN.discard(self)
        try:
            self.meter.close()
        except Exception as exc:                                         # noqa: BLE001 — never the host's error
            from . import friction
            friction.wrapper_failed(exc)


def wrap(client, *, store=None, tenant: Optional[str] = None, job: str = "agent", key_label: Optional[str] = None,
         apply_levers: bool = True, watch: bool = True) -> Wrapped:
    """`recant.wrap(client)` is all it takes: with no `store` the store is `$RECANT_HOME` or `./.recant/`, with no
    `tenant` it is "default" (`recant.home`), and every `recant` command finds the same. `watch=False` turns off the
    coverage watch (`recant.coverage`) for this process's wrappers that ask for it."""
    from . import home
    return Wrapped(client, store=home.store_or_default(store), tenant=home.tenant_or_default(tenant), job=job,
                   key_label=key_label, apply_levers=apply_levers, watch=watch)

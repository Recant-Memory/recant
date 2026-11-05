"""one switch. Off, the wrapper is a pure pass-through — the request the raw client would send, byte for byte,
the SDK's own response, nothing in the store or the log — on every path, sync and async; `--meter` meters and changes
nothing on the request; the switch takes effect on the next call; RECANT= in the environment wins; the wrapper's own
overhead is recorded and measured."""
import asyncio
import json
import time

import pytest

from recant import switch, wrap
from recant.cli import main
from recant_fixtures import TENANT
from test_wrap_paths import API, REQ, apply_breakpoint, call_async, call_sync, records, sdk

PATHS = ["create", "create_stream", "stream_helper"]


def run_once(client, path, aio, *, session=None):
    if aio:
        async def go():
            async with client.run("agent", session=session):
                return await call_async(client, path)
        return asyncio.run(go())
    with client.run("agent", session=session):
        return call_sync(client, path)


def raw_call(path, aio):
    """What the raw client sends and returns for the same call: the reference."""
    api = API()
    c = sdk(api, aio=aio)
    out = asyncio.run(call_async(c, path)) if aio else call_sync(c, path)
    return api.bodies, out


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("path", PATHS)
def test_off_is_a_byte_identical_pass_through_that_writes_nothing(layout, tmp_path, aio, path, monkeypatch):
    monkeypatch.delenv(switch.ENV, raising=False)
    apply_breakpoint(layout)                                                   # a lever applied: off must not carry it out
    switch.set_state(layout, "off")
    before = sorted(p.relative_to(layout.root) for p in layout.root.rglob("*"))
    api = API()
    client = wrap(sdk(api, aio=aio), store=layout, tenant=TENANT)
    out = run_once(client, path, aio, session="s1")
    want_bodies, want_out = raw_call(path, aio)
    assert api.bodies == want_bodies and json.dumps(api.bodies[0]) == json.dumps(want_bodies[0])   # byte for byte, key order too
    assert out == want_out and "cache_control" not in json.dumps(api.bodies)
    assert sorted(p.relative_to(layout.root) for p in layout.root.rglob("*")) == before            # nothing written
    assert records(layout) == []


def test_off_with_a_fresh_store_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(switch.ENV, "off")
    store = tmp_path / "never"
    client = wrap(sdk(API()), store=store, tenant=TENANT)
    with client.run("agent", session="s"):
        call_sync(client, "create")
    assert not store.exists()


@pytest.mark.parametrize("aio", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("path", PATHS)
def test_meter_keeps_the_request_byte_identical_and_records_the_call(layout, aio, path, monkeypatch):
    monkeypatch.delenv(switch.ENV, raising=False)
    apply_breakpoint(layout)
    switch.set_state(layout, "meter")
    api = API()
    client = wrap(sdk(api, aio=aio), store=layout, tenant=TENANT)
    run_once(client, path, aio, session="s1")
    want_bodies, _ = raw_call(path, aio)
    assert json.dumps(api.bodies) == json.dumps(want_bodies)
    rec, = records(layout)
    assert rec["session"] == "s1" and rec["usage"]["output"] == 250 and rec["overhead_ms"] >= 0
    assert (rec["ttft_ms"] is not None) == (path != "create")                  # streamed calls carry a time to first token


def test_the_switch_takes_effect_on_the_next_call_and_the_environment_wins(layout, monkeypatch, capsys):
    monkeypatch.delenv(switch.ENV, raising=False)
    apply_breakpoint(layout)
    api = API()
    client = wrap(sdk(api), store=layout, tenant=TENANT)
    marked = lambda b: "cache_control" in json.dumps(b)                       # noqa: E731
    with client.run("agent", session="s"):
        call_sync(client, "create")                                           # on: the lever carried out
        assert main(["off", "--store", str(layout.root)]) == 0
        call_sync(client, "create")                                           # same client, same run: off from this call
        assert main(["on", "--store", str(layout.root)]) == 0
        call_sync(client, "create")
    assert [marked(b) for b in api.bodies] == [True, False, True]
    assert len(records(layout)) == 2                                          # the off call left no record
    monkeypatch.setenv(switch.ENV, "off")
    with client.run("agent"):
        call_sync(client, "create")
    assert marked(api.bodies[-1]) is False and len(records(layout)) == 2
    capsys.readouterr()
    main(["status", "--store", str(layout.root)])
    st = json.loads(capsys.readouterr().out)
    assert st["state"] == "off" and st["from"] == "environment RECANT=off" and st["file"]["state"] == "on"


def test_the_wrappers_own_overhead_is_small_off_and_on(layout, monkeypatch):
    """Measured, not assumed: per call, the time the wrapper adds over the raw client on a mock transport, off and on."""
    monkeypatch.delenv(switch.ENV, raising=False)
    apply_breakpoint(layout)
    n = 30

    def per_call(client):
        t = time.perf_counter()
        for _ in range(n):
            call_sync(client, "create")
        return (time.perf_counter() - t) / n

    raw = per_call(sdk(API()))
    switch.set_state(layout, "off")
    off = per_call(wrap(sdk(API()), store=layout, tenant=TENANT))
    switch.set_state(layout, "on")
    on_client = wrap(sdk(API()), store=layout, tenant=TENANT)
    with on_client.run("agent", session="s"):
        on = per_call(on_client)
    recs = records(layout)
    assert len(recs) == n and all(r["overhead_ms"] is not None for r in recs)
    assert off - raw < 0.002                                                  # off: a file read and a dict copy, under 2 ms
    assert sorted(r["overhead_ms"] for r in recs)[n // 2] < 20                # on: the median recorded overhead
    print(f"per call: raw {raw * 1000:.2f} ms, off {off * 1000:.2f} ms, on {on * 1000:.2f} ms; "
          f"recorded overhead median {sorted(r['overhead_ms'] for r in recs)[n // 2]:.2f} ms")


def test_recant_cost_reports_latency_ttft_and_the_wrappers_own_time(layout, monkeypatch):
    """`recant cost` carries end-to-end time per call and per task, time to first token on streamed calls and
    Recant's own time, p50 and p95."""
    monkeypatch.delenv(switch.ENV, raising=False)
    client = wrap(sdk(API()), store=layout, tenant=TENANT)
    for i in range(3):
        with client.run("agent", session=f"t{i}"):
            call_sync(client, "stream_helper")
            call_sync(client, "create")
    from recant import costshape as cs
    from recant.cli import run_cost
    rep = cs.report(layout, TENANT, since="1d")
    lat = cs.latency(rep.calls)
    assert lat["call"]["n"] == 6 and lat["task"]["n"] == 3 and lat["ttft"]["n"] == 3 and lat["overhead"]["n"] == 6
    assert lat["task"]["p50"] >= lat["call"]["p50"] and lat["call"]["p95"] >= lat["call"]["p50"]
    text = run_cost(layout, TENANT, cs.HostShape(), since="1d")
    assert "== latency ==" in text and "time to first token (streamed calls): p50" in text and "Recant's own time per call" in text
    assert cs._pct([1, 2, 3, 4], 0.5) == 2 and cs._pct([1, 2, 3, 4], 0.95) == 4 and cs._pct([], 0.5) is None

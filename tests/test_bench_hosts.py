"""The bench on hosts of every shape: one test per fairness, freeze and accounting guarantee. A richer simulated Messages API than test_bench's: the cached prefix is
tools, then system, then messages, as the provider builds it; every cached prefix remembers the arm that wrote it and
every read is recorded, so a read across arms is visible; `stream=True` returns raw events and can drop mid-body after
reporting usage."""
import asyncio
import hashlib
import json
import re
import textwrap
from types import SimpleNamespace

import pytest

from recant import bench, costshape as cs, friction
from recant.store import StoreLayout


def toks(x) -> int:
    return len(json.dumps(x, sort_keys=True)) // 4


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def marked(x) -> bool:
    return "cache_control" in json.dumps(x)


class RemoteProtocolError(Exception):
    """The proxy's mid-body drop, as httpx raises it."""


class Sim:
    """Bills as the provider would, roughly. The prefix is tools + system + messages; a breakpoint on the last tool, the
    system or the last message writes the prefix up to it; a request reads the longest written prefix it starts with."""

    def __init__(self, drop=None):
        self.messages, self.cache, self.bodies, self.reads, self.drop = self, {}, [], [], drop or (lambda kw: False)

    @staticmethod
    def arm_of(kw) -> str:
        m = re.search(r"arm ([A-Z])\]", json.dumps(kw))
        return m.group(1) if m else "?"

    def _usage(self, kw):
        self.bodies.append(json.loads(json.dumps(kw)))
        arm = self.arm_of(kw)
        tools, sys_, msgs = strip(kw.get("tools") or []), strip(kw.get("system")), strip(kw.get("messages") or [])
        pref = [("t", [tools], toks(tools))] if tools else []
        pref.append(("s", [tools, sys_], toks(tools) + toks(sys_)))
        pref += [(f"m{k}", [tools, sys_, msgs[:k]], toks(tools) + toks(sys_) + toks(msgs[:k])) for k in range(1, len(msgs) + 1)]
        keyed = [(json.dumps(p, sort_keys=True), t) for _, p, t in pref]
        total = keyed[-1][1]
        hits = [(k, t) for k, t in keyed if k in self.cache]
        hit = max([t for _, t in hits] or [0])
        if hits:
            k = max(hits, key=lambda kt: kt[1])[0]
            self.reads.append((arm, self.cache[k][0]))
        writes = []
        if tools and marked((kw.get("tools") or [])[-1]):
            writes.append(keyed[0])
        if marked(kw.get("system")):
            writes.append(keyed[1 if tools else 0])
        if msgs and marked((kw.get("messages") or [])[-1]):
            writes.append(keyed[-1])
        written = max([t for k, t in writes if k not in self.cache] or [0])
        for k, t in writes:
            self.cache.setdefault(k, (arm, t))
        cw = max(0, written - hit)
        return {"input_tokens": total - hit - cw, "output_tokens": 60, "cache_read_input_tokens": hit, "cache_creation_input_tokens": cw}

    def create(self, **kw):
        u = self._usage(kw)
        if kw.get("stream"):
            return self._events(kw, u)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Send a photo.")], usage=SimpleNamespace(**u),
                               stop_reason="end_turn", model=kw["model"])

    def _events(self, kw, u):
        yield {"type": "message_start", "message": {"model": kw["model"], "usage": {**u, "output_tokens": 1}}}
        if self.drop(kw):
            raise RemoteProtocolError("peer closed connection without sending complete message body")
        yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 60}}
        yield {"type": "message_stop"}

    def count_tokens(self, **kw):
        return SimpleNamespace(input_tokens=toks(kw))


class AsyncSim(Sim):
    """The same simulation behind the async client's interface."""

    def __init__(self, drop=None):
        super().__init__(drop)
        self.messages = SimpleNamespace(create=self.acreate, count_tokens=self.count_tokens)

    async def acreate(self, **kw):
        await asyncio.sleep(0)
        return Sim.create(self, **kw)


AGENTS = textwrap.dedent('''
    import threading
    from concurrent.futures import ThreadPoolExecutor
    HANDBOOK = "You are the support agent. " + "Policy line. " * 300
    TOOLS = [{"name": "lookup", "description": "Look up an order. " * 40, "input_schema": {"type": "object", "properties": {}}},
             {"name": "refund", "description": "Refund an order. " * 40, "input_schema": {"type": "object", "properties": {}},
              "cache_control": {"type": "ephemeral"}}]                     # the host caches its own tool definitions
    QS = ("What now?", "Can I return it?", "How long?", "Express?")

    def arm(client):
        return client.messages._w.client.salt[-2]

    def turns(client, task, n=4, **extra):
        msgs = []
        for q in QS[:n]:
            msgs.append({"role": "user", "content": f"order {task['id']}: {q}"})
            r = client.messages.create(model=task.get("model", "claude-sonnet-5-5"), max_tokens=256, system=HANDBOOK, messages=msgs, **extra)
            msgs.append({"role": "assistant", "content": r.content[0].text})
        return len(msgs)

    def run_task(task, client):
        return turns(client, task)

    def run_task_tools(task, client):
        return turns(client, task, tools=TOOLS)

    def run_task_session(task, client):                                    # its own conversation id, Recant's session=
        return turns(client, task, session=f"conv-{task['id']}")

    def run_task_agents(task, client):                                     # a multi-agent host's shape: one client, several agents
        with client.run("planner", session=f"plan-{task['id']}"):
            turns(client, task, n=2)
        with client.run("executor", session=f"exec-{task['id']}"):
            turns(client, task, n=2)
        with ThreadPoolExecutor(2) as pool:                                # and two more agents on threads
            list(pool.map(lambda k: turns(client, {**task, "id": f"{task['id']}-{k}"}, n=1), range(2)))
        return 1

    async def run_task_async_agents(task, client):
        import asyncio
        async def agent(name, n):
            async with client.run(name, session=f"{name}-{task['id']}"):
                msgs = []
                for q in QS[:n]:
                    msgs.append({"role": "user", "content": f"order {task['id']}: {q}"})
                    r = await client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK, messages=msgs)
                    msgs.append({"role": "assistant", "content": r.content[0].text})
        await asyncio.gather(agent("planner", 2), agent("executor", 2))
        return 1

    def run_task_stream(task, client):
        msgs = []
        for q in QS:
            msgs.append({"role": "user", "content": f"order {task['id']}: {q}"})
            for ev in client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK, messages=msgs, stream=True):
                pass
            msgs.append({"role": "assistant", "content": "Send a photo."})
        return len(msgs)

    def run_task_fail_early_in_b(task, client):                            # fails after one call, in arm B only
        if task.get("explode") and arm(client) == "B":
            turns(client, task, n=1)
            raise RuntimeError("the agent fell over")
        return turns(client, task)

    def run_task_interrupted_once(task, client):                           # the process dies mid-task, once
        import os
        mark = f"interrupted-{task['id']}-{arm(client)}"
        if task.get("interrupt") and arm(client) == "B" and not os.path.exists(mark):
            turns(client, task, n=2)
            open(mark, "w").close()
            raise KeyboardInterrupt
        return turns(client, task)

    def grade(task, result):
        return True
''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "shape_agents.py").write_text(AGENTS)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    def tasks(n, **marks):
        p = tmp_path / "tasks.jsonl"
        rows = []
        for i in range(n):
            row = {"id": i}
            for k, ids in marks.items():
                if k == "model":
                    row["model"] = ids
                elif i in ids:
                    row[k] = True
            rows.append(row)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return str(p)
    return tmp_path, tasks


def run(t, out, api, fn="run_task", grade=None, **kw):
    lines = []
    rep = bench.run(tasks_path=t, run_spec=f"shape_agents:{fn}", grade_spec=grade, out=str(out), yes=True,
                    sdk_factory=lambda aio: api, say=lines.append, **kw)
    return rep, lines


def calls_seen(api, salt) -> int:
    return sum(1 for b in api.bodies if salt in json.dumps(b))


# ---------------------------------------------------------------- attribution
@pytest.mark.parametrize("fn,per_task", [("run_task_session", 4), ("run_task_agents", 6)])
def test_every_call_of_a_task_is_counted_however_run_task_uses_the_client(env, fn, per_task):
    tmp, tasks = env
    api = Sim()
    rep, _ = run(tasks(8), tmp / fn, api, fn=fn, grade="shape_agents:grade", calibrate=2)
    for a in ("A", "B"):
        assert rep["tokens"][a]["calls_per_task"] == per_task, (a, rep["tokens"][a])
        assert calls_seen(api, rep["header"]["salts"][a]) == 6 * per_task                  # every call the arm made
    assert rep["cost"]["per_task"]["A"] > 0 and rep["cost"]["per_task"]["B"] < rep["cost"]["per_task"]["A"]


def test_async_agents_in_flight_together_are_counted(env):
    tmp, tasks = env
    rep, _ = run(tasks(6), tmp / "aio", AsyncSim(), fn="run_task_async_agents", calibrate=2)
    assert rep["tokens"]["A"]["calls_per_task"] == 4 and rep["tokens"]["B"]["calls_per_task"] == 4


def test_an_unpriced_model_is_reported_not_a_crash(env):
    tmp, tasks = env
    rep, _ = run(tasks(6, model="claude-unpriced-9"), tmp / "unpriced", Sim(), grade="shape_agents:grade", calibrate=0, arm_b="off")
    assert any("not priced" in x for x in rep["verdict"])


# ---------------------------------------------------------------- the freeze
def test_an_edited_overlay_cannot_move_a_verdict(env):
    tmp, tasks = env
    run(tasks(8), tmp / "ov", Sim(), calibrate=2)
    ovp = next((tmp / "ov" / "arm-B").glob("costshape/**/overlay.json"))
    ov = json.loads(ovp.read_text())
    ov["breakpoint.add"]["predicted_share"] = 0.99
    ovp.write_text(json.dumps(ov))
    with pytest.raises(SystemExit, match="differ from the frozen predictions"):
        bench.build_report(tmp / "ov")


def test_editing_predictions_and_manifest_together_fails(env):
    tmp, tasks = env
    run(tasks(8), tmp / "both", Sim(), calibrate=2)
    pp, mp = tmp / "both" / "predictions.json", tmp / "both" / "manifest.json"
    pr = json.loads(pp.read_text())
    pr["arms"]["B"]["levers"][0]["share"] = 0.9
    raw = (json.dumps(pr, indent=1, sort_keys=True) + "\n").encode()
    pp.write_bytes(raw)
    m = json.loads(mp.read_text())
    m["predictions_hash"] = hashlib.sha256(raw).hexdigest()
    mp.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="outside the run directory"):
        bench.build_report(tmp / "both")


def test_predictions_written_after_the_arms_ran_fail(env):
    tmp, tasks = env
    t = tasks(8)
    run(t, tmp / "late", Sim(), calibrate=2)
    mp = tmp / "late" / "manifest.json"
    m = json.loads(mp.read_text())
    m["predictions_hash"] = None
    mp.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="after the first paired task"):
        run(t, tmp / "late", Sim(), calibrate=2)


# ---------------------------------------------------------------- failed and dropped calls
def test_a_dropped_streams_billed_usage_is_counted(env):
    tmp, tasks = env
    drop = lambda kw: Sim.arm_of(kw) == "B" and "order 4: How long?" in json.dumps(kw["messages"][-1])   # noqa: E731
    api = Sim(drop=drop)
    rep, _ = run(tasks(8), tmp / "drop", api, fn="run_task_stream", calibrate=2)
    t4 = 4
    b = StoreLayout(tmp / "drop" / "arm-B")
    recs = [r for p in sorted(b.log_dir("bench").glob("*.jsonl")) for r in map(json.loads, p.read_text().splitlines())
            if r["kind"] == "model_call" and r.get("error")]
    assert len(recs) == 1 and recs[0]["usage"]["input"] + recs[0]["usage"]["cache_read"] > 0
    assert rep["failures"]["B"]["failed_calls"] == 1
    priced = cs.LIST_PRICES["claude-sonnet-5-5"].bill(uncached=recs[0]["usage"]["input"], write=recs[0]["usage"]["cache_write"],
                                                       read=recs[0]["usage"]["cache_read"], output=recs[0]["usage"]["output"])
    assert rep["cost"]["failed_calls_usd"]["B"] == pytest.approx(priced)
    assert cs.report(b, "bench", since=0).health()["failed_calls"] == 1                     # recant cost sees it too


def test_a_failed_task_never_counts_as_a_saving(env):
    tmp, tasks = env
    rep, _ = run(tasks(12, explode=set(range(2, 12, 2))), tmp / "early", Sim(), fn="run_task_fail_early_in_b",
                 calibrate=2, arm_b="off")                                                   # A/A: nothing to save but failures
    d = rep["cost"]["paired_difference_per_task"]
    assert d["delta"] >= 0, d
    assert any("failure" in x and "never a saving" in x for x in rep["verdict"])


# ---------------------------------------------------------------- the cache across arms
def test_on_on_neither_arm_reads_a_cache_the_other_wrote(env):
    tmp, tasks = env
    api = Sim()
    rep, _ = run(tasks(10), tmp / "onon", api, fn="run_task_tools", calibrate=2, arm_a="safe", arm_b="safe")
    assert rep["header"]["arms"] == {"A": "safe", "B": "safe"}
    crossed = [(reader, writer) for reader, writer in api.reads if reader != writer]
    assert api.reads and not crossed, crossed[:5]
    c = rep["cost"]["paired_difference_per_task"]
    assert c["interval"][0] <= 0 <= c["interval"][1]


# ---------------------------------------------------------------- a copied store
def test_store_copies_memory_not_logs_levers_or_switch(env):
    tmp, tasks = env
    src = tmp / "dev-store"
    lay = StoreLayout(src)
    lay.ensure("bench")
    (lay.tenant_dir("bench") / "marker.md").write_text("a memory")
    from recant.log import LogWriter
    with LogWriter(lay, "bench", "agent", artifact_hash="host", kernel_hash="host", params={"old": True}) as log:
        old_log = log.path.name                                                           # the developer's own history
    cs.opt_in(lay, "bench", "reasoning.agent", action={"reasoning": "off"}, evidence="the developer's own")
    (src / "recant.switch.json").write_text('{"state": "off"}')
    rep, _ = run(tasks(6), tmp / "copy", Sim(), calibrate=2, store=str(src))
    b = StoreLayout(tmp / "copy" / "arm-B")
    assert (b.tenant_dir("bench") / "marker.md").exists()
    assert not (b.log_dir("bench") / old_log).exists()
    assert "reasoning.agent" not in cs.overlay(b, "bench")
    assert "tenants" in rep["header"]["store_copied"] and set(rep["header"]["store_copied"]) <= {"tenants", "versions", "machinery"}


# ---------------------------------------------------------------- an interrupted task
def test_an_interrupted_attempt_is_discarded_on_resume(env):
    tmp, tasks = env
    t = tasks(8, interrupt={4})
    api = Sim()
    with pytest.raises(KeyboardInterrupt):
        run(t, tmp / "int", api, fn="run_task_interrupted_once", calibrate=2)
    rep, _ = run(t, tmp / "int", api, fn="run_task_interrupted_once", calibrate=2)
    assert rep["tokens"]["B"]["calls_per_task"] == 4 == rep["tokens"]["A"]["calls_per_task"]
    assert rep["discarded"]["B"]["calls"] == 2 and rep["discarded"]["B"]["usd"] > 0


# ---------------------------------------------------------------- smaller cases
def test_each_arm_has_its_own_raw_client(env):
    tmp, tasks = env
    made = []

    def factory(aio):
        made.append(Sim())
        return made[-1]
    bench.run(tasks_path=tasks(6), run_spec="shape_agents:run_task", out=str(tmp / "clients"), yes=True, calibrate=2,
              sdk_factory=factory, say=lambda *_: None)
    assert len(made) >= 3 and all(len(m.bodies) for m in made)


def test_order_is_balanced_and_seeded_not_task_parity(env):
    tmp, tasks = env
    rep, _ = run(tasks(42), tmp / "order", Sim(), calibrate=2, arm_b="off")
    man = json.loads((tmp / "order" / "manifest.json").read_text())
    rows = [json.loads(x) for x in (tmp / "order" / "results.jsonl").read_text().splitlines()]
    first = {}
    for r in rows:
        if r["arm"] in ("A", "B"):
            first.setdefault(r["task"], r["arm"])
    assert sum(v == "A" for v in first.values()) == 20 and "order_seed" in man
    assert [first[i] for i in sorted(first)] != ["A" if k % 2 == 0 else "B" for k in range(40)]


def test_parallel_tasks_keep_every_call_in_its_own_task_and_arm(env):
    """--parallel: tasks in flight at once (each worker its own clients), every call still counted in its own task and
    arm, no arm reading another's cache, every pair complete."""
    tmp, tasks = env
    api = Sim()
    rep, _ = run(tasks(14), tmp / "par", api, fn="run_task_agents", calibrate=4, parallel=4)
    assert rep["header"]["tasks_paired"] == 10
    assert rep["tokens"]["A"]["calls_per_task"] == 6 == rep["tokens"]["B"]["calls_per_task"]
    assert not [(r, w) for r, w in api.reads if r != w]
    assert rep["discarded"]["A"]["calls"] == 0 == rep["discarded"]["B"]["calls"]


def test_a_host_that_retries_a_dropped_call_and_finishes_is_not_failed(env):
    """The grader decides a task, not whether any call failed. A dropped stream the host retries counts its
    billed usage and is reported, and the task that finishes is graded as usual (an earlier rule had it failed)."""
    tmp, tasks = env
    (tmp / "shape_retry.py").write_text(textwrap.dedent('''
        HANDBOOK = "You are the support agent. " + "Policy line. " * 300
        def run_task(task, client):
            msgs = []
            for q in ("What now?", "Can I return it?", "How long?"):
                msgs.append({"role": "user", "content": f"order {task['id']}: {q}"})
                for attempt in range(3):                                   # the host's own retry
                    try:
                        for ev in client.messages.create(model="claude-sonnet-5-5", max_tokens=256, system=HANDBOOK,
                                                         messages=msgs, stream=True):
                            pass
                        break
                    except Exception:
                        continue
                msgs.append({"role": "assistant", "content": "Send a photo."})
            return True
        def grade(task, result):
            return bool(result)
    '''))
    dropped = set()

    def drop(kw):                                                          # each task's second question drops once in B
        key = json.dumps(kw["messages"][-1])
        if Sim.arm_of(kw) == "B" and "Can I return it?" in key and key not in dropped:
            dropped.add(key)
            return True
        return False
    tasks(8)
    rep = bench.run(tasks_path=str(tmp / "tasks.jsonl"), run_spec="shape_retry:run_task", grade_spec="shape_retry:grade",
                    out=str(tmp / "retry2"), yes=True, calibrate=2, sdk_factory=lambda aio: Sim(drop=drop), say=lambda *_: None)
    assert rep["failures"]["B"]["failed_calls"] == 6 and rep["failures"]["B"]["failed_tasks"] == 0
    assert rep["quality"]["pass_rate"] == {"A": 1.0, "B": 1.0} and rep["cost"]["failed_calls_usd"]["B"] > 0

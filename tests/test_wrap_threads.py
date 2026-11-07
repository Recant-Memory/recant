"""recant.wrap with no client.run: the calls of many conversations in flight at once land in one host run. Each call
is given the conversation it continues, found by content (the longest earlier request whose messages are a prefix of
its own), so `breakpoint.add` is sized per conversation; and the host run closes when the process exits, so the
coverage watch writes. (An outside developer's wrap-only app, 40 tasks four at a time, read "no safe saving": 3 of
64 calls were seen to extend their predecessor, and the report said the watch was not on.)"""
import json
import subprocess
import sys
from pathlib import Path

from recant import costshape as cs, wrap
from recant.log import read_run
from test_stranger import FakeSDK, TENANT

HERE = Path(__file__).parent


def _interleaved(client, n=8, turns=3, model="claude-sonnet-5-5"):
    convs = [[] for _ in range(n)]
    for k in range(turns):                                     # every conversation's turn k before any turn k+1
        for i, msgs in enumerate(convs):
            msgs.append({"role": "user", "content": f"conversation {i} turn {k} " * 40})
            client.messages.create(model=model, max_tokens=50, system="You are terse. " * 200, messages=list(msgs))
            msgs.append({"role": "assistant", "content": "ok"})


def _records(layout):
    return [r for p in sorted(layout.log_dir(TENANT).glob("*.jsonl")) for r in read_run(p, typed=False)]


def test_interleaved_conversations_are_threaded_by_content(layout):
    client = wrap(FakeSDK(), store=layout, tenant=TENANT)
    _interleaved(client)
    calls = [r for r in _records(layout) if r["kind"] == "model_call"]
    assert len(calls) == 24 and len({c["session"] for c in calls}) == 8
    assert sum(1 for c in calls if c["extends_previous"]) == 16           # every call after a conversation's first
    by = {}
    for c in calls:
        by.setdefault(c["session"], []).append(c["extends_previous"])
    assert all(v == [False, True, True] for v in by.values())


def test_the_same_opening_twice_is_two_conversations(layout):
    client = wrap(FakeSDK(), store=layout, tenant=TENANT)
    for _ in range(2):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=50, messages=[{"role": "user", "content": "hi"}])
    calls = [r for r in _records(layout) if r["kind"] == "model_call"]
    assert len({c["session"] for c in calls}) == 2


def test_a_wrap_only_app_is_offered_the_saving(layout):
    client = wrap(FakeSDK(), store=layout, tenant=TENANT)
    _interleaved(client, n=10)
    client.close()
    host = json.loads((layout.costshape_dir(TENANT) / "host.json").read_text())
    rep = cs.report(layout, TENANT, cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in host.items()}),
                    since="30d")
    lv = cs.lever_breakpoint_add(rep)
    assert lv is not None and lv.saves_usd > 0 and lv.evidence["reused_tokens"] > 0
    assert cs.eligible(rep, lv)[0], cs.eligible(rep, lv)                   # 30 calls outside any run: ready


def test_the_host_run_closes_when_the_process_exits(tmp_path):
    script = tmp_path / "app.py"
    script.write_text(f"""import sys
sys.path[:0] = [{str(HERE)!r}]
from test_stranger import FakeSDK
from recant import wrap
c = wrap(FakeSDK(), store={str(tmp_path / 's')!r}, tenant="t")
c.messages.create(model="claude-sonnet-5-5", max_tokens=5, messages=[{{"role": "user", "content": "hi"}}])
""")
    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path)
    from recant.store import StoreLayout
    recs = [r for p in StoreLayout(tmp_path / "s").log_dir("t").glob("*.jsonl") for r in read_run(p, typed=False)]
    assert {r["kind"] for r in recs} >= {"run_started", "model_call", "run_finished"}


def test_a_continuation_that_branches_off_an_earlier_point_is_its_own_conversation(layout):
    """Two conversations open alike and then diverge: the second continuation does not join the first's conversation,
    so no reuse of the other's context is claimed for it."""
    client = wrap(FakeSDK(), store=layout, tenant=TENANT)
    opening = [{"role": "user", "content": "same opening " * 50}]
    for _ in range(2):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=5, messages=opening)
    for tail in ("first branch", "second branch"):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=5,
                               messages=opening + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": tail}])
    calls = [r for r in _records(layout) if r["kind"] == "model_call"]
    assert [c["extends_previous"] for c in calls] == [False, False, True, False]
    assert len({c["session"] for c in calls}) == 3                 # the second branch is a conversation of its own


def test_a_wrapper_let_go_is_freed_and_one_closed_is_not_closed_again(layout):
    import gc
    import weakref
    wrap_mod = sys.modules["recant.wrap"]                         # the module (recant.wrap is the function)
    client = wrap(FakeSDK(), store=layout, tenant=TENANT)
    client.messages.create(model="claude-sonnet-5-5", max_tokens=5, messages=[{"role": "user", "content": "hi"}])
    client.close()
    assert client not in wrap_mod._OPEN
    ref = weakref.ref(client)
    del client
    gc.collect()
    assert ref() is None
    wrap_mod._close_at_exit()                                     # nothing left to close: no error, nothing written
    finished = [r for r in _records(layout) if r["kind"] == "run_finished"]
    assert len(finished) == 1

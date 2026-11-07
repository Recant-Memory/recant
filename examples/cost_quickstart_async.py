"""The cost quickstart on the async client, streamed: the same support agent as `cost_quickstart.py` (the same handbook,
questions and model), written on `anthropic.AsyncAnthropic` with the `messages.stream(...)` helper and several tasks in
flight at once. The only Recant line is the wrap; each task is its own run (`async with client.run(...)`), so
concurrent tasks keep their own logs, and the levers Recant applies are carried out on every streamed request.

    pip install "recant[anthropic]"
    ANTHROPIC_API_KEY=... python examples/cost_quickstart_async.py --tasks 30    # nothing applied yet; store ./.recant
    recant cost                                                                 # spent, can save, measured on
    recant apply                                                                # turn the safe savings on
    ANTHROPIC_API_KEY=... python examples/cost_quickstart_async.py --tasks 50    # the same code, levers carried out
    recant cost --nightly                                                       # check: realized against predicted
    recant cost"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import recant
from cost_quickstart import HANDBOOK, ITEMS, MODEL, QUESTIONS, TENANT       # the same agent as the sync quickstart


async def conversation(client, i: int) -> list[str]:
    """One support conversation: six questions, each streamed with the whole conversation so far."""
    msgs, replies = [], []
    for q in QUESTIONS:
        msgs.append({"role": "user", "content": q.format(n=40_000 + i, item=ITEMS[i % len(ITEMS)], d=10 + i % 40)})
        async with client.messages.stream(model=MODEL, max_tokens=1024, system=HANDBOOK, messages=msgs) as s:
            replies.append(await s.get_final_text())
        msgs.append({"role": "assistant", "content": replies[-1] or "…"})
    return replies


async def task(client, i: int) -> None:
    async with client.run("agent", session=f"task-{i}"):
        await conversation(client, i)


async def run_task(t: dict, client) -> list[str]:           # `recant bench`: the same agent, async and streamed
    return await conversation(client, int(t["i"]))


from cost_quickstart import grade  # noqa: E402,F401 — the same minimal check


async def run(client, start: int, n: int, *, parallel: int, mark: Path) -> None:
    gate = asyncio.Semaphore(parallel)
    done = []

    async def one(i):
        async with gate:
            try:
                await task(client, i)
            except Exception as exc:                                     # one dropped stream must not end the others
                print(f"task {i} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            done.append(i)
            mark.write_text(str(start + len(done)))
    await asyncio.gather(*(one(i) for i in range(start, start + n)))


def main(argv=None, sdk=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", nargs="?", help="the store; default Recant's own ($RECANT_HOME or ./.recant)")
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=5, help="tasks in flight at once")
    a = ap.parse_args(argv)
    if sdk is None:
        import anthropic
        sdk = anthropic.AsyncAnthropic(timeout=anthropic.Timeout(120.0, connect=10.0))   # your own idle timeout: the SDK's 600 s does not cut a stream that goes silent
    client = recant.wrap(sdk, store=a.store)              # the one line
    mark = client.layout.root / "cost_quickstart.next"
    start = int(mark.read_text()) if mark.exists() else 0
    asyncio.run(run(client, start, a.tasks, parallel=a.parallel, mark=mark))
    print(f"{a.tasks} tasks run; levers carried out on the last call: {client.last_sent_markers or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

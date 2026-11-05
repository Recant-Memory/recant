"""Managed notes, Recant's default memory, in a complete agent on the raw Anthropic SDK. Two
sessions: in the first the principal states some facts and the agent records them with the `note_fact` tool; Recant
checks each entry against the session's own lines and keeps what passes. The second session starts with the notes in
its prefix, and the agent answers from them.

    pip install "recant[anthropic]"
    ANTHROPIC_API_KEY=... python examples/notes_agent.py STORE_DIR       # a real model
    python examples/notes_agent.py STORE_DIR --offline                    # no key: a scripted client, same code path

What the host does, and nothing more: open a session file and append each turn to it; at a session start, put
`notes.INSTRUCTION` and the block `notes.session_start` returns into the system prompt; offer `notes.TOOL_SPECS`; hand
each call to one of them to `notes.handle` and return what it says. Prices default to the library's list prices for the
model, tokens are counted with the client's own counter, and the switch records go to a log Recant opens."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from recant import Store, StoreLayout, episodes, notes
from recant.machinery import Machinery

TENANT = "00000000-0000-0000-0000-000000000001"          # Recant partitions everything by tenant; one is fine
PRINCIPAL = "me"
MODEL = "claude-sonnet-5-5"
CHARACTER = "You are a concise assistant for a small business owner."

SESSIONS = [
    ("2026-01-05", ["Some housekeeping: our accountant is Alice Example, and the shop rent is 3,300 a month from February."]),
    ("2026-01-06", ["Who is our accountant again, and what is the rent?"]),
]


def run_session(client, layout, store, machinery, day: str, messages: list[str]) -> list[str]:
    """One session: the notes block in the prefix, the notes tools offered, each tool call handled by Recant."""
    block = notes.session_start(layout, TENANT, store, principal_id=PRINCIPAL, serve=machinery.kernel.serve, today=day,
                                block=f"user:{PRINCIPAL}", model=MODEL, client=client)
    system = CHARACTER + "\n\n" + notes.INSTRUCTION + ("\n\n" + block["text"] if block and block["text"] else "")
    session = episodes.open_session(layout, TENANT, day)          # Recant keeps the file; the tool's evidence is in it
    history, replies = [], []
    for i, said in enumerate(messages):
        session.append("user", said, who=PRINCIPAL, when=f"{day}T09:{i:02d}:00+00:00")
        history.append({"role": "user", "content": said})
        while True:
            r = client.messages.create(model=MODEL, max_tokens=1024, system=system, tools=notes.TOOL_SPECS, messages=history)
            history.append({"role": "assistant", "content": r.content})
            said_back = "".join(b.text for b in r.content if b.type == "text").strip()
            if said_back:
                session.append("assistant", said_back)
            calls = [b for b in r.content if b.type == "tool_use"]
            if not calls:
                replies.append(said_back)
                break
            results = []
            for call in calls:
                out = notes.handle(call.name, call.input, layout=layout, tenant=TENANT, today=day, principal_id=PRINCIPAL,
                                   lines=episodes.read_session(layout, TENANT, session.episode_path), episode_path=session.episode_path)
                session.append("tool", f"{call.name}: {out.splitlines()[0]}")
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": out})
            history.append({"role": "user", "content": results})
    return replies


def run(store_dir: Path, client, sessions=SESSIONS) -> dict:
    layout = StoreLayout(store_dir)
    layout.ensure(TENANT)
    Machinery.seed_defaults(layout)                                # the serve settings (notes threshold) among Recant's defaults
    machinery = Machinery.load(layout)
    store = Store(layout, list(machinery.kernel.corpus.allowlist) + list(episodes.HOT_PATTERNS))
    replies = [run_session(client, layout, store, machinery, day, msgs) for day, msgs in sessions]
    kept = notes.NotesStore(layout, TENANT, f"user:{PRINCIPAL}").entries()
    return {"replies": replies, "notes": [f"{e.scope}/{e.entity or '-'}/{e.attribute} = {e.value}" for e in kept]}


class OfflineClient:
    """Replace this with `anthropic.Anthropic()`. Scripted: in a session where the principal states facts it calls
    note_fact with them (and one value they never said, which Recant refuses); otherwise it answers from the prefix."""

    def __init__(self):
        self.messages = self

    def create(self, *, model, max_tokens, system, tools, messages):
        last = messages[-1]["content"]
        block = lambda **kw: SimpleNamespace(**kw)                                                    # noqa: E731
        if isinstance(last, list):                                     # the tool results came back: reply
            return SimpleNamespace(content=[block(type="text", text="Noted.")])
        if "accountant is" in last:
            entries = [{"scope": "company", "attribute": "accountant", "value": "Alice Example", "source": "principal"},
                       {"scope": "company", "attribute": "rent", "value": "3,300 a month", "as_of": "2026-02-01", "source": "principal"},
                       {"scope": "company", "attribute": "auditor", "value": "Sample & Partners", "source": "principal"}]
            return SimpleNamespace(content=[block(type="tool_use", id="t1", name="note_fact", input={"entries": entries})])
        known = "Alice Example" in system and "3,300" in system
        return SimpleNamespace(content=[block(type="text", text="Alice Example; the rent is 3,300 a month." if known else "I don't know.")])


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    if offline:
        c = OfflineClient()
    else:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("set ANTHROPIC_API_KEY (and pip install \"recant[anthropic]\"), or pass --offline")
        import anthropic
        c = anthropic.Anthropic(timeout=anthropic.Timeout(120.0, connect=10.0))          # your own idle timeout (the SDK's default is 600 s)
    out = run(Path(sys.argv[1]), c)
    for (day, msgs), replies in zip(SESSIONS, out["replies"]):
        for m, r in zip(msgs, replies):
            print(f"[{day}] principal: {m}\n[{day}] agent: {r}")
    print("notes kept:", out["notes"])

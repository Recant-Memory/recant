"""The whole loop in one file: write a day's conversation through Recant, extract it into
proposals, review them, fit what was kept to a budget. Nothing here belongs to any one consumer; nothing here caps
spend — that is yours.

    pip install "recant[anthropic]"
    ANTHROPIC_API_KEY=... python examples/memory_v2_agent.py STORE_DIR     # real models, Recant's default machinery
    python examples/memory_v2_agent.py STORE_DIR --offline                  # no key: a placeholder client, no memories

This is memory v2, the fallback mode (extraction and retrieval). The default, managed notes, is examples/notes_agent.py;
the cost saver on its own is examples/cost_quickstart.py.

The store is seeded with Recant's default machinery (`Machinery.seed_defaults`): working, host-neutral prompts,
Sonnet 5 with reasoning off, the default budgets. Copy recant/defaults/*.yaml to change any of it. With a key, the
reference client (`recant.clients.anthropic.AnthropicClient`) runs extraction for a few cents; any object with
`complete(request) -> Completion` works in its place (`count_tokens` and a batch path are optional).

Your agent decides when a session starts and ends and what a line says; Recant owns the file. Mark a tool call that
itself failed with `append("tool", ..., failed=True)`: extraction keeps no memory resting only on such a failure."""
from __future__ import annotations

import sys
from pathlib import Path

import recant
from recant import Ledger, Store, StoreLayout, budget, episodes
from recant.log import LogWriter
from recant.machinery import Machinery
from recant.provider import Completion, ModelRequest, Usage, text
from recant.store.review import decide_memory, memory_queue
from recant.store.search import load_entries

TENANT = "00000000-0000-0000-0000-000000000001"          # Recant partitions everything by tenant; one is fine

# A day of conversation, as your agent would append it turn by turn: (role, who, text).
TURNS = [
    ("user", "me", "We decided: the scanner ships on Fridays from now on."),
    ("assistant", None, "Noted. Fridays it is."),
    ("user", "me", "Monday's outage was caused by the stale cache key."),
    ("assistant", None, "Understood: the stale cache key."),
    ("user", "me", "Remind me to rotate the cache keys next week."),
]


class MyClient:
    """Replace this. The example client answers every request with an empty, well-formed completion."""

    def complete(self, request: ModelRequest) -> Completion:
        return Completion(content=(text("{}"),), stop_reason="end_turn", usage=Usage(input=0, output=0))


def run(store_dir: Path, client, *, day: str = "2026-01-05", turns=TURNS, principal_id: str = "me",
        question: str = "what did we decide?") -> dict:
    layout = StoreLayout(store_dir)
    layout.ensure(TENANT)
    Machinery.seed_defaults(layout)                                                 # Recant's defaults, once; or seed your own
    m = Machinery.load(layout)
    allow = list(m.kernel.corpus.allowlist) + list(episodes.HOT_PATTERNS)          # memories may anchor in sessions
    store, ledger = Store(layout, allow), Ledger(layout, allow)

    # 1. Write. Your agent appends each turn as it happens; Recant numbers the lines episode ids will cite.
    session = episodes.open_session(layout, TENANT, day)
    for i, (role, who, said) in enumerate(turns):
        session.append(role, said, who=who, when=f"{day}T09:{i:02d}:00+00:00" if role == "user" else None)

    # 2. Extract. Schedule this yourself (nightly, say). Proposals go to the review queue; tasks to the ledger.
    result = recant.extract(layout, TENANT, day, machinery=m, client=client, store=store, ledger=ledger,
                            principal_id=principal_id)
    proposals = [r for r in memory_queue(store, TENANT, principal_id=principal_id) if r["state"] == "proposed"]
    for row in proposals:
        print(f"proposed: {row['claim']}  [{row['episode']}]")

    # 3. Review. A person, or your own reviewer; nothing proposed is served until it is decided.
    with LogWriter(layout, TENANT, "review", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
        for row in proposals:
            decide_memory(store, TENANT, log, row["id"], to_state="probation", reason="confirmed",
                          reviewer={"kind": "human", "id": principal_id}, reasons=m.artifact.review_reasons)

    # 4. Serve. What was kept, fitted to your budget in a stable order so the prompt prefix caches.
    entries = load_entries(store, TENANT, "as_reviewed", m.artifact.trust_rank)
    served = budget.fit(entries, family=None, serve=m.kernel.serve, view="as_reviewed", question=question)
    return {"session": session.episode_path, "lines": session.lines, "spans": result["spans"],
            "proposed": len(proposals), "tasks": len(ledger.list(TENANT)), "usage": result["usage"], "entries": len(entries),
            "core_tokens": served["core_tokens"], "slice_tokens": served["slice_tokens"],
            "prefix": served["core_text"] + "\n\n" + served["slice_text"]}


def client():
    """The reference Anthropic client when a key is set; otherwise None."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from recant.clients.anthropic import AnthropicClient
    return AnthropicClient()


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    c = None if offline else client()
    if c is None and not offline:
        sys.exit("set ANTHROPIC_API_KEY (and pip install \"recant[anthropic]\"), or pass --offline")
    out = run(Path(sys.argv[1]), c or MyClient())
    print({k: v for k, v in out.items() if k != "prefix"})
    print(out["prefix"])

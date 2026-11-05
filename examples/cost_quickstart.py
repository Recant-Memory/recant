"""See what your agent costs, and let Recant carry out the safe savings. A small support agent
written on the raw Anthropic SDK with no prompt caching set up: a long handbook in the system prompt and a few
follow-up questions per task. The only Recant line is the wrap; every call is metered into Recant's log, and the
levers Recant applies (cache breakpoints, cache lifetime) are carried out on the next request with no other change.

    pip install "recant[anthropic]"
    ANTHROPIC_API_KEY=... python examples/cost_quickstart.py STORE_DIR --tasks 30    # shadow: nothing applied yet
    recant cost --store STORE_DIR --tenant demo                                       # what it costs; the levers, priced
    recant cost --store STORE_DIR --tenant demo --nightly                             # apply what is safe and pays
    ANTHROPIC_API_KEY=... python examples/cost_quickstart.py STORE_DIR --tasks 50    # the same code, levers carried out
    recant cost --store STORE_DIR --tenant demo --nightly                             # check: realized against predicted
    recant cost --store STORE_DIR --tenant demo

Add --offline to run without a key (a placeholder client that bills cached reads as the API does). `recant cost` reads
only the store: it calls nothing and sends nothing. A lever is checked after 50 runs (or seven days) since it was applied,
which is why the second pass is 50 tasks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import recant

TENANT = "demo"
MODEL = "claude-sonnet-5-5"

# Your agent's own prompt. Long and stable, as most agents' instructions and reference material are.
HANDBOOK = "You are the support agent for Northwind Hardware, an online shop. Answer in two or three sentences.\n\n" + "\n\n".join(
    f"Policy {i + 1}: {topic}. Customers who ask about {topic.lower()} get the standard answer: {rule} If the order is "
    f"older than {30 + 5 * i} days, or the item was marked final sale, refer them to the returns desk and give the "
    f"case code NW-{100 + i}. Never promise a date the warehouse has not confirmed; quote the tracking page instead."
    for i, (topic, rule) in enumerate([
        ("Returns", "unused items come back free within 30 days with the original packing slip."),
        ("Refunds", "refunds go to the original payment method within five business days of the return scan."),
        ("Shipping", "standard shipping is free over $50 and takes three to five business days."),
        ("Express delivery", "express costs $14 and ships the same day for orders placed before 2 pm."),
        ("Damaged goods", "send a photo within 48 hours and a replacement ships at once, no return needed."),
        ("Warranty", "power tools carry a two-year warranty; hand tools are covered for life."),
        ("Price matching", "we match a published price from a named competitor within 14 days of purchase."),
        ("Gift cards", "gift cards never expire and cannot be exchanged for cash."),
        ("Bulk orders", "orders over 100 units get a quote from the trade desk within one business day."),
        ("Account access", "password resets go to the email on file; we never reset over chat."),
        ("Order changes", "an order can be changed until it is picked, usually within two hours."),
        ("Cancellations", "a cancelled order is refunded in full if it has not shipped."),
        ("International", "we ship to Canada and Mexico; duties are collected at checkout."),
        ("Backorders", "a backordered item ships when stock arrives; the customer may cancel it at any time."),
        ("Installation", "we do not install; the product page lists certified installers by region."),
        ("Rentals", "tool rentals are by the day, with a deposit refunded on return."),
        ("Recalls", "a recalled item is refunded in full with or without a receipt."),
        ("Loyalty points", "points are earned at one per dollar and redeemed at checkout in steps of 500."),
        ("Store pickup", "pickup orders are held for seven days at the counter."),
        ("Invoices", "a VAT invoice is emailed with the shipping confirmation."),
    ]))

QUESTIONS = ["My order {n} arrived with a cracked {item}. What now?", "Can I still return it? It came {d} days ago.",
             "And how long until the money is back?", "Could you send the replacement express instead?",
             "Do I need to keep the packaging?", "Thanks. One more: do I earn points on the replacement?"]
ITEMS = ["drill", "level", "saw", "sander", "toolbox", "ladder", "lamp", "clamp"]


def conversation(client, i: int) -> list[str]:
    """One support conversation: six questions, each sent with the whole conversation so far, as most agents do."""
    msgs, replies = [], []
    for q in QUESTIONS:
        msgs.append({"role": "user", "content": q.format(n=40_000 + i, item=ITEMS[i % len(ITEMS)], d=10 + i % 40)})
        r = client.messages.create(model=MODEL, max_tokens=1024, system=HANDBOOK, messages=msgs)   # thinking counts against it
        replies.append("".join(b.text for b in r.content if b.type == "text"))
        msgs.append({"role": "assistant", "content": replies[-1] or "…"})
    return replies


def task(client, i: int) -> None:
    with client.run("agent", session=f"task-{i}"):                     # one log run per task (a wrapped-client call)
        conversation(client, i)


# `recant bench`: the bench opens each task's run and hands its own client; tasks: quickstart_tasks.jsonl
#     recant bench --tasks examples/quickstart_tasks.jsonl --run cost_quickstart:run_task --grade cost_quickstart:grade
def run_task(t: dict, client) -> list[str]:
    return conversation(client, int(t["i"]))


def grade(t: dict, replies: list[str]) -> bool:
    """A minimal check: every question got a reply with text in it (a capped or broken call leaves one empty). These
    tasks are too easy to show a quality difference; give your own agent a real grader."""
    return len(replies) == len(QUESTIONS) and all(r.strip() for r in replies)


class OfflineSDK:
    """Replace with `anthropic.Anthropic()`. Bills a request as the API would, roughly: a marked request reads what an
    earlier one wrote."""

    def __init__(self):
        self.messages, self.cached = self, set()

    def create(self, *, model, max_tokens, system, messages, **_):
        toks = lambda x: len(json.dumps(x)) // 4                       # noqa: E731
        total = toks(system) + toks(messages)
        marked = isinstance(system, list) and "cache_control" in system[-1]
        hit = toks(system) + toks(messages[:-1]) if marked and (len(messages) > 1 or "sys" in self.cached) else 0
        if marked:
            self.cached.add("sys")
        usage = SimpleNamespace(input_tokens=total - hit if not marked else 20, output_tokens=60, cache_read_input_tokens=hit,
                                cache_creation_input_tokens=max(0, total - hit - 20) if marked else 0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Send a photo within 48 hours.")], usage=usage,
                               stop_reason="end_turn", model=model)

    def count_tokens(self, **kw):
        return SimpleNamespace(input_tokens=len(json.dumps(kw)) // 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--start", type=int, default=None, help="first task number (default: after the last one run)")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    if a.offline:
        sdk = OfflineSDK()
    else:
        import anthropic
        sdk = anthropic.Anthropic(timeout=anthropic.Timeout(120.0, connect=10.0))        # your own idle timeout (the SDK's default is 600 s); retries are yours too
    client = recant.wrap(sdk, store=a.store, tenant=TENANT)            # the one line: meter every call, carry out the levers
    mark = Path(a.store) / "cost_quickstart.next"
    start = a.start if a.start is not None else int(mark.read_text()) if mark.exists() else 0
    for i in range(start, start + a.tasks):
        task(client, i)
        mark.write_text(str(i + 1))
    print(f"{a.tasks} tasks run; levers carried out on the last call: {client.last_sent_markers or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

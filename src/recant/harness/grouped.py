"""Memory v2's extractor: nearby moments of one episode in one bounded request.

The shared context is shown once; the answer is lists — memories, tasks (when the kernel's `one_pass` is on and a
ledger is given) and declined moments — each item tagged with its moment and anchored on its own lines. Every item
goes through the same checks as the one-moment extractor (`Extractor.build`, `TaskExtractor.build`).

Safeguards: an answer cut off by the output cap, or one that does not parse, is split in half and retried;
a moment the answer leaves out is retried alone; a moment still without a decision is logged as such and left
undecided, so a later run takes it again. Nothing marks a moment decided except a decision about it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..records import KINDS
from .common import numbered, obj, parse_json, request, run_all
from .extract import CROSS_REF_LINES, NOT_A_MEMORY, Extractor, Target, TaskExtractor, cited_roles, day_of, decided_spans, group_spans

GROUP_MAX_TOKENS = 8000          # several moments' lists share one answer; a cut-off answer is split, never lost
ROUNDS = 4                       # retry rounds: a split group or a left-out moment goes again, at most this often
VALUE_STATUS = ("current", "historical", "concurrent")


def _memory_item() -> dict:
    return obj({"moment": {"type": "string"}, "claim": {"type": "string"}, "kind": {"type": "string", "enum": [*KINDS]},
                "actor_class": {"type": "string", "enum": ["principal", "agent"]}, "actor_id": {"type": "string"},
                "quote": {"type": "string"}, "quote_start_line": {"type": "integer"}, "quote_end_line": {"type": "integer"},
                "timestamp": {"type": "string"}, "scope": {"type": "string"}, "entity": {"type": "string"},
                "attribute": {"type": "string"}, "value_status": {"type": "string", "enum": list(VALUE_STATUS)},
                "supersession_condition": {"type": "string"}, "notes": {"type": "string"}})


def _task_item() -> dict:
    return obj({"moment": {"type": "string"}, "claim": {"type": "string"}, "scope": {"type": "string"},
                "actor_class": {"type": "string", "enum": ["principal", "agent"]}, "actor_id": {"type": "string"},
                "quote": {"type": "string"}, "quote_start_line": {"type": "integer"}, "quote_end_line": {"type": "integer"},
                "status": {"type": "string", "enum": ["open", "done", "dropped"]}, "closed_line": {"type": "integer"},
                "timestamp": {"type": "string"}, "notes": {"type": "string"}})


def grouped_schema(one_pass: bool) -> dict:
    """memories, tasks (one pass only) and declined: a moment with nothing to keep is declined with its reason; in
    two passes a task-shaped moment is declined as `task`, for the ledger's own pass."""
    reasons = [r for r in NOT_A_MEMORY if r != "task" or not one_pass]
    props = {"memories": {"type": "array", "items": _memory_item()}}
    if one_pass:
        props["tasks"] = {"type": "array", "items": _task_item()}
    props["declined"] = {"type": "array", "items": obj({"moment": {"type": "string"}, "reason": {"type": "string", "enum": reasons}})}
    return obj(props)


@dataclass
class Group:
    path: str
    moments: list[Target]
    a: int
    b: int


def make_groups(targets: list[Target], n_lines, context: int, max_moments: int, max_lines: int) -> list[Group]:
    """Moments of one episode whose windows (their lines ± `context`) overlap or touch go together, up to
    `max_moments` and `max_lines` shown; anything further starts a new group."""
    out: list[Group] = []
    for t in sorted(targets, key=lambda t: (t.path, t.start, t.end)):
        a, b = max(1, t.start - context), min(n_lines(t.path), t.end + context)
        g = out[-1] if out else None
        if g and g.path == t.path and a <= g.b + 1 and len(g.moments) < max_moments and max(b, g.b) - g.a + 1 <= max_lines:
            g.moments.append(t)
            g.b = max(g.b, b)
        else:
            out.append(Group(t.path, [t], a, b))
    return out


class GroupedExtractor(Extractor):
    def __init__(self, *args, ledger=None, **kw):
        super().__init__(*args, **kw)
        self.one_pass = bool(self.rules is not None and self.rules.one_pass and ledger is not None
                             and self.m.artifact.extractor.tasks is not None)
        self.tx = None
        if self.one_pass:
            self.tx = TaskExtractor(self.layout, self.tenant, self.m, ledger, self.provider, self.log, principal_id=self.principal_id,
                                    batch=self.batch, wait=self.wait, dry_run=self.dry_run, agent_id=self.agent_id, model=self.model)

    def _keys_block(self) -> list[str]:
        if self.rules is None or not self.rules.keys:
            return []
        keys = sorted(self.keyed)[:200]
        return ["", "Keys already in the store (scope::entity::attribute — current claim):" if keys else "Keys already in the store: none yet.",
                *(f"- {k} — {self.keyed[k][1][:120]}" for k in keys)]

    def group_text(self, g: Group, ids: list[str]) -> str:
        descs = {m.name: m.description for m in self.m.artifact.markers}
        lines = self.lines(g.path)
        head = [f"Episode: {g.path}", f"Day: {day_of(g.path) or 'unknown'}", f"Lines shown: {g.a}-{g.b}.", "",
                numbered(lines[g.a - 1:g.b], g.a), "", "Moments to decide (each marked by nomination):"]
        for mid, t in zip(ids, g.moments):
            head.append(f"- {mid}: lines {t.start}-{t.end}; markers: " + "; ".join(f"{m} ({descs.get(m, '')})" for m in t.markers))
        cross, seen = [], set()
        for t in g.moments:
            for c in t.cross:
                if c not in seen and len(cross) < CROSS_REF_LINES:
                    seen.add(c)
                    cross.append(c)
        if cross:
            head += ["", "Same-day lines from the other source that share words with a moment "
                         "(for context; every anchor must stay in the episode above):", *(f"{p}#L{n}: {line}" for p, n, line in cross)]
        return "\n".join(head + self._keys_block())

    def group_request(self, g: Group, cid: str):
        ids = [f"m{i + 1}" for i in range(len(g.moments))]
        return request("extract", model=self.model, system=self.ex.prompt, user=self.group_text(g, ids),
                       schema=grouped_schema(self.one_pass), max_tokens=GROUP_MAX_TOKENS, custom_id=cid,
                       reasoning=self.ex.reasoning)

    def run(self, spans) -> dict:
        targets = [self.prepare(t) for t in group_spans(spans)]
        skip = set() if self.dry_run else decided_spans(self.layout, self.tenant, self.fingerprint, job=self.log.job)
        todo = [t for t in targets if t.episode not in skip]
        self.log.count("spans_in", len(spans))
        self.log.count("moments", len(targets))
        self.log.count("already_decided", len(targets) - len(todo))
        r = self.rules
        groups = make_groups(todo, lambda p: len(self.lines(p)), self.context, r.group_max_moments, r.group_max_lines)
        out = {"calls": 0, "groups": len(groups), "written": 0, "accepted": 0, "rejected": 0, "candidates": [],
               "split": 0, "retried_alone": 0, "undecided": 0}
        tasks_out = {"calls": 0, "written": 0, "closed": 0, "accepted": 0, "rejected": 0, "candidates": []}
        rechecked: set[str] = set()                          # stated facts declined in a group, re-asked alone once
        out["rechecked_alone"] = 0
        task_items: list[tuple[Target, dict]] = []
        pending, n = groups, 0
        for rnd in range(ROUNDS):
            if not pending:
                break
            reqs, meta = [], {}
            for g in pending:
                cid = f"g{n}"
                n += 1
                meta[cid] = g
                reqs.append(self.group_request(g, cid))
            out["calls"] += len(reqs)
            pending = []
            for cid, resp in run_all(self.provider, reqs, batch=self.batch, wait=self.wait):
                g = meta[cid]
                self.log.examined += 1
                d, err = parse_json(resp)
                if err or not isinstance(d, dict):
                    if len(g.moments) > 1 and rnd < ROUNDS - 1:      # cut off or malformed: split and retry
                        half = len(g.moments) // 2
                        for part in (g.moments[:half], g.moments[half:]):
                            pending.append(self._regroup(g.path, part))
                        out["split"] += 1
                        self.log.count("group_split")
                        continue
                    for t in g.moments:
                        self._log(self._shown(t, g), {"raw": None}, verdict="rejected", reason=f"malformed: {err or 'not an object'}")
                        out["rejected"] += 1
                    continue
                by_id = {f"m{i + 1}": self._shown(t, g) for i, t in enumerate(g.moments)}
                decided: set[str] = set()
                for item in d.get("memories") or []:
                    t = by_id.get(str(item.get("moment")))
                    if t is None:
                        self.log.count("item_unknown_moment")
                        continue
                    decided.add(item["moment"])
                    self._decide(t, {**item, "verdict": "memory", "not_a_memory_reason": "none"}, out)
                for item in (d.get("tasks") or []) if self.one_pass else []:
                    t = by_id.get(str(item.get("moment")))
                    if t is None:
                        self.log.count("item_unknown_moment")
                        continue
                    decided.add(item["moment"])
                    task_items.append((t, {**item, "verdict": "task", "not_a_task_reason": "none"}))
                for item in d.get("declined") or []:
                    mid = str(item.get("moment"))
                    t = by_id.get(mid)
                    if t is None or mid in decided:
                        continue                                       # a moment with an item is not also declined
                    src = g.moments[int(mid[1:]) - 1]
                    if (len(g.moments) > 1 and rnd < ROUNDS - 1 and "stated_fact" in src.markers and src.episode not in rechecked
                            and any(r == "user" for r, _ in cited_roles(self.lines(src.path), src.start, src.end))):
                        # a principal's stated fact declined inside a group, the agent's later
                        # doubt about it in view; asked once more alone, with its own lines only
                        rechecked.add(src.episode)
                        decided.add(mid)
                        pending.append(self._regroup(g.path, [src]))
                        out["rechecked_alone"] += 1
                        self.log.count("stated_fact_rechecked_alone")
                        continue
                    decided.add(mid)
                    why = item.get("reason") if item.get("reason") in NOT_A_MEMORY else "no_proposition"
                    self.log.count(f"not_a_memory: {why}")
                    self._log(t, item, verdict="rejected", reason=f"not_a_memory: {why}")
                    out["rejected"] += 1
                missing = [g.moments[i] for i in range(len(g.moments)) if f"m{i + 1}" not in decided]
                for t in missing:
                    if len(g.moments) > 1 and rnd < ROUNDS - 1:      # left out of a group: once more, alone
                        pending.append(self._regroup(g.path, [t]))
                        out["retried_alone"] += 1
                        self.log.count("moment_retried_alone")
                    else:                                             # no decision: left undecided for a later run
                        out["undecided"] += 1
                        self.log.count("moment_without_decision")
        for t, item in sorted(task_items, key=lambda x: (x[0].path, x[0].start)):   # a close must see what it closes
            self.tx.apply(t, item, tasks_out)
        if self.one_pass:
            out["tasks"] = tasks_out
        return out

    def _regroup(self, path: str, moments: list[Target]) -> Group:
        n = len(self.lines(path))
        return Group(path, moments, max(1, min(t.start for t in moments) - self.context), min(n, max(t.end for t in moments) + self.context))

    @staticmethod
    def _shown(t: Target, g: Group) -> Target:
        """The moment as decided inside its group: an anchor may sit anywhere in the lines the group showed."""
        return replace(t, a=g.a, b=g.b)

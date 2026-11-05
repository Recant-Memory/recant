"""Tool loops, one definition for the advisor and for a host's runtime.

A loop is the same tool called with the same arguments `TOOL_LOOP_MIN` times in a run, or again after its own error:
the agent did not take the result in. The advisor's `fix.tool_loop` finding counts them after the fact; `ToolLoopGuard`
stops one as it happens — after `TOOL_LOOP_MIN` identical failed calls in a turn, the host stops offering that tool for
the rest of the turn. Removing a tool changes the tool list at the head of the prompt, so the turn's next call rewrites
the cached prefix once; against a loop of model calls that is cheap. Any host can use it: it sees names, inputs and
whether each call failed, nothing else."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Optional

TOOL_LOOP_MIN = 3


def args_hash(tool_input: Optional[dict[str, Any]]) -> str:
    """A tool call's arguments, hashed (no text kept): the identity of a repeat."""
    args = json.dumps(tool_input or {}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(args.encode()).hexdigest()[:16]


class ToolLoopGuard:
    """One per turn. `observe` each call's outcome; `allowed` says whether a tool is still offered."""

    def __init__(self, n: int = TOOL_LOOP_MIN):
        self.n = n
        self.failed: Counter = Counter()
        self.removed: set[str] = set()

    def allowed(self, name: str) -> bool:
        return name not in self.removed

    def observe(self, name: str, tool_input: Optional[dict], error: Optional[str]) -> Optional[str]:
        """Record one call. Returns the line to add to the tool's result when this call removed the tool."""
        if not error:
            return None
        key = (name, args_hash(tool_input))
        self.failed[key] += 1
        if self.failed[key] >= self.n and name not in self.removed:
            self.removed.add(name)
            return (f"{name} failed {self.n} times with the same arguments; it is no longer offered in this turn. "
                    "Carry on without it.")
        return None

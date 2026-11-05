"""The reader: one model, one prompt, for every arm. It answers from the material an arm gives it, or
says the record holds nothing relevant, in a form every arm's judge sees the same way."""
from __future__ import annotations

import re

from ..machinery import Machinery
from ..provider import ContentBlock, Message, ModelRequest, ModelResponse, ToolSpec, text

NOTHING_MARK = "NOTHING RELEVANT"
_NOTHING = re.compile(r"^\W*nothing relevant\b", re.I)

GREP_TOOL = ToolSpec(
    name="grep",
    description="Search the whole record for lines containing EVERY one of the given words or short phrases, "
                "case-insensitively. Returns matching lines with their file, line number and a little context. "
                "Use two to four specific terms; too many terms find nothing, one common word finds too much.",
    input_schema={"type": "object", "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
                  "required": ["terms"], "additionalProperties": False})   # no minItems/maxItems: the API rejects them on a strict tool


MEMORY_TOOL = ToolSpec(
    name="search_memory",
    description="Search the memory store for entries carrying EVERY one of the given words or short phrases, "
                "case-insensitively, across each entry's claim, anchor quote, scope and notes. Returns entries with "
                "their id, status, kind, who asserted them, scope, claim, anchor and supersession condition. "
                "Use two to four specific terms; too many terms find nothing, one common word finds too much.",
    input_schema={"type": "object", "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
                  "required": ["terms"], "additionalProperties": False})


def declared_nothing(answer: str) -> bool:
    return bool(_NOTHING.match(answer.strip()))


def system_blocks(machinery: Machinery, material: str | None, *, cache_material: bool = False,
                  ttl: str = "5m", tool_note: str = "", uncached_tail: str | None = None) -> tuple[ContentBlock, ...]:
    r = machinery.kernel.eval.reader
    prompt = r.prompt.replace("{nothing}", NOTHING_MARK) + (f"\n\n{tool_note}" if tool_note else "")
    blocks = [text(prompt, cache=not cache_material)]
    if material is not None:
        blocks.append(text(material, cache=cache_material, ttl=ttl))   # the breakpoint sits after the material
    if uncached_tail:
        blocks.append(text(uncached_tail))                             # per-call material after the breakpoint
    return tuple(blocks)


def request(machinery: Machinery, *, system: tuple[ContentBlock, ...], messages: tuple[Message, ...], job: str,
            custom_id: str, tools: tuple[ToolSpec, ...] = ()) -> ModelRequest:
    r = machinery.kernel.eval.reader
    return ModelRequest(job=job, model=machinery.artifact.reader.model, system=system, messages=messages,
                        max_tokens=r.max_tokens, tools=tools, effort=r.effort, custom_id=custom_id)


def answer_of(resp: ModelResponse) -> tuple[str, bool]:
    """(text, cut_off)."""
    return resp.text.strip(), resp.stop_reason == "max_tokens"

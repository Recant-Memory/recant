from __future__ import annotations

import json
from typing import Any, Iterable, Iterator

from ..provider import Message, ModelRequest, ModelResponse, ProviderError, text


def numbered(lines: list[str], start: int = 1) -> str:
    return "\n".join(f"{n}: {line}" for n, line in enumerate(lines, start))


def request(job: str, *, system: str, user: str, schema: dict, max_tokens: int, custom_id: str,
            tier=None, model=None, effort=None, reasoning=None) -> ModelRequest:
    return ModelRequest(job=job, tier=tier, model=model, system=(text(system, cache=True),),
                        messages=(Message("user", (text(user),)),), max_tokens=max_tokens,
                        output_schema=schema, custom_id=custom_id, effort=effort, reasoning=reasoning)


def run_all(provider, reqs: list[ModelRequest], *, batch: bool, wait
            ) -> Iterator[tuple[str, ModelResponse | ProviderError]]:
    """Send every request, by the batch path or one at a time. Yields by custom id. The batch path is taken only
    when it was asked for and the model client has one; otherwise live."""
    if not reqs:
        return
    if batch and getattr(provider, "can_batch", True):
        handle = provider.submit_batch(reqs)
        while provider.batch_status(handle) != "ended":
            wait()
        yield from provider.batch_results(handle)
    else:
        for r in reqs:
            try:
                yield r.custom_id, provider.call(r)
            except ProviderError as exc:
                yield r.custom_id, exc


def parse_json(resp: ModelResponse | ProviderError) -> tuple[Any, str | None]:
    """(data, error). A refusal, a provider error and malformed output all land here."""
    if isinstance(resp, ProviderError):
        return None, str(resp)
    if resp.stop_reason == "max_tokens":
        return None, "the model ran out of output tokens before it answered; raise the job's cap"
    try:
        return json.loads(resp.text), None
    except (ValueError, TypeError) as exc:
        return None, f"output is not JSON: {exc}"


def obj(properties: dict, required: Iterable[str] | None = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(required if required is not None else properties), "additionalProperties": False}

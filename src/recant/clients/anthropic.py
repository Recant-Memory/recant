"""The Anthropic model client: the reference implementation of `ModelClient`, installed by the `recant[anthropic]`
extra. The only module in the library that imports the SDK."""
from __future__ import annotations

import json
from typing import Any, Callable, Iterator, Optional

import anthropic

from ..coverage import through
from ..provider.base import BaseProvider
from ..provider.interface import Completion, ContentBlock, ModelRequest, ProviderError, Usage

_CACHE = {"type": "ephemeral"}


def _block(b: ContentBlock) -> dict[str, Any]:
    if b.kind == "text":
        out: dict[str, Any] = {"type": "text", "text": b.text or ""}
    elif b.kind == "tool_use":
        out = {"type": "tool_use", "id": b.tool_use_id, "name": b.tool_name, "input": b.tool_input or {}}
    else:
        out = {"type": "tool_result", "tool_use_id": b.tool_use_id, "content": b.text or ""}
        if b.is_error:
            out["is_error"] = True
    if b.cache:
        out["cache_control"] = dict(_CACHE) if b.cache_ttl == "5m" else {**_CACHE, "ttl": "1h"}
    return out


def build_params(model: str, req: ModelRequest, *, supports_effort: bool, can_disable_reasoning: bool = True) -> dict[str, Any]:
    """The request in the vendor's shape. Thinking follows `req.reasoning`: "off" disables it (refused, loudly, for a
    model that cannot turn it off), "on" is adaptive, unset leaves the model's default. Nothing is prefilled."""
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": req.max_tokens,
        "messages": [{"role": m.role, "content": [_block(b) for b in m.content]} for m in req.messages],
    }
    if req.reasoning == "off":
        from ..levers import thinking_off                # the off form differs by model (sonnet-5-5 takes "between_tools")
        off = thinking_off(model)
        if not can_disable_reasoning or off is None:
            raise ProviderError("bad_request", f"{model} cannot turn its reasoning off; the job asks for it off")
        params["thinking"] = dict(off)
    elif req.reasoning == "on":
        params["thinking"] = {"type": "adaptive"}
    if req.system:
        params["system"] = [_block(b) for b in req.system]
    if req.tools:
        params["tools"] = [{"name": t.name, "description": t.description,
                            "input_schema": t.input_schema, "strict": t.strict} for t in req.tools]
    output_config: dict[str, Any] = {}
    if req.output_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": req.output_schema}
    if req.effort is not None and supports_effort:
        output_config["effort"] = req.effort
    if output_config:
        params["output_config"] = output_config
    return params


def _raw(message: Any) -> Completion:
    blocks = []
    for b in message.content:
        if b.type == "text":
            blocks.append(ContentBlock(kind="text", text=b.text))
        elif b.type == "tool_use":
            tool_input = b.input if isinstance(b.input, dict) else json.loads(b.input)
            blocks.append(ContentBlock(kind="tool_use", tool_name=b.name, tool_input=tool_input,
                                       tool_use_id=b.id))
        # thinking and other block types carry nothing the caller reads
    u = message.usage
    details = getattr(u, "output_tokens_details", None)
    usage = Usage(input=u.input_tokens or 0, output=u.output_tokens or 0,
                  cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                  cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
                  reasoning=(getattr(details, "thinking_tokens", 0) or 0) if details is not None else 0,
                  cache_write_long=getattr(getattr(u, "cache_creation", None), "ephemeral_1h_input_tokens", 0) or 0)
    return Completion(content=tuple(blocks), stop_reason=message.stop_reason, usage=usage,
               request_id=getattr(message, "_request_id", None))


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, anthropic.RateLimitError):
        return ProviderError("rate_limit", str(exc), retryable=True)
    if isinstance(exc, anthropic.NotFoundError):
        return ProviderError("not_found", str(exc))
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return ProviderError("auth", str(exc))
    if isinstance(exc, anthropic.BadRequestError):
        return ProviderError("bad_request", str(exc))
    if isinstance(exc, anthropic.APIStatusError):
        return ProviderError("server" if exc.status_code >= 500 else "api", str(exc),
                             retryable=exc.status_code >= 500)
    if isinstance(exc, anthropic.APIConnectionError):
        return ProviderError("connection", str(exc), retryable=True)
    return ProviderError("unknown", str(exc))


def _transport_errors() -> tuple:
    """The HTTP library's transport errors (whichever the SDK uses). A stream that stalls mid-read raises one of these
    directly, not an SDK error: it is a connection failure, retryable like any other."""
    out = []
    for name in ("httpx2", "httpx"):
        try:
            out.append(__import__(name).TransportError)
        except (ImportError, AttributeError):
            pass
    return tuple(out)


_TRANSPORT = _transport_errors()
STREAM_ABOVE = 16_000                   # output tokens; the SDK's own limit is near 21k (10 minutes at its 128k-per-hour estimate)


class AnthropicClient:
    """The reference `ModelClient` (the `recant[anthropic]` extra). `supports_effort` says whether a model takes the
    effort setting; it is the caller's knowledge, not the SDK's."""

    def __init__(self, client: Any = None, *, supports_effort: Callable[[str], bool] = lambda model: True,
                 can_disable_reasoning: Callable[[str], bool] = lambda model: True):
        self.client = client if client is not None else anthropic.Anthropic()
        self.supports_effort = supports_effort
        self.can_disable_reasoning = can_disable_reasoning

    def _params(self, req: ModelRequest) -> dict[str, Any]:
        return build_params(req.model, req, supports_effort=self.supports_effort(req.model),
                            can_disable_reasoning=self.can_disable_reasoning(req.model))

    def complete(self, request: ModelRequest) -> Completion:
        params = self._params(request)
        try:
            with through():                             # Recant's own call: not traffic past it
                if params["max_tokens"] > STREAM_ABOVE:     # the SDK refuses a non-streaming call it expects to pass 10 minutes
                    with self.client.messages.stream(**params) as s:
                        return _raw(s.get_final_message())
                return _raw(self.client.messages.create(**params))
        except anthropic.APIError as exc:
            raise _translate(exc) from exc
        except _TRANSPORT as exc:
            raise ProviderError("connection", f"{type(exc).__name__}: {exc}", retryable=True) from exc

    def count_tokens(self, request: ModelRequest) -> int:
        params = self._params(request)
        params.pop("max_tokens")
        params.pop("output_config", None)
        try:
            with through():
                return self.client.messages.count_tokens(**params).input_tokens
        except anthropic.APIError as exc:
            raise _translate(exc) from exc

    def submit(self, requests: list[ModelRequest]) -> str:
        body = [{"custom_id": r.custom_id, "params": self._params(r)} for r in requests]
        try:
            with through():
                return self.client.messages.batches.create(requests=body).id
        except anthropic.APIError as exc:
            raise _translate(exc) from exc

    def poll(self, handle: str) -> Optional[Iterator]:
        if self.client.messages.batches.retrieve(handle).processing_status != "ended":
            return None
        return self._results(handle)

    def _results(self, batch_id: str) -> Iterator:
        for entry in self.client.messages.batches.results(batch_id):
            result = entry.result
            if result.type == "succeeded":
                yield entry.custom_id, _raw(result.message)
            else:
                err = getattr(result, "error", None)
                detail = " ".join(str(x) for x in (getattr(err, "type", None), getattr(err, "message", None)) if x) or result.type
                yield entry.custom_id, ProviderError(f"batch_{result.type}", detail,
                                                      retryable=result.type in ("expired", "errored"))


class AnthropicProvider(BaseProvider):
    """The client behind the library's metering wrapper. `supports_effort` says per model whether the effort setting
    is sent; the consumer knows its models, so it passes the predicate (default: every model takes it)."""

    def __init__(self, machinery, log, client: Any = None, *, prices=None, supports_effort: Callable[[str], bool] = lambda model: True,
                 can_disable_reasoning: Callable[[str], bool] = lambda model: True, key_label=None):
        super().__init__(machinery, log, AnthropicClient(client, supports_effort=supports_effort,
                                                         can_disable_reasoning=can_disable_reasoning), prices=prices,
                         key_label=key_label)


def raw_sdk(is_async: bool = False, *, idle_timeout_s: float = 120.0):
    """The vendor SDK's own client, for `recant bench` to hand a developer's `run_task`: sync or async, with an
    idle timeout of its own (the SDK's default read timeout, 600 s, does not cut a stream that goes silent)."""
    import anthropic
    t = anthropic.Timeout(idle_timeout_s, connect=10.0)
    return anthropic.AsyncAnthropic(timeout=t) if is_async else anthropic.Anthropic(timeout=t)

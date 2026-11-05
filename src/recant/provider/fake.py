"""A provider with no network, for tests and dry runs. It records what it was
asked to send, which is how a test proves that nothing was."""
from __future__ import annotations

from typing import Callable, Iterator, Optional

from .base import BaseProvider, Raw
from .interface import ModelRequest, ProviderError, Usage, text

Responder = Callable[[str, ModelRequest], "Raw | ProviderError"]


def echo(model: str, req: ModelRequest) -> Raw:
    return Raw(content=(text("ok"),), stop_reason="end_turn", usage=Usage(input=10, output=2))


class FakeProvider(BaseProvider):
    def __init__(self, machinery, log, responder: Optional[Responder] = None, *, prices=None):
        super().__init__(machinery, log, prices=prices)
        self.responder = responder or echo
        self.sent: list[tuple[str, ModelRequest]] = []
        self.batches: dict[str, list[tuple[str, str, ModelRequest]]] = {}

    def _send(self, model, req):
        self.sent.append((model, req))
        out = self.responder(model, req)
        if isinstance(out, ProviderError):
            raise out
        return out

    def _count_tokens(self, model, req):
        return 0

    def _submit(self, items):
        batch_id = f"fakebatch_{len(self.batches) + 1}"
        self.batches[batch_id] = list(items)
        self.sent.extend((model, req) for _, model, req in items)
        return batch_id

    def _status(self, batch_id):
        return "ended"

    def _results(self, batch_id) -> Iterator:
        for cid, model, req in reversed(self.batches[batch_id]):  # out of order, as real batches may be
            yield cid, self.responder(model, req)

"""recant: file-canonical memory with provenance and a lifecycle, a task ledger, extraction from episodes, the
eval harness that says whether it beat grep, and a budget fitter that returns memories in a stable order so a
prompt prefix caches. A memory library, not an agent framework: no agent loop, no model client of its own, no
spend or time limit of any kind. The consumer brings its runtime and its model client (`ModelClient`) and reads
usage back per call.

The package knows nothing about who consumes it. Tenant ids, corpus paths, allowlists, model ids, prices and the
trust table's contents are all supplied by the consumer as configuration (the machinery files).

Public API: `Store`, `Ledger`, `episodes`, `extract`, `harness`, `budget.fit`, `ModelClient`. Everything else is
private and may move.
"""

__version__ = "0.4.1"

from .provider.interface import ModelClient  # noqa: E402
from .store import MemoryStore as Store, TaskStore as Ledger, StoreLayout  # noqa: E402
from .store import episodes, budget  # noqa: E402
from . import harness  # noqa: E402
from .extraction import extract  # noqa: E402 — a day's sessions into proposals; Extractor is recant.harness.extract.Extractor
from . import costshape  # noqa: E402 — the host's cost shape from metered calls
from . import notes, levers  # noqa: E402 — managed notes; the safe levers on an Anthropic-shaped request
from .wrap import wrap  # noqa: E402 — the drop-in path for a host on the raw SDK

__all__ = ["ModelClient", "Store", "Ledger", "StoreLayout", "episodes", "budget", "extract", "harness", "costshape", "notes", "levers", "wrap"]

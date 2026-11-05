"""Secret-shaped strings are kept out of anything drafted and anything served."""
from __future__ import annotations

import re

# Secret-shaped strings never land in a key or a question. A short list of shapes, and an allowlist
# for the things that look like secrets and are not: git shas, UUIDs, ULIDs, hashes.
_SECRET = re.compile(r"""(
      sk-[A-Za-z0-9_-]{16,} | xox[abps]-[A-Za-z0-9-]{10,} | gh[pousr]_[A-Za-z0-9]{20,} | AKIA[0-9A-Z]{16}
    | gsk_(?:live|test)_[A-Za-z0-9]{8,} | eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}
    | (?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*[0-9])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])
    | (?i:(?:token|password|passwd|secret|api[_\ -]?key|bearer)\s*[:=]\s*)["']?[^\s"'`,;]{8,}
)""", re.X)
_NOT_SECRET = re.compile(r"^(?:[0-9a-f]{7,64}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9A-HJKMNP-TV-Z]{26})$", re.I)
REDACTED = "[redacted]"


def redact_secrets(text_: str) -> tuple[str, int]:
    hits = 0

    def sub(m):
        nonlocal hits
        if _NOT_SECRET.match(m.group(0)):
            return m.group(0)
        hits += 1
        return REDACTED
    return _SECRET.sub(sub, text_), hits

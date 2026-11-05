"""The reference `BillingSource`: the organization's usage report through the Anthropic Admin API,
per day, API key and model. Plain HTTP — these endpoints are not in the SDK. The admin key is the caller's, held in
memory for the requests and never logged, printed or kept; a local key is matched to its id through the partial hint
the API lists, without either being shown."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Optional

from ..billing import BilledUsage

API = "https://api.anthropic.com"


class AnthropicBilling:
    def __init__(self, admin_key: str, *, opener=None, timeout: float = 120.0):
        self._key = admin_key
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=timeout).read())

    def _get(self, path: str, params: list[tuple[str, str]]) -> dict:
        url = f"{API}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"x-api-key": self._key, "anthropic-version": "2023-06-01"})
        return json.loads(self._open(req))

    def usage(self, start_day: str, end_day: str) -> list[BilledUsage]:
        params = [("starting_at", f"{start_day}T00:00:00Z"), ("ending_at", f"{end_day}T00:00:00Z"), ("bucket_width", "1d"),
                  ("group_by[]", "api_key_id"), ("group_by[]", "model"), ("limit", "31")]
        out, page = [], None
        while True:
            data = self._get("/v1/organizations/usage_report/messages", params + ([("page", page)] if page else []))
            for bucket in data.get("data", []):
                day = bucket["starting_at"][:10]
                for r in bucket.get("results", []):
                    cc = r.get("cache_creation") or {}
                    out.append(BilledUsage(day=day, key_id=r.get("api_key_id") or "none", model=r.get("model") or "unknown",
                                           uncached=r.get("uncached_input_tokens") or 0,
                                           cache_write_5m=cc.get("ephemeral_5m_input_tokens") or 0,
                                           cache_write_1h=cc.get("ephemeral_1h_input_tokens") or 0,
                                           cache_read=r.get("cache_read_input_tokens") or 0, output=r.get("output_tokens") or 0))
            if not data.get("has_more"):
                return out
            page = data.get("next_page")

    def cost(self, start_day: str, end_day: str) -> list[dict]:
        """Dollars per day, workspace and model (the cost report cannot split by key)."""
        params = [("starting_at", f"{start_day}T00:00:00Z"), ("ending_at", f"{end_day}T00:00:00Z"),
                  ("group_by[]", "workspace_id"), ("group_by[]", "description")]
        out, page = [], None
        while True:
            data = self._get("/v1/organizations/cost_report", params + ([("page", page)] if page else []))
            for bucket in data.get("data", []):
                for r in bucket.get("results", []):
                    out.append({"day": bucket["starting_at"][:10], "workspace_id": r.get("workspace_id"), "model": r.get("model"),
                                "usd": float(r.get("amount") or 0) / 100.0, "token_type": r.get("token_type"),
                                "cost_type": r.get("cost_type")})
            if not data.get("has_more"):
                return out
            page = data.get("next_page")

    def key_ids(self, secrets: dict[str, str]) -> dict[str, str]:
        listed, after = [], None
        while True:
            data = self._get("/v1/organizations/api_keys", [("limit", "100")] + ([("after_id", after)] if after else []))
            listed += data.get("data", [])
            if not data.get("has_more"):
                break
            after = data.get("last_id")
        out = {}
        for label, secret in secrets.items():
            for k in listed:
                hint = k.get("partial_key_hint") or ""
                if "..." in hint:
                    head, tail = hint.split("...", 1)
                    if secret and secret.startswith(head) and secret.endswith(tail):
                        out[label] = k["id"]
                        break
        return out

    def workspaces_of(self, secrets: dict[str, str]) -> dict[str, Optional[str]]:
        """Each labelled key's workspace (None: the Default workspace)."""
        ids = self.key_ids(secrets)
        data = self._get("/v1/organizations/api_keys", [("limit", "100")])
        ws = {k["id"]: k.get("workspace_id") for k in data.get("data", [])}
        return {label: ws.get(kid) for label, kid in ids.items()}

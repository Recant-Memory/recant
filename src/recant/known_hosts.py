"""The model-API hosts Recant knows, one line each: a host, or a host suffix starting with a dot,
and the provider id it belongs to. `recant.coverage` counts the requests to these that did not go through Recant; a
request to any other host is counted only when its body looks like a model call, as `other`, with the domain hashed.
Adding a host is one line here."""
from __future__ import annotations

import re
from typing import Optional

HOSTS = {
    "api.anthropic.com": "anthropic",
    "api.openai.com": "openai",
    "openrouter.ai": "openrouter",
    "generativelanguage.googleapis.com": "google",          # Gemini API
    "aiplatform.googleapis.com": "google_vertex",
    ".openai.azure.com": "azure_openai",
    ".services.ai.azure.com": "azure_openai",
    "api.mistral.ai": "mistral",
    "api.groq.com": "groq",
    "api.together.xyz": "together",
    "api.together.ai": "together",
    "api.fireworks.ai": "fireworks",
    "api.deepseek.com": "deepseek",
    "api.x.ai": "xai",
    "api.z.ai": "zai",
    "open.bigmodel.cn": "zai",
}

# Hosts with a region in the middle, as patterns: one line each too.
PATTERNS = (
    (r"bedrock-runtime(-fips)?\.[a-z0-9-]+\.amazonaws\.com", "aws_bedrock"),
    (r"[a-z0-9-]+-aiplatform\.googleapis\.com", "google_vertex"),            # regional Vertex: us-central1-aiplatform…
)

PROVIDERS = tuple(sorted(set(HOSTS.values()) | {p for _, p in PATTERNS})) + ("other",)

# A model name's provider, for usage files that name the model and not the host: a vendor prefix (`z-ai/glm-4.6`,
# `~z-ai/glm-latest` through a router), else the model family.
_MODEL_FAMILIES = (
    (r"claude", "anthropic"), (r"(gpt|o\d|chatgpt|text-embedding)", "openai"), (r"gemini", "google"),
    (r"glm", "zai"), (r"(mistral|mixtral|codestral|magistral)", "mistral"), (r"deepseek", "deepseek"),
    (r"grok", "xai"),
)
_VENDORS = {"anthropic": "anthropic", "openai": "openai", "google": "google", "z-ai": "zai", "zai": "zai",
            "mistralai": "mistral", "deepseek": "deepseek", "x-ai": "xai"}


def provider_of_host(host: Optional[str]) -> Optional[str]:
    """The provider id of a known model-API host, else None."""
    h = (host or "").lower().rstrip(".")
    if h in HOSTS:
        return HOSTS[h]
    for k, v in HOSTS.items():
        if k.startswith(".") and h.endswith(k):
            return v
    for pat, v in PATTERNS:
        if re.fullmatch(pat, h):
            return v
    return None


def provider_of_model(model: Optional[str]) -> str:
    """The provider a model name belongs to, from its vendor prefix or its family; `other` when neither says."""
    m = (model or "").lower().lstrip("~")
    if "/" in m:
        vendor, _, m = m.partition("/")
        if vendor in _VENDORS:
            return _VENDORS[vendor]
    for pat, prov in _MODEL_FAMILIES:
        if re.match(pat, m):
            return prov
    return "other"


def route_of_model(model: Optional[str]) -> Optional[str]:
    """The router a model name says it went through: a `vendor/model` name is a router's naming (OpenRouter's)."""
    return "openrouter" if "/" in (model or "") else None

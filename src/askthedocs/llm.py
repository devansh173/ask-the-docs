"""Provider-agnostic LLM layer.

Every node in the graph talks to a ``BaseChatModel``; nothing below this module
knows which vendor answered. That is what lets the frontend offer a provider
dropdown and a "paste your own key" box, and it is why the generation code is
written against LangChain's chat interface rather than a single vendor SDK.

Keys supplied from the browser live only for the duration of the request that
carries them. They are never written to disk, never logged, and never put in a
trace - see ``api/main.py``, where the key is read off the request and dropped.
When no key is supplied the process environment is used, which is how the
deployed demo and the eval harness run.

Constructor kwargs are mapped per provider rather than relying on
``init_chat_model`` to normalise them, because the three cloud providers each
spell the key and the output-token cap differently.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    env_key: str | None
    key_field: str | None
    max_tokens_field: str | None
    models: tuple[str, ...]
    default_model: str
    package: str
    needs_key: bool = True
    notes: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        id="anthropic",
        label="Anthropic (Claude)",
        env_key="ANTHROPIC_API_KEY",
        key_field="api_key",
        max_tokens_field="max_tokens",
        models=(
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5",
        ),
        default_model="claude-opus-5",
        package="langchain_anthropic",
        notes="Default. Opus 5 for generation, Sonnet 5 for the grader nodes.",
    ),
    "google_genai": ProviderSpec(
        id="google_genai",
        label="Google (Gemini)",
        env_key="GOOGLE_API_KEY",
        key_field="google_api_key",
        max_tokens_field="max_output_tokens",
        models=(
            "gemini-3.1-pro-preview",
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ),
        default_model="gemini-3.5-flash",
        package="langchain_google_genai",
        notes=(
            "Free-tier quota is per model per day, so pointing LLM_MODEL and "
            "GRADER_MODEL at different models doubles the effective allowance. "
            "Avoid gemini-2.5-flash-lite: superseded, and capped at 20/day."
        ),
    ),
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI",
        env_key="OPENAI_API_KEY",
        key_field="api_key",
        max_tokens_field="max_tokens",
        models=("gpt-5.2", "gpt-5.2-mini", "gpt-5.1"),
        default_model="gpt-5.2",
        package="langchain_openai",
    ),
    "ollama": ProviderSpec(
        id="ollama",
        label="Ollama (local)",
        env_key=None,
        key_field=None,
        max_tokens_field="num_predict",
        models=("llama3.2", "qwen3", "mistral", "phi4"),
        default_model="llama3.2",
        package="langchain_ollama",
        needs_key=False,
        notes="Runs against a local Ollama server. No API key, no data leaves the machine.",
    ),
}

DEFAULT_PROVIDER = "anthropic"

# The canonical ids are the LangChain package names, which are not what people
# type. Accept the obvious names for each vendor so LLM_PROVIDER=gemini (or
# =claude, or =gpt) works instead of failing with "Unknown provider".
PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google_genai",
    "google": "google_genai",
    "googlegenai": "google_genai",
    "google-genai": "google_genai",
    "vertex": "google_genai",
    "claude": "anthropic",
    "gpt": "openai",
    "open-ai": "openai",
    "local": "ollama",
}


def resolve_provider(name: str | None) -> str:
    """Map a user-supplied provider name onto a registry id."""
    if not name:
        return DEFAULT_PROVIDER
    key = name.strip().lower().replace("_", "").replace(" ", "")
    if key in PROVIDERS:
        return key
    normalised = name.strip().lower().replace("_", "-").replace(" ", "-")
    for candidate in (key, normalised, normalised.replace("-", "")):
        if candidate in PROVIDER_ALIASES:
            return PROVIDER_ALIASES[candidate]
        if candidate in PROVIDERS:
            return candidate
    return name


class LLMConfigError(RuntimeError):
    """Raised when a provider cannot be constructed - surfaced to the UI as 400."""


def provider_catalog() -> list[dict[str, Any]]:
    """What the frontend renders in its provider picker.

    ``key_in_env`` tells the UI whether it can skip asking for a key.
    """
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "models": list(spec.models),
            "default_model": spec.default_model,
            "needs_key": spec.needs_key,
            "key_in_env": bool(spec.env_key and os.getenv(spec.env_key)),
            "notes": spec.notes,
        }
        for spec in PROVIDERS.values()
    ]


def _resolve_key(spec: ProviderSpec, api_key: str | None) -> str | None:
    if not spec.needs_key:
        return None
    key = (api_key or "").strip() or (os.getenv(spec.env_key) if spec.env_key else None)
    if not key:
        raise LLMConfigError(
            f"{spec.label} needs an API key. Paste one in the sidebar, "
            f"or set {spec.env_key} in the environment."
        )
    return key


def build_chat_model(
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **extra: Any,
) -> BaseChatModel:
    provider = resolve_provider(provider)
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise LLMConfigError(
            f"Unknown provider {provider!r}. Available: {', '.join(PROVIDERS)} "
            f"(aliases: {', '.join(sorted(PROVIDER_ALIASES))})."
        )

    key = _resolve_key(spec, api_key)
    kwargs: dict[str, Any] = {"model": model or spec.default_model, **extra}

    # Ollama has no temperature ceiling issues; the cloud providers all accept 0.
    kwargs["temperature"] = temperature
    if spec.max_tokens_field:
        kwargs[spec.max_tokens_field] = max_tokens
    if spec.key_field and key:
        kwargs[spec.key_field] = key

    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**kwargs)
        if provider == "google_genai":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(**kwargs)
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(**kwargs)
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            kwargs.setdefault("base_url", os.getenv("OLLAMA_BASE_URL",
                                                    "http://localhost:11434"))
            return ChatOllama(**kwargs)
    except ImportError as exc:
        raise LLMConfigError(
            f"{spec.label} support needs the {spec.package} package: {exc}"
        ) from exc
    except Exception as exc:
        raise LLMConfigError(f"Could not initialise {spec.label}: {exc}") from exc

    raise LLMConfigError(f"Unhandled provider {provider!r}")


@dataclass
class LLMBundle:
    """The two models a graph run uses.

    ``generator`` writes the answer; ``grader`` runs the relevance and
    groundedness checks. They are separate because the graders fire on every
    query and on every golden question during an eval run, so pointing them at a
    cheaper model is the single biggest lever on what this project costs to
    evaluate. When the caller supplies a provider from the UI, both use it.
    """

    generator: BaseChatModel
    grader: BaseChatModel
    provider: str
    generator_model: str
    grader_model: str


def build_bundle(
    settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMBundle:
    provider = resolve_provider(provider or settings.provider)
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise LLMConfigError(f"Unknown provider {provider!r}")

    configured = resolve_provider(settings.provider)
    gen_model = model or (
        settings.model if provider == configured else spec.default_model
    )
    # Only reuse the configured grader model when we are on its provider.
    if provider == resolve_provider(settings.grader_provider):
        grader_model = settings.grader_model
    else:
        # Prefer the cheapest listed model for the chosen provider.
        grader_model = spec.models[-1] if spec.models else gen_model

    generator = build_chat_model(
        provider, gen_model, api_key=api_key, max_tokens=settings.max_tokens
    )
    grader = build_chat_model(
        provider, grader_model, api_key=api_key, max_tokens=1024
    )
    return LLMBundle(
        generator=generator,
        grader=grader,
        provider=provider,
        generator_model=gen_model,
        grader_model=grader_model,
    )

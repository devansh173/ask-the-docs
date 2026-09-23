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

**Automatic fallback across sibling models.** Free-tier quotas are metered per
model, discovered the hard way while evaluating this project against Gemini's
free tier: gemini-2.5-flash-lite is capped at 20 requests/day, and even a
current model's per-*minute* cap (15 RPM at one point) can be blown through by
a judge library's default concurrency. ``build_bundle`` therefore builds each
role (generator, grader) as a *chain* - the configured model first, then up to
``settings.max_fallbacks`` other models from the same provider - and
``call_with_fallback`` walks the chain on a quota refusal. A provider whose
account is simply out of credit fails identically on every model in the chain,
so this only ever helps with the case it targets; it does not paper over a
missing key or a genuine service outage.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, TypeVar

from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)

T = TypeVar("T")


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
        # Ordered flash-lite-first: probing this account's actual quotas found
        # gemini-3.1-flash-lite capped at 500 requests/day against
        # gemini-3.6-flash's 20/day - the opposite of "newer or non-lite means
        # more quota". That is an observed pattern on one account on one day,
        # not a documented guarantee, and it visibly changes: gemini-2.5-flash-lite
        # measured at 20/day earlier the same day this comment was written and
        # was back to answering within the hour. Static ordering is a mild
        # preference at best - build_bundle's fallback chain is what actually
        # copes with whichever models happen to be exhausted right now.
        models=(
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-2.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-3.6-flash",
            "gemini-3.1-pro-preview",
            "gemini-2.5-pro",
        ),
        default_model="gemini-3.5-flash-lite",
        package="langchain_google_genai",
        notes=(
            "Free-tier quota is metered per model per day, and how much varies "
            "by model in ways that are not documented and shift over time - "
            "the app falls back across sibling models automatically on a "
            "quota refusal (LLM_MAX_FALLBACKS) rather than relying on a fixed "
            "'good model' list."
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


class AllFallbacksExhausted(RuntimeError):
    """Every model in a fallback chain refused on quota."""


# Provider-agnostic substrings for "you are out of quota" - as opposed to a
# transient network error or a bad request, neither of which a different model
# on the same account would fix. Matched case-insensitively against the
# exception's str(), since each SDK raises its own exception class.
_QUOTA_MARKERS = (
    "resource_exhausted",
    "exceeded your current quota",
    "insufficient_quota",
    "quota exceeded",
    "rate_limit_exceeded",
    "billing",
    "credit balance is too low",
)


def is_quota_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def call_with_fallback(
    chain: list[tuple[str, BaseChatModel]],
    fn: Callable[[BaseChatModel], T],
) -> tuple[T, str]:
    """Run ``fn`` against each model in ``chain`` in order.

    Returns ``(result, model_name)`` from the first model that does not raise a
    quota-shaped error. A non-quota error propagates immediately rather than
    burning through the rest of the chain - a bad prompt or a malformed
    response is not fixed by trying a different model, and hiding it behind
    three retries would only slow down the real failure.
    """
    if not chain:
        raise ValueError("call_with_fallback needs at least one (name, model) pair")

    last_exc: BaseException | None = None
    for name, model in chain:
        try:
            return fn(model), name
        except Exception as exc:
            if not is_quota_error(exc):
                raise
            log.warning("%s refused on quota, falling back", name)
            last_exc = exc

    raise AllFallbacksExhausted(
        f"every model in the chain refused on quota: {[n for n, _ in chain]}"
    ) from last_exc


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

    ``generator_chain`` / ``grader_chain`` carry the configured model plus its
    fallbacks, as ``(model_name, model)`` pairs with the configured one first.
    They default to empty - ``generator_models()`` / ``grader_models()`` fall
    back to a single-item chain built from ``generator``/``grader`` when empty,
    so code built against a bundle that predates fallbacks (tests, mainly)
    keeps working unchanged.
    """

    generator: BaseChatModel
    grader: BaseChatModel
    provider: str
    generator_model: str
    grader_model: str
    generator_chain: list[tuple[str, BaseChatModel]] = field(default_factory=list)
    grader_chain: list[tuple[str, BaseChatModel]] = field(default_factory=list)

    def generator_models(self) -> list[tuple[str, BaseChatModel]]:
        return self.generator_chain or [(self.generator_model, self.generator)]

    def grader_models(self) -> list[tuple[str, BaseChatModel]]:
        return self.grader_chain or [(self.grader_model, self.grader)]


def _fallback_chain(
    provider: str,
    spec: ProviderSpec,
    primary_model: str,
    primary: BaseChatModel,
    *,
    api_key: str | None,
    max_tokens: int,
    max_fallbacks: int,
) -> list[tuple[str, BaseChatModel]]:
    """The primary model plus up to ``max_fallbacks`` siblings from ``spec``.

    Siblings are taken in the provider's declared order (roughly
    best-to-cheapest) and skip the primary itself. Each is a live client
    object, but constructing one does not make a network call for any provider
    here, so building the whole chain eagerly costs nothing until it is used.
    A sibling that fails to construct (missing optional package, for instance)
    is skipped rather than aborting the whole chain.
    """
    chain: list[tuple[str, BaseChatModel]] = [(primary_model, primary)]
    if max_fallbacks <= 0:
        return chain

    for candidate in spec.models:
        if len(chain) - 1 >= max_fallbacks:
            break
        if candidate == primary_model:
            continue
        try:
            model_obj = build_chat_model(
                provider, candidate, api_key=api_key, max_tokens=max_tokens
            )
        except LLMConfigError as exc:
            log.debug("skipping fallback candidate %s: %s", candidate, exc)
            continue
        chain.append((candidate, model_obj))
    return chain


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

    max_fallbacks = getattr(settings, "max_fallbacks", 2)
    generator_chain = _fallback_chain(
        provider, spec, gen_model, generator,
        api_key=api_key, max_tokens=settings.max_tokens, max_fallbacks=max_fallbacks,
    )
    grader_chain = _fallback_chain(
        provider, spec, grader_model, grader,
        api_key=api_key, max_tokens=1024, max_fallbacks=max_fallbacks,
    )

    return LLMBundle(
        generator=generator,
        grader=grader,
        provider=provider,
        generator_model=gen_model,
        grader_model=grader_model,
        generator_chain=generator_chain,
        grader_chain=grader_chain,
    )

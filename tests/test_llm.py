"""Provider layer tests.

These cover the parts that must hold without any vendor being reachable: the
catalog the UI renders, the error raised when a key is missing, and - most
importantly - that a supplied API key never leaks into a log line.
"""

from __future__ import annotations

import pytest

from askthedocs.api.schemas import AskRequest
from askthedocs.llm import (
    DEFAULT_PROVIDER,
    PROVIDER_ALIASES,
    PROVIDERS,
    AllFallbacksExhausted,
    LLMBundle,
    LLMConfigError,
    build_chat_model,
    call_with_fallback,
    is_quota_error,
    provider_catalog,
    resolve_provider,
)


class _FakeModel:
    """Stands in for a BaseChatModel. Raises on the first N calls, then answers."""

    def __init__(self, fail_times: int = 0, exc: Exception | None = None):
        self.fail_times = fail_times
        self.exc = exc or RuntimeError(
            "429 RESOURCE_EXHAUSTED: You exceeded your current quota"
        )
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return "ok"


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def test_every_provider_is_internally_consistent():
    for provider_id, spec in PROVIDERS.items():
        assert spec.id == provider_id
        assert spec.models, f"{provider_id} lists no models"
        assert spec.default_model in spec.models, (
            f"{provider_id} default {spec.default_model} is not in its model list"
        )
        if spec.needs_key:
            assert spec.env_key and spec.key_field, (
                f"{provider_id} needs a key but declares no env var or field"
            )


def test_default_provider_exists():
    assert DEFAULT_PROVIDER in PROVIDERS


def test_ollama_needs_no_key():
    assert PROVIDERS["ollama"].needs_key is False


def test_catalog_exposes_what_the_ui_needs(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    catalog = provider_catalog()
    assert catalog
    for entry in catalog:
        assert set(entry) >= {
            "id", "label", "models", "default_model", "needs_key", "key_in_env",
        }
    anthropic = next(e for e in catalog if e["id"] == "anthropic")
    assert anthropic["key_in_env"] is False


def test_catalog_reports_a_key_present_in_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    anthropic = next(e for e in provider_catalog() if e["id"] == "anthropic")
    assert anthropic["key_in_env"] is True


# --------------------------------------------------------------------------- #
# Provider aliases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "typed,expected",
    [
        ("gemini", "google_genai"),
        ("Gemini", "google_genai"),
        ("google", "google_genai"),
        ("google-genai", "google_genai"),
        ("google_genai", "google_genai"),
        ("claude", "anthropic"),
        ("anthropic", "anthropic"),
        ("gpt", "openai"),
        ("openai", "openai"),
        ("local", "ollama"),
    ],
)
def test_vendor_names_resolve_to_registry_ids(typed, expected):
    """LLM_PROVIDER=gemini must work.

    The registry is keyed by LangChain package name (google_genai), which is not
    what anyone types into a .env file. Without aliasing, a perfectly reasonable
    LLM_PROVIDER=gemini fails with "Unknown provider".
    """
    assert resolve_provider(typed) == expected


def test_no_alias_shadows_a_real_provider_id():
    assert not (set(PROVIDER_ALIASES) & set(PROVIDERS))


def test_every_alias_points_at_a_real_provider():
    for alias, target in PROVIDER_ALIASES.items():
        assert target in PROVIDERS, f"alias {alias} points at unknown {target}"


def test_empty_provider_falls_back_to_the_default():
    assert resolve_provider(None) == DEFAULT_PROVIDER
    assert resolve_provider("") == DEFAULT_PROVIDER


def test_unrecognised_name_is_returned_unchanged_for_the_error_message():
    assert resolve_provider("not-a-vendor") == "not-a-vendor"


def test_gemini_flash_lite_is_selectable():
    """The cheapest Gemini model, and a sensible grader - it must be listed."""
    assert "gemini-2.5-flash-lite" in PROVIDERS["google_genai"].models


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_unknown_provider_is_rejected_by_name():
    with pytest.raises(LLMConfigError, match="Unknown provider"):
        build_chat_model("not-a-provider")


def test_missing_key_produces_an_actionable_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError) as exc:
        build_chat_model("anthropic", "claude-opus-5")
    message = str(exc.value)
    assert "ANTHROPIC_API_KEY" in message, "the error should name the variable to set"
    assert "sidebar" in message, "the error should mention the UI alternative"


def test_a_request_supplied_key_satisfies_the_check(monkeypatch):
    """A key from the UI must work when the environment has none."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = build_chat_model("anthropic", "claude-opus-5", api_key="sk-ant-supplied")
    assert model is not None  # constructed; no network call is made here


def test_blank_key_is_treated_as_absent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        build_chat_model("anthropic", "claude-opus-5", api_key="   ")


# --------------------------------------------------------------------------- #
# Key handling - the property that matters most
# --------------------------------------------------------------------------- #
def test_redacted_request_never_contains_the_key():
    request = AskRequest(
        question="how does RRF fusion work?",
        provider="anthropic",
        api_key="sk-ant-super-secret-value",
    )
    redacted = request.redacted()

    assert "sk-ant-super-secret-value" not in str(redacted)
    assert redacted["api_key"] == "<supplied>"
    assert redacted["question"] == "how does RRF fusion work?"


def test_redacted_request_distinguishes_absent_from_supplied():
    request = AskRequest(question="a question with enough characters")
    assert request.redacted()["api_key"] is None


def test_request_defaults_to_the_agentic_pipeline():
    assert AskRequest(question="a question with enough characters").config == "agentic"


def test_short_questions_are_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AskRequest(question="hi")


# --------------------------------------------------------------------------- #
# Automatic fallback across sibling models
#
# Built after discovering, while evaluating this project against Gemini's
# free tier, that quota is metered per model per day and can be exhausted by
# ordinary testing well before a real workload runs. These tests use fake
# models rather than live ones - the behaviour under test is the chain-walking
# logic, not any particular provider's quota, and it needs to hold regardless
# of which vendor is configured.
# --------------------------------------------------------------------------- #
def test_quota_errors_are_recognised_across_provider_wordings():
    for message in (
        "429 RESOURCE_EXHAUSTED",
        "You exceeded your current quota, please check your plan",
        "insufficient_quota",
        "rate_limit_exceeded",
        "Your credit balance is too low",
    ):
        assert is_quota_error(RuntimeError(message)), message


def test_ordinary_errors_are_not_mistaken_for_quota():
    """A bad prompt or a malformed response is not fixed by a different model."""
    for message in ("connection reset by peer", "invalid request: bad schema"):
        assert not is_quota_error(RuntimeError(message)), message


def test_fallback_advances_past_a_quota_refusal():
    primary = _FakeModel(fail_times=1)
    backup = _FakeModel(fail_times=0)
    chain = [("primary", primary), ("backup", backup)]

    result, used = call_with_fallback(chain, lambda m: m())

    assert result == "ok"
    assert used == "backup"
    assert primary.calls == 1, "the failed model should be tried exactly once"


def test_fallback_prefers_the_primary_when_it_works():
    primary = _FakeModel(fail_times=0)
    backup = _FakeModel(fail_times=0)
    chain = [("primary", primary), ("backup", backup)]

    result, used = call_with_fallback(chain, lambda m: m())

    assert used == "primary"
    assert backup.calls == 0, "a working primary must not touch the fallback"


def test_a_non_quota_error_is_not_retried_on_the_next_model():
    """Retrying a bad request on a different model wastes a call and hides
    the real bug - only quota-shaped failures should advance the chain."""
    primary = _FakeModel(fail_times=1, exc=ValueError("malformed input"))
    backup = _FakeModel(fail_times=0)
    chain = [("primary", primary), ("backup", backup)]

    with pytest.raises(ValueError, match="malformed input"):
        call_with_fallback(chain, lambda m: m())
    assert backup.calls == 0


def test_every_model_exhausted_raises_a_named_error():
    chain = [("a", _FakeModel(fail_times=99)), ("b", _FakeModel(fail_times=99))]
    with pytest.raises(AllFallbacksExhausted):
        call_with_fallback(chain, lambda m: m())


def test_call_with_fallback_needs_at_least_one_model():
    with pytest.raises(ValueError):
        call_with_fallback([], lambda m: m())


def test_bundle_without_a_chain_falls_back_to_the_single_configured_model():
    """Bundles built before fallback chains existed (most test fixtures) must
    keep working unchanged: an empty chain means "just the one model"."""
    bundle = LLMBundle(
        generator="gen-model", grader="grade-model",
        provider="anthropic", generator_model="gen-name", grader_model="grade-name",
    )
    assert bundle.generator_models() == [("gen-name", "gen-model")]
    assert bundle.grader_models() == [("grade-name", "grade-model")]


def test_bundle_with_an_explicit_chain_uses_it_instead():
    bundle = LLMBundle(
        generator="primary", grader="g", provider="anthropic",
        generator_model="primary-name", grader_model="g-name",
        generator_chain=[("primary-name", "primary"), ("backup-name", "backup")],
    )
    assert bundle.generator_models() == [("primary-name", "primary"), ("backup-name", "backup")]


def test_build_bundle_populates_a_fallback_chain(monkeypatch):
    """The real construction path: a provider with several models must
    produce a chain longer than one, bounded by max_fallbacks."""
    from askthedocs.config import settings as base_settings
    from askthedocs.llm import build_bundle

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = base_settings.variant(
        provider="anthropic", model="claude-opus-5",
        grader_provider="anthropic", grader_model="claude-sonnet-5",
        max_fallbacks=2,
    )
    bundle = build_bundle(cfg)

    assert bundle.generator_chain[0][0] == "claude-opus-5"
    assert 2 <= len(bundle.generator_chain) <= 3  # primary + up to 2 fallbacks
    names = [n for n, _ in bundle.generator_chain]
    assert len(names) == len(set(names)), "no model should appear twice in a chain"


def test_max_fallbacks_zero_disables_the_chain(monkeypatch):
    from askthedocs.config import settings as base_settings
    from askthedocs.llm import build_bundle

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = base_settings.variant(provider="anthropic", max_fallbacks=0)
    bundle = build_bundle(cfg)

    assert len(bundle.generator_chain) == 1
    assert len(bundle.grader_chain) == 1

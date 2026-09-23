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
    LLMConfigError,
    build_chat_model,
    provider_catalog,
    resolve_provider,
)


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

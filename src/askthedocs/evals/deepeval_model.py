"""Bridge DeepEval's judge interface onto this project's LLM layer.

DeepEval reaches for OpenAI by default. That would mean the CI quality gate
depended on a vendor the application itself never calls, and on a second API
key. Wrapping the configured grader model instead keeps one provider choice for
the whole project - whatever the app uses, the gate judges with.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage

from ..config import Settings, settings as default_settings
from ..llm import build_chat_model

log = logging.getLogger(__name__)


class LangChainJudge(DeepEvalBaseLLM):
    """A DeepEval judge backed by any provider the app supports."""

    def __init__(
        self,
        settings: Settings | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        settings = settings or default_settings
        self._provider = provider or settings.grader_provider
        self._model_name = model or settings.grader_model
        self._chat = build_chat_model(
            self._provider, self._model_name, temperature=0.0, max_tokens=2048
        )
        super().__init__(model_name=self._model_name)

    # -- required surface ---------------------------------------------------
    def load_model(self, *args: Any, **kwargs: Any):
        return self._chat

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"{self._provider}:{self._model_name}"

    @staticmethod
    def _text(response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        return str(content)

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return self._text(self._chat.invoke([HumanMessage(content=prompt)]))

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        response = await self._chat.ainvoke([HumanMessage(content=prompt)])
        return self._text(response)

    # -- structured output --------------------------------------------------
    # DeepEval's metrics ask for a pydantic schema back. Native structured
    # output is used where the provider supports it; the JSON fallback covers
    # local models that do not.
    def generate_with_schema(self, prompt: str, *args: Any, schema=None, **kwargs: Any):
        if schema is None:
            return self.generate(prompt)
        try:
            return self._chat.with_structured_output(schema).invoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as exc:
            log.debug("structured judge output failed (%s); parsing json", exc)
            return self._parse(self.generate(prompt), schema)

    async def a_generate_with_schema(
        self, prompt: str, *args: Any, schema=None, **kwargs: Any
    ):
        if schema is None:
            return await self.a_generate(prompt)
        try:
            return await self._chat.with_structured_output(schema).ainvoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as exc:
            log.debug("structured judge output failed (%s); parsing json", exc)
            return self._parse(await self.a_generate(prompt), schema)

    @staticmethod
    def _parse(text: str, schema):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"judge returned no JSON object: {text[:200]!r}")
        return schema.model_validate(json.loads(text[start : end + 1]))

    # -- capability flags ---------------------------------------------------
    def supports_json_mode(self) -> bool:
        return True

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_temperature(self) -> bool:
        return True

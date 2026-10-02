import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class StructuredLLMOutput(BaseModel):
    answer: str = Field(min_length=1)
    supported_by_context: bool


@dataclass(frozen=True)
class LLMResponse:
    content: str
    supported_by_context: bool
    usage: dict[str, Any] = field(default_factory=dict)


def parse_structured_llm_output(raw_content: str) -> StructuredLLMOutput:
    content = raw_content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(content)
        return StructuredLLMOutput.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError("LLM returned invalid structured output") from exc


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate an answer from structured system and user prompts."""

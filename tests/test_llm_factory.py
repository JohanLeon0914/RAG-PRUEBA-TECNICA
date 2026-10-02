import pytest

from app.config.settings import Settings
from app.llm.factory import LLMFactory
from app.llm.gemini import GeminiLLMProvider
from app.llm.groq import GroqLLMProvider


def test_llm_factory_creates_groq_provider() -> None:
    settings = Settings(
        LLM_PROVIDER="groq",
        GROQ_API_KEY="test-key",
        LLM_MODEL="openai/gpt-oss-120b",
    )

    provider = LLMFactory.create(settings)

    assert isinstance(provider, GroqLLMProvider)
    assert provider.model == "openai/gpt-oss-120b"


def test_llm_factory_creates_gemini_provider() -> None:
    settings = Settings(
        LLM_PROVIDER="gemini",
        GEMINI_API_KEY="test-key",
        LLM_MODEL="gemini-1.5-flash",
    )

    provider = LLMFactory.create(settings)

    assert isinstance(provider, GeminiLLMProvider)
    assert provider.model == "gemini-1.5-flash"


def test_llm_factory_rejects_missing_groq_key() -> None:
    settings = Settings(LLM_PROVIDER="groq", GROQ_API_KEY=None)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        LLMFactory.create(settings)


def test_llm_factory_rejects_invalid_provider() -> None:
    settings = Settings.model_construct(llm_provider="unknown")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMFactory.create(settings)

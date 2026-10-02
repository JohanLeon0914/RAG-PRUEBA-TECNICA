from app.config.settings import Settings
from app.llm.base import LLMProvider
from app.llm.gemini import GeminiLLMProvider
from app.llm.groq import GroqLLMProvider


class LLMFactory:
    @staticmethod
    def create(settings: Settings) -> LLMProvider:
        if settings.llm_provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
            return GroqLLMProvider(
                api_key=settings.groq_api_key,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        if settings.llm_provider == "gemini":
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
            return GeminiLLMProvider(
                api_key=settings.gemini_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

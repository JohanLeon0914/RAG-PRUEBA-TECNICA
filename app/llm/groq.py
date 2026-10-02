from app.llm.base import LLMProvider, LLMResponse, parse_structured_llm_output


class GroqLLMProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int = 30,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            response = self._load_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError("Groq LLM request failed") from exc

        content = response.choices[0].message.content or ""
        structured = parse_structured_llm_output(content)
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        return LLMResponse(
            content=structured.answer,
            supported_by_context=structured.supported_by_context,
            usage=usage,
        )

    def _load_client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)
        return self._client

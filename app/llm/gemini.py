from app.llm.base import LLMProvider, LLMResponse, parse_structured_llm_output


class GeminiLLMProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = None

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            response = self._load_model().generate_content(
                [system_prompt, user_prompt],
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
        except Exception as exc:
            raise RuntimeError("Gemini LLM request failed") from exc

        structured = parse_structured_llm_output(response.text or "")
        return LLMResponse(
            content=structured.answer,
            supported_by_context=structured.supported_by_context,
            usage={},
        )

    def _load_model(self):
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model)
        return self._model

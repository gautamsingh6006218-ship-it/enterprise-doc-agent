import os

import ollama

from agent.llm.base import BaseLLMProvider, LLMResult

_DEFAULT_MODEL = "llama3.2"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseLLMProvider):
    """
    LLM provider backed by a locally running Ollama instance.

    Environment variables:
      OLLAMA_LLM_MODEL   — model name to use (default: llama3.2)
      OLLAMA_BASE_URL    — Ollama server URL (default: http://localhost:11434)
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self._model = model or os.getenv("OLLAMA_LLM_MODEL", _DEFAULT_MODEL)
        self._client = ollama.Client(
            host=base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        )

    def generate(self, system_prompt: str, user_message: str) -> LLMResult:
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            message = response.message
            usage = response.usage if hasattr(response, "usage") and response.usage else None
            return LLMResult(
                success=True,
                answer=message.content or "",
                model=self._model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as exc:
            return LLMResult(success=False, error=str(exc))

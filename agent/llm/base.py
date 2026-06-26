from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResult:
    success: bool
    answer: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> LLMResult:
        ...

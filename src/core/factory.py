import os
import instructor
from groq import Groq
from google import genai
from pydantic import BaseModel
from typing import Type, TypeVar, Iterable
from loguru import logger
from .metrics import TokenMetrics

T = TypeVar("T", bound=BaseModel)

class LLMFactory:
    def __init__(self, provider: str = None):
        self.provider = provider or os.getenv("DEFAULT_PROVIDER", "groq")
        self.model_name = "llama-3.3-70b-versatile"
        self._setup_client()

    def _setup_client(self):
        if self.provider == "groq":
            self.client = instructor.from_groq(
                Groq(api_key=os.getenv("GROQ_API_KEY")),
                model="llama-3.3-70b-versatile"
            )
        elif self.provider == "gemini":
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def get_structured(self, response_model: Type[T], user_prompt: str) -> tuple[T, TokenMetrics]:
        # We use create_with_completion to get both parsed object and raw completion.
        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=response_model,
            messages=[{"role": "user", "content": user_prompt}],
        )

        usage = completion.usage
        metrics = TokenMetrics(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            provider=self.provider,
        )

        return response, metrics
    def get_stream(self, prompt: str) -> Iterable[str]:
        logger.info(f"Initiating stream via {self.provider}")
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

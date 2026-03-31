import os
import instructor
from groq import Groq
from google import genai
from pydantic import BaseModel
from typing import Type, TypeVar, Iterable
from loguru import logger

T = TypeVar("T", bound=BaseModel)

class LLMFactory:
    def __init__(self, provider: str = None):
        self.provider = provider or os.getenv("DEFAULT_PROVIDER", "groq")
        self._setup_client()

    def _setup_client(self):
        if self.provider == "groq":
            self.client = instructor.from_groq(
                Groq(api_key=os.getenv("GROQ_API_KEY")),
                model="llama-3.3-70b-versatile"
            )
        elif self.provider == "gemini":
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def get_structured(self, response_model: Type[T], prompt: str) -> T:
        logger.info(f"Extracting structured data via {self.provider}")
        return self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
        )

    def get_stream(self, prompt: str) -> Iterable[str]:
        logger.info(f"Initiating stream via {self.provider}")
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

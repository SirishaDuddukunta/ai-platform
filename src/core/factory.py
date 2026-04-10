import instructor
from groq import Groq
from src.core.metrics import TokenMetrics

class LLMFactory:
    def __init__(self, provider="groq"):
        self.provider = provider
        self.client = instructor.from_groq(Groq())
        self.model_name = "llama-3.3-70b-versatile"

    # Day 1-4: Structured Extraction
    def get_structured(self, response_model, user_prompt):
        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=response_model,
            messages=[
                {"role": "system", "content": "You are a resilient Data Extraction Agent."},
                {"role": "user", "content": user_prompt}
            ]
        )
        metrics = TokenMetrics(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
            provider=self.provider
        )
        return response, metrics

    # Day 5-6: Streaming Chat logic
    def stream_chat(self, messages):
        # We use the raw groq client here for streaming (instructor support varies)
        raw_client = Groq() 
        stream = raw_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        return stream
import instructor
from groq import Groq
from src.core.metrics import TokenMetrics

class LLMFactory:
    def __init__(self, provider="groq"):
        self.provider = provider
        if provider == "groq":
            self.client = instructor.from_groq(Groq())
            self.model_name = "llama-3.3-70b-versatile"
        else:
            # Add Gemini logic here if needed
            pass

    def get_structured(self, response_model, user_prompt):
        # DAY 4 UPGRADE: Added max_retries
        # This catches validation errors and asks the LLM to fix them!
        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=response_model,
            messages=[{"role": "user", "content": user_prompt}],
            max_retries=3  
        )
        
        # Day 3 Metrics Logic
        usage = completion.usage
        metrics = TokenMetrics(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            provider=self.provider
        )
        
        return response, metrics
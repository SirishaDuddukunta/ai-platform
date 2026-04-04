import instructor
from groq import Groq
from src.core.metrics import TokenMetrics

class LLMFactory:
    def __init__(self, provider="groq"):
        self.provider = provider
        # Initialize the instructor client
        self.client = instructor.from_groq(Groq())
        self.model_name = "llama-3.3-70b-versatile"

    def get_structured(self, response_model, user_prompt):
        system_instruction = """
        You are a resilient Data Extraction Agent. 
        If data is 'TBD' or 0, explain your logic in the 'reasoning' field.
        
        EXAMPLE:
        Input: "Qty 0, Price TBD"
        Reasoning: "Quantity was 0; adjusted to 1 for pre-order. Price was TBD; defaulted to 0.0."
        """

        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            max_retries=3 
        )
        
        # Calculate metrics for the Day 3/4 report
        metrics = TokenMetrics(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
            provider=self.provider
        )
        
        return response, metrics
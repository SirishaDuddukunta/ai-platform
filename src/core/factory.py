import instructor
from groq import Groq
from src.core.metrics import TokenMetrics

class LLMFactory:
    def __init__(self, provider="groq"):
        self.provider = provider
        self.client = instructor.from_groq(Groq())
        self.raw_client = Groq() # Needed for tool calling/streaming
        self.model_name = "llama-3.3-70b-versatile"

    # ------------------------------------------
    # DAY 1-7: STRUCTURED & STREAMING LOGIC (KEEPING FOR HISTORY)
    # ------------------------------------------
    def get_structured(self, response_model, user_prompt):
        # [Old Day 7 Logic remains active here...]
        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=response_model,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response, TokenMetrics(total_tokens=completion.usage.total_tokens)

    # ------------------------------------------
    # DAY 8: TOOL-ENABLED CHAT
    # ------------------------------------------
    # DEFINITION:
    # What: A method that sends a prompt + a list of available tools to the LLM.
    # Why: We need to see if the LLM returns 'content' (talking) or 'tool_calls' (acting).
    
    def chat_with_tools(self, messages, tools):
        """
        Sends tools to the LLM and returns the raw response to check for tool_calls.
        """
        response = self.raw_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools, # This is the Day 8 addition
            tool_choice="auto" 
        )
        return response.choices[0].message
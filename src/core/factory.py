import instructor
from groq import Groq
from groq import BadRequestError
from src.core.metrics import TokenMetrics
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

__all__ = ["LLMFactory", "SentenceTransformer"]


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
        try:
            response = self.raw_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools, # This is the Day 8 addition
                tool_choice="auto",
                temperature=0
            )
        except BadRequestError as exc:
            # Some models occasionally emit malformed function-call text.
            # Retry once with an explicit instruction for strict tool-calling format.
            if "tool_use_failed" not in str(exc):
                raise

            retry_messages = [
                {
                    "role": "system",
                    "content": (
                        "When you need a tool, respond ONLY with a valid tool call. "
                        "Never use XML-style tags like <function=...>. "
                        "Arguments must be valid JSON that matches the tool schema."
                    ),
                },
                *messages,
            ]

            response = self.raw_client.chat.completions.create(
                model=self.model_name,
                messages=retry_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0
            )
        return response.choices[0].message
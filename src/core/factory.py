import instructor
import chromadb 
import time
from groq import Groq
from src.core.metrics import TokenMetrics
from sentence_transformers import SentenceTransformer

__all__ = ["LLMFactory", "SentenceTransformer", "TOOLS_SCHEMA"]

class LLMFactory:
    def __init__(self, provider="groq"):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="engineer_docs")
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.provider = provider
        self.client = instructor.from_groq(Groq())
        self.raw_client = Groq()
        self.model_name = "llama-3.3-70b-versatile"

    def add_to_library(self, text: str, doc_id: str, metadata: dict = None):
        vector = self.get_embedding(text)
        self.collection.add(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[metadata or {}],
            documents=[text]
        )
        return f"Successfully indexed doc: {doc_id}"

    def get_structured(self, response_model, user_prompt):
        response, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=response_model,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response, TokenMetrics(total_tokens=completion.usage.total_tokens)

    def chat_with_tools(self, messages, tools):
        """
        Refined method: Force System Prompt to prevent XML tag hallucination
        and use tool_choice='auto' for deterministic tool routing.
        """
        system_instruction = {
            "role": "system",
            "content": "You are a helpful assistant. You have access to tools. If you use a tool, you must output a valid JSON tool call. Do not use custom XML tags like <function>."
        }

        # Combine system instruction with the provided message history
        # We assume messages is a list of dicts. If it's a list, we prepend.
        full_messages = [system_instruction] + messages

        response = self.raw_client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            tools=tools,
            tool_choice="auto" 
        )
        print(f"DEBUG: LLM Response type: {type(response)}")
        return response
    
    def get_embedding(self, text: str):
        start = time.time()
        vector = self.embed_model.encode(text).tolist()
        latency = (time.time() - start) * 1000
        print(f"📊 Embedding Latency: {latency:.2f}ms")
        return vector

# Global Schema Definition
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Searches the knowledge base for information on a specific topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_string": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query_string"]
            }
        }
    }
]
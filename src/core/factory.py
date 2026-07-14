import instructor
import chromadb
import sys
import time
from groq import Groq
from src.core.metrics import TokenMetrics
from sentence_transformers import SentenceTransformer

__all__ = ["LLMFactory", "SentenceTransformer"]

class LLMFactory:
    def __init__(self, provider="groq", embed_model: SentenceTransformer = None):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="engineer_docs")
        # Allow callers to inject a shared model instance so it isn't loaded into memory twice.
        self.embed_model = embed_model or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

    def search_library(self, query: str, n_results: int = 3):
        # Turn the search text into the same kind of vector we used when we
        # stored documents with add_to_library(). Two pieces of text can only
        # be compared once they're both vectors in the same "embedding space".
        query_vector = self.get_embedding(query)

        # Ask Chroma for the n_results documents whose vectors are closest
        # (most similar in meaning) to our query vector.
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )

        # Chroma returns parallel lists (one per query) nested inside dicts,
        # e.g. results["documents"] == [["doc text 1", "doc text 2"]].
        # We only ever send one query at a time, so we grab index [0].
        matched_docs = results.get("documents", [[]])[0]
        matched_ids = results.get("ids", [[]])[0]

        # Zip pairs each doc up with its id so the caller gets both, not just
        # raw text with no way to know which stored document it came from.
        return [{"id": doc_id, "text": doc_text} for doc_id, doc_text in zip(matched_ids, matched_docs)]

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
        # Debug logs must go to stderr, not stdout: this class is also used
        # inside the MCP server (src/core/mcp_server.py), which uses stdout
        # as its actual protocol channel to talk to the client. A stray
        # print() to stdout there would corrupt every tool call.
        print(f"DEBUG: LLM Response type: {type(response)}", file=sys.stderr)
        return response

    def get_embedding(self, text: str):
        start = time.time()
        vector = self.embed_model.encode(text).tolist()
        latency = (time.time() - start) * 1000
        print(f"📊 Embedding Latency: {latency:.2f}ms", file=sys.stderr)
        return vector
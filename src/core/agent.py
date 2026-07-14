import os
import json
from groq import Groq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- 1. VECTOR ENGINE SETUP ---
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
embedder = SentenceTransformer('all-MiniLM-L6-v2') # Must match the model used in ingest.py

def semantic_search(query: str):
    """
    Retrieves project context using semantic similarity with the Query API.
    """
    # 1. Convert query to vector space
    query_vector = embedder.encode(query).tolist()
    
    # 2. Use 'query_points' instead of 'search'
    # This is the modern Qdrant Query API method
    search_result = client.query_points(
        collection_name="project_docs",
        query=query_vector,
        limit=2
    )
    
    # 3. Extract the payload text
    # Note: query_points returns a different structure than search
    return "\n".join([point.payload["text"] for point in search_result.points])

# --- 2. TOOL REGISTRY ---
TOOLS = {
    "semantic_search": semantic_search,
}

# --- 3. THE AGENT (The Brain) ---
class NativeAgent:
    def __init__(self, system_prompt):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # This prompt now enforces a strict Data Contract
        self.system_prompt = f"""{system_prompt}
        CRITICAL: You are an agent. 
        1. If calling a tool: {{"thought": "...", "action": "semantic_search", "action_input": {{"query": "..."}}}}
        2. If providing the final answer, YOU MUST RETURN THIS JSON OBJECT:
           {{
             "thought": "...",
             "answer": {{
               "project_name": "Alpha",
               "status": "in_progress",
               "percentage_complete": 75,
               "expected_completion_date": "2026-12-31"
             }}
           }}
        Respond ONLY in valid JSON.
        """
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _call_llm(self):
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=self.history,
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def run(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        
        for i in range(5):
            response = self._call_llm()
            
            if "answer" in response:
                ans = response['answer']
                # If the answer is a string but looks like JSON, parse it
                if isinstance(ans, str):
                    try:
                        return json.loads(ans)
                    except json.JSONDecodeError:
                        return {"status": "unknown", "message": ans} # Fallback
                return ans # It's already a dict
            
            if "action" in response:
                tool_name = response["action"]
                tool_args = response.get("action_input", {})

                # Execute tool via registry. Self-correction: if the tool is unknown
                # or raises, feed the error back as an observation instead of crashing.
                try:
                    if tool_name not in TOOLS:
                        tool_result = f"Error: Tool '{tool_name}' not found."
                    else:
                        tool_result = TOOLS[tool_name](**tool_args)
                except Exception as e:
                    tool_result = f"TOOL_EXECUTION_ERROR: {str(e)}"

                self.history.append({"role": "assistant", "content": json.dumps(response)})
                self.history.append({"role": "user", "content": f"Observation: {tool_result}"})

        # Loop exhausted without the model returning a final answer.
        return {"status": "unknown", "message": "Agent did not converge within the step limit."}
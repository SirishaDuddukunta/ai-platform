import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. DEFINE TOOLS GLOBALLY
def get_project_status(project_id: str):
    mock_db = {"Alpha": "Design", "Beta": "Development"}
    return mock_db.get(project_id, "Unknown Project")

# This dictionary MUST contain the exact tool names the LLM uses
TOOLS = {
    "project_status": get_project_status,  # Match the name the LLM hallucinated
    "get_project_status": get_project_status # Keep both for safety
}

# 2. THE AGENT CLASS
class NativeAgent:
    def __init__(self, system_prompt):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.system_prompt = f"""{system_prompt}
        You are a reasoning agent. Respond ONLY in valid JSON.
        If you need to call a tool, use this format:
        {{"thought": "...", "action": "tool_name", "action_input": {{"project_id": "Alpha"}}}}
        If you have the final answer, use this format:
        {{"thought": "...", "answer": "..."}}
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
            
            # --- FINAL ANSWER ---
            if "answer" in response:
                print(f"Agent Final Answer: {response['answer']}")
                return response['answer']
            
            # --- ACTION DISPATCH ---
            if "action" in response:
                tool_name = response["action"]
                tool_args = response.get("action_input", {})
                
                print(f"Agent Thought: {response.get('thought')}")
                print(f"Agent Action: {tool_name} with {tool_args}")
                
                # EXECUTION: Use the global TOOLS dict
                func = TOOLS.get(tool_name)
                if func:
                    tool_result = func(**tool_args)
                else:
                    tool_result = f"Error: Tool {tool_name} not found."
                
                print(f"Agent Observed: {tool_result}")
                
                self.history.append({"role": "assistant", "content": json.dumps(response)})
                self.history.append({"role": "user", "content": f"Observation: {tool_result}"})

# 3. ENTRY POINT (Only runs if this file is executed directly)
if __name__ == "__main__":
    agent = NativeAgent("You are a helpful CRM assistant.")
    final_result = agent.run("What is the status of Alpha project?")
    print(f"Result: {final_result}")
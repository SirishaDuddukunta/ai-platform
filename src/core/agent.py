import os                       # Allows Python to talk to your computer (e.g., read API keys)
import json                     # Allows Python to convert data into "text" the AI can understand
from groq import Groq           # Imports the "Brain" (the LLM client)
from dotenv import load_dotenv  # Loads your secret keys from the .env file safely

# Load the keys from .env into the system memory so the code can use them
load_dotenv()
# --- 1. THE TOOLS (The Drawer) ---
def get_project_status(project_id: str):
    """A simple 'manual' the agent can read."""

    # ------------------- CHANGED -------------------
    # Earlier this tool returned only a string ("Design"). Your latest test expects a JSON/dictionary containing
    #  project_name, status, percentage_complete, etc. So the mock database now stores dictionaries.
    # -----------------------------------------------
    mock_db = {
        "Alpha": {
            "project_name": "Alpha",
            "status": "in_progress",
            "percentage_complete": 75,
            "expected_completion_date": "2024-03-15"
        },
        "Beta": {
            "project_name": "Beta",
            "status": "development",
            "percentage_complete": 40,
            "expected_completion_date": "2024-05-20"
        }
    }

    return mock_db.get(
        project_id,
        {
            "project_name": project_id,
            "status": "unknown"
        }
    )


# Add this function to src/core/agent.py
def failing_tool(project_id: str):
    """A tool that simulates a database outage."""
    raise ConnectionError("Database connection timed out!")


# A directory of available tools so the agent knows what it is allowed to do
TOOLS = {
    "project_status": get_project_status,
    "failing_tool": failing_tool  # New tool for testing
}


# --- 2. THE AGENT (The Assistant) ---
class NativeAgent:
    def __init__(self, system_prompt):
        # Connect to Groq using the key stored in the environment
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # ------------------- CHANGED -------------------
        # Added stronger instructions so the LLM is encouraged
        # to use the available tool instead of answering from memory.
        # -----------------------------------------------
        self.system_prompt = f"""
        {system_prompt}

        You have access to the following tool:

        project_status(project_id)

        For ANY question asking about project status,
        ALWAYS call the project_status tool before answering.

        Respond ONLY in JSON.

        Tool call format:
        {{
            "thought":"...",
            "action":"project_status",
            "action_input": {{
                "project_id":"Alpha"
            }}
        }}

        Final answer format:
        {{
            "answer": {{
                "project_name":"Alpha",
                "status":"in_progress",
                "percentage_complete":75,
                "expected_completion_date":"2024-03-15"
            }}
        }}
        """

        # 'self.history' is the agent's short-term memory. It starts with the instructions.
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _call_llm(self):
        """The 'Brain' function: Sends the notepad to the AI."""

        # Send current memory to Groq
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=self.history,
            temperature=0,  # '0' means 'be precise, not creative'
            response_format={"type": "json_object"}  # Force the AI to speak in JSON
        )

        # ------------------- CHANGED -------------------
        # Added try/except so invalid JSON won't crash the agent.
        # -----------------------------------------------
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "answer": {
                    "error": "Model returned invalid JSON."
                }
            }

    def run(self, user_input):
        """The 'Loop' function: Keeps the assistant working until done."""

        # Add user's question to the notepad
        self.history.append({"role": "user", "content": user_input})

        # Safety limit: only allow 5 tries so the agent doesn't loop forever (cost control)
        for i in range(5):

            # Ask the AI what to do next
            response = self._call_llm()

            # ------------------- CHANGED -------------------
            # Always save the assistant response. Earlier, if the model answered immediately,history contained only System + User.
            # -----------------------------------------------
            self.history.append({
                "role": "assistant",
                "content": json.dumps(response)
            })

            # --- DECISION NODE ---
            # If the AI says 'answer', it has finished the job.
            if "answer" in response:
                return response["answer"]

            # --- ACTION DISPATCH ---
            # If the AI says 'action', it wants to use a tool.
            if "action" in response:

                tool_name = response["action"]
                tool_args = response.get("action_input", {})

                # --- DAY 32: THE SELF-HEALING BLOCK ---
                try:
                    func = TOOLS.get(tool_name)

                    if not func:
                        raise ValueError(f"Tool {tool_name} does not exist.")

                    # Execute tool
                    tool_result = func(**tool_args)

                    print(f"Agent Observed: {tool_result}")

                except Exception as e:

                    # Capture the error and feed it back to the "Brain"
                    error_msg = f"ERROR: Tool execution failed: {str(e)}"

                    print(f"Agent Observation (Error): {error_msg}")

                    tool_result = error_msg

                # Write the tool's result onto the notepad so the AI can read it next time
                self.history.append({
                    "role": "user",
                    "content": f"Observation: {json.dumps(tool_result)}"
                })

        # ------------------- CHANGED -------------------
        # Added fallback return.
        # Without this, Python returns None if 5 iterations finish.
        # -----------------------------------------------
        return {
            "error": "Maximum iterations exceeded."
        }


# 3. ENTRY POINT (Only runs if this file is executed directly)
if __name__ == "__main__":
    agent = NativeAgent("You are a helpful CRM assistant.")
    final_result = agent.run("What is the status of Alpha project?")
    print(f"Result: {final_result}")
import os
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.tools import tool
# 1. Define a SIMPLE tool (Don't use MCP session directly)
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)
@tool("knowledge_search")
def knowledge_search(query: str):
    """Use this to search the knowledge base."""
    # This simulates what your MCP tool *would* do
    return f"Here is the search result for: {query}. (System is functional!)"

# 2. Setup the Agent with the tool
researcher = Agent(
    role='Senior AI Infrastructure Engineer',
    goal='Use the provided tool to answer technical queries.',
    backstory='You are an expert at navigating documentation.',
    tools=[knowledge_search], # Use the defined tool, not the raw session
    llm=llm,
    verbose=True
)

# 3. Setup the Task
task = Task(
    description="How do I search the knowledge base?",
    agent=researcher,
    expected_output="A concise, technical answer."
)

# 4. Run the Crew
crew = Crew(agents=[researcher], tasks=[task], verbose=True)

if __name__ == "__main__":
    print("DEBUG: Starting Crew execution...")
    result = crew.kickoff()
    print("DEBUG: Execution finished!")
    print(result)
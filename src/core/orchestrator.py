import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

# 1. Define the tool globally (it's safe to keep here)
@tool("knowledge_search")
def knowledge_search(query: str):
    """Use this to search the knowledge base."""
    return f"Here is the search result for: {query}. (System is functional!)"

# The LLM config and Agent persona are static across calls, so build them once
# instead of re-creating them (and re-authenticating the LLM client) per request.
_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

_researcher = Agent(
    role='Senior AI Infrastructure Engineer',
    goal='Use the provided tool to answer technical queries.',
    backstory='You are an expert at navigating documentation.',
    tools=[knowledge_search],
    llm=_llm,
    verbose=True
)

# 2. Define the runner function
def run_agent_task(user_query: str):
    task = Task(
        description=user_query,
        agent=_researcher,
        expected_output="A concise, technical answer."
    )

    crew = Crew(agents=[_researcher], tasks=[task], verbose=True)

    # Run the crew and return the result
    result = crew.kickoff()
    return str(result)
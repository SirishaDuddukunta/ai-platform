import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

# 1. Define the tool globally (it's safe to keep here)
@tool("knowledge_search")
def knowledge_search(query: str):
    """Use this to search the knowledge base."""
    return f"Here is the search result for: {query}. (System is functional!)"

# 2. Define the runner function
def run_agent_task(user_query: str):
    # Everything happens inside this function
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    researcher = Agent(
        role='Senior AI Infrastructure Engineer',
        goal='Use the provided tool to answer technical queries.',
        backstory='You are an expert at navigating documentation.',
        tools=[knowledge_search],
        llm=llm,
        verbose=True
    )

    task = Task(
        description=user_query, 
        agent=researcher, 
        expected_output="A concise, technical answer."
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=True)
    
    # Run the crew and return the result
    result = crew.kickoff()
    return str(result)
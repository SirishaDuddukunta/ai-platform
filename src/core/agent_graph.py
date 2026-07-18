# WHAT THIS FILE IS:
# The SAME agent as agent.py (NativeAgent), rebuilt using LangGraph.
# Same brain, same tool, same data — different "skeleton".
#
# WHY BUILD IT TWICE?
# Interviewers ask: "What does LangGraph actually do for you?"
# Having built BOTH versions, you can answer from experience:
#   - agent.py:      the loop is a for-loop, state is a plain list (self.history)
#   - agent_graph.py: the loop is a GRAPH (boxes + arrows), state is a typed object
#
# MENTAL MODEL — A FLOWCHART:
#   START ──> [model node] ──> decision: answered? ──yes──> END
#                  ^                    │no
#                  │                    v
#                  └──────── [tool node]
# The model "thinks", maybe uses a tool, thinks again... until it answers.

# --- Standard library imports ---
import os        # reads environment variables (our API keys) 
import json      # converts between JSON text and Python dicts

# TypedDict lets us declare "a dict that MUST have these exact keys with these types".
# Annotated lets us attach extra info to a type (LangGraph uses this — see AgentState).
from typing import TypedDict, Annotated

# operator.add is literally the "+" operation as a function.
# LangGraph uses it as a rule: "when a node returns this field, ADD (append) to it,
# don't overwrite it". That's how chat history grows instead of being replaced.
import operator

# --- Third-party imports ---
from groq import Groq                      # the LLM API client
from dotenv import load_dotenv             # loads .env file (where GROQ_API_KEY lives)
from qdrant_client import QdrantClient     # the vector database client
from sentence_transformers import SentenceTransformer  # text → vector converter

# THE LANGGRAPH IMPORTS:
# StateGraph = the graph builder ("give me boxes and arrows")
# START, END = special markers for "where the graph begins" and "where it stops"
from langgraph.graph import StateGraph, START, END

# Read the .env file NOW so os.getenv() below can find our keys.
load_dotenv()

# ==========================================
# THE TOOL (identical to agent.py)
# ==========================================
# One Qdrant connection for the whole file. The "_" prefix is a Python
# convention meaning "private — don't import this from other files".
_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

# Load the embedding model ONCE (it's ~90MB — loading per-call would be very slow).
# Must be the SAME model used in ingest.py, or the vectors won't be comparable.
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_search(query: str) -> str:
    """Finds stored text whose MEANING is closest to the query. Same as agent.py."""
    # Step 1: turn the question into a list of 384 numbers (a vector).
    query_vector = _embedder.encode(query).tolist()

    # Step 2: ask Qdrant "which stored vectors are closest to this one?"
    search_result = _client.query_points(
        collection_name="project_docs",  # the "table" we search in
        query=query_vector,              # our question, as numbers
        limit=2,                         # give us the 2 best matches only
    )

    # Step 3: pull the original text out of each match and glue them
    # together with newlines into one string.
    return "\n".join(point.payload["text"] for point in search_result.points)


# The tool registry: maps a tool's NAME (what the LLM says) to the FUNCTION (what runs).
# To add a tool later: write a function, add one line here, mention it in the prompt.
TOOLS = {"semantic_search": semantic_search}

# One Groq client for the whole file.
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Which model to use, and the safety cap on how many times the agent may loop.
MODEL = "llama-3.3-70b-versatile"
MAX_STEPS = 5  # without this, a confused model could loop forever and burn money

# The instructions we give the model. It MUST reply in JSON, in one of two shapes:
# shape 1 = "I want to use a tool", shape 2 = "here is my final answer".
SYSTEM_PROMPT = """You are a Senior AI Infrastructure agent.
1. To call a tool, return: {"thought": "...", "action": "semantic_search", "action_input": {"query": "..."}}
2. To give the final answer, return: {"thought": "...", "answer": {...}}
Respond ONLY in valid JSON."""


# ==========================================
# STEP 1: THE STATE (the "clipboard" passed between nodes)
# ==========================================
# In agent.py, state was self.history — a list the class mutated directly.
# In LangGraph, state is an explicit object that flows through the graph.
# Every node RECEIVES the current state and RETURNS the pieces it wants to change.
class AgentState(TypedDict):
    # The chat history. The Annotated[..., operator.add] part is a RULE for LangGraph:
    # "when a node returns {'messages': [new_msg]}, APPEND new_msg to the existing
    # list" — instead of replacing the whole list. That's how history accumulates.
    messages: Annotated[list, operator.add]

    # A simple counter: how many times has the model been called?
    # (Replaces the `for i in range(5)` loop counter from agent.py.)
    steps: int

    # Starts as None. The moment the model gives a final answer, we store it here.
    # The router (should_continue below) checks this field to know when to stop.
    final_answer: dict | None   # "dict | None" = "either a dict, or nothing yet"


# ==========================================
# STEP 2: THE NODES (each node = one box in the flowchart)
# ==========================================
def call_model(state: AgentState) -> dict:
    """NODE 1: Ask the LLM what to do next. (= NativeAgent._call_llm in agent.py)"""
    # Send the ENTIRE conversation so far to the model.
    response = _groq.chat.completions.create(
        model=MODEL,
        messages=state["messages"],              # full history from state
        temperature=0,                           # 0 = deterministic, no randomness
        response_format={"type": "json_object"}, # FORCES the reply to be valid JSON
    )

    # Pull out the reply text, then parse the JSON text into a Python dict.
    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # CASE A: the model gave a final answer.
    if "answer" in parsed:
        # A node returns ONLY the state fields it wants to update:
        return {
            "messages": [{"role": "assistant", "content": raw}],  # append this turn to history
            "steps": state["steps"] + 1,                           # count this LLM call
            "final_answer": parsed["answer"],                      # capture the answer → router will stop
        }

    # CASE B: the model asked for a tool. Just record the turn and count it.
    # (final_answer stays None, so the router will send us to the tool node.)
    return {
        "messages": [{"role": "assistant", "content": raw}],
        "steps": state["steps"] + 1,
    }


def run_tool(state: AgentState) -> dict:
    """NODE 2: Execute whichever tool the model just requested."""
    # The model's tool request is the LAST message in history. Parse it back to a dict.
    last = json.loads(state["messages"][-1]["content"])

    tool_name = last.get("action")             # e.g. "semantic_search"
    tool_args = last.get("action_input", {})   # e.g. {"query": "Alpha status"}

    # SELF-CORRECTION (same pattern as agent.py): never crash the agent.
    # If the tool is unknown or explodes, turn the error into TEXT and feed it
    # back to the model — the model can then apologize, retry, or work around it.
    try:
        if tool_name not in TOOLS:
            result = f"Error: Tool '{tool_name}' not found."
        else:
            # TOOLS[tool_name] fetches the function; (**tool_args) calls it with
            # the dict unpacked into keyword arguments: semantic_search(query="...")
            result = TOOLS[tool_name](**tool_args)
    except Exception as e:
        result = f"TOOL_EXECUTION_ERROR: {e}"

    # Report the tool's output back into history as an "Observation".
    # Next loop, the model reads this and decides what to do with it.
    return {"messages": [{"role": "user", "content": f"Observation: {result}"}]}


# ==========================================
# STEP 3: THE ROUTER (the diamond decision box in the flowchart)
# ==========================================
# After call_model runs, LangGraph calls THIS function to ask: "where next?"
# It returns a STRING — the name of the route to take.
# (In agent.py this logic was the `if "answer" in response: return` line
#  plus the for-loop's 5-iteration cap.)
def should_continue(state: AgentState) -> str:
    # Did call_model capture a final answer? Then we're done.
    if state.get("final_answer") is not None:
        return "end"
    # Have we hit the safety cap? Stop anyway (prevents infinite loops / runaway cost).
    if state["steps"] >= MAX_STEPS:
        return "end"
    # Otherwise: the model asked for a tool → go run it.
    return "tool"


# ==========================================
# STEP 4: WIRE THE GRAPH (draw the boxes and arrows)
# ==========================================
def build_agent():
    # Create an empty graph whose state has the AgentState shape.
    graph = StateGraph(AgentState)

    # Register our two functions as named nodes (the boxes).
    graph.add_node("model", call_model)
    graph.add_node("tool", run_tool)

    # ARROW 1: when the graph starts, go to the "model" node first.
    graph.add_edge(START, "model")

    # ARROW 2 (the smart one): after "model" runs, call should_continue().
    # The dict maps its return string → which node to go to:
    #   returns "tool" → go to the "tool" node
    #   returns "end"  → go to END (graph stops, invoke() returns)
    graph.add_conditional_edges(
        "model",
        should_continue,
        {"tool": "tool", "end": END},
    )

    # ARROW 3: after a tool runs, ALWAYS go back to the model
    # (so it can read the Observation and think again). This closes the loop.
    graph.add_edge("tool", "model")

    # compile() checks the wiring is valid and returns a runnable agent.
    return graph.compile()


# Build the agent ONCE when this file is imported (not per-request — compiling is work).
_agent = build_agent()


# ==========================================
# STEP 5: THE PUBLIC ENTRY POINT (what other files call)
# ==========================================
def run_graph_agent(user_input: str) -> dict:
    """Give it a question, get back the answer dict. Mirrors NativeAgent.run()."""
    # Build the STARTING state: system prompt + the user's question, zero steps, no answer.
    initial_state: AgentState = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "steps": 0,
        "final_answer": None,
    }

    # invoke() runs the whole flowchart: model → maybe tool → model → ... → END.
    # It returns the FINAL state after the graph stops.
    result = _agent.invoke(initial_state)

    # Return the answer if the model produced one; otherwise a graceful fallback
    # (this happens if we hit MAX_STEPS without converging).
    return result["final_answer"] or {
        "status": "unknown",
        "message": "Agent did not converge within the step limit.",
    }


# This block ONLY runs when you execute the file directly:
#   python -m src.core.agent_graph
# It does NOT run when main.py imports this file.
if __name__ == "__main__":
    import sys
    # Windows console fix: default encoding can't print emoji; force UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    # A quick smoke test — same question you'd ask NativeAgent.
    print(run_graph_agent("What is the status of the Alpha project?"))
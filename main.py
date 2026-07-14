# FastAPI Endpoints
# This file wires together the HTTP API: every @app.post/@app.get function
# below becomes a URL that a client (browser, curl, frontend app) can call.

import json          # parses the JSON string arguments the LLM sends back for a tool call
import sys            # used below to fix a Windows console encoding quirk
import time           # used to measure how long things take (latency)
import uuid           # generates unique IDs for background tasks
from contextlib import asynccontextmanager  # lets us define startup/shutdown code for the app

# On Windows, stdout defaults to the system codepage (cp1252), which can't
# encode the emoji our print() calls use (e.g. vector_memory.py's "✅"/"⚠️").
# Without this, those prints raise UnicodeEncodeError and crash app startup.
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # loads secrets (API keys) from a local .env file into the environment
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request  # core FastAPI building blocks
from sentence_transformers import SentenceTransformer  # the embedding model class (turns text into vectors)
from starlette.concurrency import run_in_threadpool  # runs a blocking (non-async) function without freezing the server

from src.core import vector_memory  # our own module: stores/searches "memory" as vectors (semantic search)
from src.core.factory import LLMFactory  # our own class that wraps calls to the LLM (Groq) and embeddings
from src.core.logger import log_trace  # our own helper: writes structured log lines for debugging
from src.core.memory import load_history, save_history  # our own module: saves/loads plain chat history per user
from src.core.orchestrator import run_agent_task  # our own module: runs the multi-step "agent" reasoning loop
from src.core.tools import (
    TOOLS_SCHEMA,          # the JSON description of tools that we tell the LLM it's allowed to call
    get_server_status,     # actual Python function behind the "get_server_status" tool
    query_database,        # actual Python function behind the "query_database" tool
    restart_server,        # actual Python function behind the "restart_server" tool
    security_scanner,      # checks user input for dangerous text before we do anything with it
)
from src.core.vector_memory import search_memory, upsert_to_vector_db  # search/save semantic memories
from src.schemas.invoice import Invoice  # the Pydantic model describing what an "invoice" looks like

load_dotenv()  # actually read the .env file now, so os.environ has our API keys
log_trace("System", {"status": "Application is booting up..."})  # write a log line marking startup

# In a real app these would live in Redis/Postgres; kept in-memory for now.
ml_models = {}    # holds heavy objects (like the loaded LLM factory) shared across all requests
task_store = {}   # holds background task results, keyed by task_id, so /engineer/status can look them up

# Maps a tool's name (string) to the actual Python function that performs it.
# This lets us call the right function dynamically instead of writing if/elif chains.
TOOL_DISPATCH = {
    "get_server_status": get_server_status,
    "restart_server": restart_server,
    "query_database": query_database,
}


def execute_tool_calls(tool_calls):
    """Runs each tool the model requested and returns the resulting `tool` messages."""
    tool_messages = []  # will collect one chat "message" per tool call, to feed back to the LLM
    for tool_call in tool_calls:  # the LLM can ask for multiple tools in one turn
        f_name = tool_call.function.name  # e.g. "get_server_status"
        args = json.loads(tool_call.function.arguments)  # the LLM sends args as a JSON string; parse it into a dict
        func = TOOL_DISPATCH.get(f_name)  # look up the real function; None if the LLM asked for an unknown tool

        try:
            # If we found a matching function, call it with the args as keyword arguments
            # (e.g. func(hostname="prod-db-01")). Otherwise, report that the tool doesn't exist.
            result = func(**args) if func else f"Error: Tool {f_name} not found."
        except Exception as e:
            # Self-correction: feed the error back to the model instead of crashing.
            result = f"TOOL_EXECUTION_ERROR: {str(e)}"

        # Build the special "tool" role message format the LLM API expects as a reply
        # to its tool call, so it can see the result and respond to the user.
        tool_messages.append(
            {
                "role": "tool",                    # tells the LLM "this is a tool's output, not a person talking"
                "tool_call_id": tool_call.id,       # links this result back to the specific call the LLM made
                "name": f_name,                     # which tool produced this result
                "content": str(result),             # the actual result, always as a string
            }
        )
    return tool_messages  # give all the tool results back to the caller


# The "personality"/instructions we could give the LLM for a general agent persona.
# (Currently defined but not directly used by the endpoints below.)
SYSTEM_PROMPT = {
    "role": "system",  # "system" messages set behavior/rules, as opposed to "user" or "assistant" messages
    "content": (
        "You are 'AI-Eng-Core', a Senior AI Engineer specializing in Agentic Workflows. "
        "Your mission is to ensure the AI platform is efficient, resilient, and accurate. "
        "PRINCIPLES: "
        "1. Latency is the enemy. Be concise to save tokens and time. "
        "2. If an LLM call fails, suggest optimizations for the prompt or tool schema. "
        "3. No corporate fluff. Provide technical solutions and reasoning traces."
    ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP: Load heavy resources ONCE and share them across modules ---
    # Everything before `yield` runs once when the server starts (not per-request).
    print("Loading heavy resources...")
    shared_embedder = SentenceTransformer("all-MiniLM-L6-v2")  # load the embedding model into memory (slow, do once)
    vector_memory.set_encoder(shared_embedder)  # tell the vector_memory module to reuse this same model
    ml_models["factory"] = LLMFactory(embed_model=shared_embedder)  # build our LLM wrapper, reusing the embedder
    vector_memory.init_qdrant()  # connect to / set up the Qdrant vector database

    yield  # The app is now running; every incoming request is handled while execution is paused here

    # --- SHUTDOWN: Clean up resources ---
    # Everything after `yield` runs once when the server is stopping.
    ml_models.clear()  # drop references to free memory
    print("Resources cleaned up.")


# Create the actual FastAPI application, wiring in our startup/shutdown logic above.
app = FastAPI(title="Sirisha's AI Platform", lifespan=lifespan)


# ==========================================
# LATENCY TRACKING MIDDLEWARE
# ==========================================
# Middleware runs around EVERY request, regardless of which endpoint is called.
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()              # record the time right before the request is handled
    response = await call_next(request)   # actually run the matching endpoint function
    process_time = time.time() - start_time  # how long that took, in seconds
    response.headers["X-Process-Time"] = str(process_time)  # attach it as a response header for debugging
    return response  # send the (now-annotated) response back to the client


# ==========================================
# STRUCTURED EXTRACTION
# ==========================================
@app.post("/extract/invoice")  # defines a POST endpoint at /extract/invoice
async def api_extract_invoice(text: str):  # `text` is read from the request (query/body param)
    factory = ml_models["factory"]  # grab our shared LLM wrapper
    try:
        # Ask the LLM to read `text` and return data matching the `Invoice` schema.
        # run_in_threadpool moves this blocking call off the main event loop so
        # other requests aren't blocked while we wait for the LLM.
        invoice, stats = await run_in_threadpool(factory.get_structured, Invoice, text)
        return {"data": invoice, "metrics": stats}  # FastAPI auto-converts this dict to JSON
    except Exception as e:
        # Convert any failure into a proper HTTP 500 error response instead of crashing the server.
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# TOOL-CALLING CHAT (stateless, no memory)
# ==========================================
@app.post("/chat/tools")
async def api_tool_chat(user_query: str):
    factory = ml_models["factory"]  # grab our shared LLM wrapper

    is_safe, message = security_scanner(user_query)  # check the input for dangerous patterns first
    if not is_safe:
        return {"response": message, "security_status": "REJECTED"}  # stop here, don't call the LLM at all

    # Build the initial conversation: one system instruction + the user's actual question.
    messages = [
        {"role": "system", "content": "You are a DevOps Assistant. If a tool fails, explain the error."},
        {"role": "user", "content": user_query},
    ]

    start_total = time.time()  # start a timer to measure total latency for this request
    # Ask the LLM to respond, telling it which tools (TOOLS_SCHEMA) it's allowed to use.
    response = await run_in_threadpool(factory.chat_with_tools, messages, TOOLS_SCHEMA)
    reply = response.choices[0].message  # the actual message object is nested inside the raw API response

    if not reply.tool_calls:
        # The model answered directly without needing any tool — just return its text.
        return {"response": reply.content, "security_status": "CLEAN"}

    # The model wants to call one or more tools before it can answer.
    messages.append(reply.model_dump())  # add the model's own "I want to call a tool" message to the history
    messages.extend(execute_tool_calls(reply.tool_calls))  # actually run the tool(s) and append their results

    # Send the conversation (now including tool results) back to the LLM so it can
    # produce a final, human-readable answer based on what the tools returned.
    final = await run_in_threadpool(factory.chat_with_tools, messages, TOOLS_SCHEMA)

    return {
        "response": final.choices[0].message.content,               # the model's final answer text
        "latency_ms": round((time.time() - start_total) * 1000, 2),  # total time taken, in milliseconds
        "security_status": "CLEAN",
        "version": "1.1.0-day11",
    }


# ==========================================
# CHAT WITH PERSISTENT + SEMANTIC MEMORY
# ==========================================
@app.post("/chat")
async def chat_with_memory(user_input: str, user_id: str = "default_user"):
    factory = ml_models["factory"]  # grab our shared LLM wrapper
    log_trace("User Input", {"user": user_id, "input": user_input})  # log what the user asked

    # 1. Semantic retrieval of relevant past turns
    # Search the vector database for up to 3 past messages that are semantically
    # similar to what the user just said (not just keyword matches).
    relevant_memories = await run_in_threadpool(search_memory, user_id, user_input, 3)
    context_text = "\n".join(relevant_memories)  # turn the list of memories into one text block
    log_trace("Memory Retrieval", {"found_count": len(relevant_memories), "context": context_text})

    # 2. Load session history
    session_history = await run_in_threadpool(load_history, user_id)  # load this user's full past conversation
    if not session_history:
        # First-ever message from this user: start a fresh conversation with a system message.
        session_history = [{"role": "system", "content": f"You are AI-Eng-Core. The user is {user_id}."}]
    session_history.append({"role": "user", "content": user_input})  # add the new message to the history

    # A separate system message injecting the retrieved memories as extra context.
    memory_instruction = {
        "role": "system",
        "content": f"Use the following retrieved memories to answer the user if relevant:\n{context_text}",
    }
    full_prompt = [memory_instruction] + session_history  # memory instruction goes first, then the real history
    log_trace("LLM Input", {"prompt_len": len(full_prompt)})  # log how many messages we're sending

    # 3. Generation (+ tool calling)
    response = await run_in_threadpool(factory.chat_with_tools, full_prompt, TOOLS_SCHEMA)
    if response is None:
        # Defensive check: if the LLM call somehow returned nothing, fail loudly instead of crashing later.
        log_trace("Error", {"status": "LLM returned None"})
        raise HTTPException(status_code=500, detail="The AI assistant failed to provide a response.")

    message = response.choices[0].message  # unwrap the actual message from the raw API response

    if message.tool_calls:
        # The model wants to use a tool before answering — same pattern as /chat/tools above.
        session_history.append(message.model_dump())         # record the model's "I want to call a tool" turn
        session_history.extend(execute_tool_calls(message.tool_calls))  # run the tool(s), append their results

        # Ask the model again, now that it has the tool results, to get its real answer.
        final_turn = await run_in_threadpool(factory.chat_with_tools, session_history, TOOLS_SCHEMA)
        final_answer = final_turn.choices[0].message.content or "Analysis complete."  # fallback if content is empty
    else:
        final_answer = message.content or "How can I help?"  # no tool needed; use the direct reply

    session_history.append({"role": "assistant", "content": final_answer})  # record the final answer in history
    log_trace("LLM Output", {"response": final_answer[:50] + "..."})  # log a preview (avoid huge log lines)

    # 4. Persist history + upsert new memories
    await run_in_threadpool(save_history, user_id, session_history)  # save the updated conversation for next time
    await run_in_threadpool(upsert_to_vector_db, user_id, "user", user_input)      # store this turn as a memory
    await run_in_threadpool(upsert_to_vector_db, user_id, "assistant", final_answer)  # store the reply as a memory

    return {"response": final_answer}  # send just the final text back to the client


# ==========================================
# EMBEDDINGS / VECTOR LIBRARY
# ==========================================
@app.post("/engineer/embed")
async def api_generate_embedding(text: str):
    """Test endpoint to see how the AI 'digitizes' technical concepts."""
    factory = ml_models["factory"]
    vector = await run_in_threadpool(factory.get_embedding, text)  # convert the text into a list of numbers
    return {
        "text": text,
        "vector_dimensions": len(vector),  # how many numbers make up the vector (e.g. 384)
        "sample": vector[:5],              # just show the first 5 numbers, not the whole vector
        "status": "SEMANTIC_READY",
    }


@app.post("/engineer/index")
async def api_index_document(text: str, doc_id: str):
    """Store technical knowledge in the long-term vector brain."""
    factory = ml_models["factory"]
    try:
        # Embed `text` and store it permanently (in Chroma) under `doc_id`, so it can be
        # searched/retrieved later — this is how the system "learns" new documents.
        status = await run_in_threadpool(factory.add_to_library, text, doc_id, {"source": "manual_upload"})
        return {"status": status, "message": "Knowledge acquired."}
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# BACKGROUND AGENT TASKS
# ==========================================
@app.post("/engineer/ask")
async def ask_agent_async(query: str, background_tasks: BackgroundTasks):
    """Returns a task_id immediately and runs the agent in the background."""
    task_id = str(uuid.uuid4())  # generate a random unique ID to track this task
    task_store[task_id] = {"status": "processing", "result": None}  # mark it as started

    def process_agent_task(tid, q):
        # This runs later, in the background, after the HTTP response has already been sent.
        try:
            result = run_agent_task(q)  # run the (potentially slow) multi-step agent reasoning loop
            task_store[tid] = {"status": "completed", "result": result}  # save the successful result
        except Exception as e:
            task_store[tid] = {"status": "failed", "error": str(e)}  # save the failure reason

    background_tasks.add_task(process_agent_task, task_id, query)  # schedule the function above to run after we return

    return {"status": "accepted", "task_id": task_id}  # respond immediately; client polls /engineer/status/{task_id}


@app.get("/engineer/status/{task_id}")  # {task_id} in the URL becomes the task_id function parameter
async def get_task_status(task_id: str):
    """Check the status of a long-running research task."""
    task = task_store.get(task_id)  # look up what we know about this task
    if not task:
        return {"status": "not_found"}  # no such task_id was ever created
    return task  # {"status": "processing"|"completed"|"failed", ...}

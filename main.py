# FastAPI Endpoints
import os, json, time, uuid
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from dotenv import load_dotenv
from src.core.memory import load_history, save_history

# Core Platform Imports
from src.core.factory import LLMFactory
from src.core.processor import DocumentProcessor
from src.core.tools import (
    get_server_status, restart_server, query_database, 
    TOOLS_SCHEMA, security_scanner
)
from src.schemas.invoice import Invoice

# 1. INITIALIZATION
load_dotenv()
app = FastAPI(title="Sirisha's AI Platform")
factory = LLMFactory()
# ==========================================
# DAY 10: MIDDLEWARE FOR LATENCY TRACKING
# ==========================================
# What: Middleware that intercepts every request to measure speed.
# Why: To provide real-time performance metrics (X-Process-Time) in headers.
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ==========================================
# DAY 4: RESILIENCE & EXTRACTION (API)
# ==========================================
# What: Extracts structured JSON from messy text using Pydantic.
@app.post("/extract/invoice")
async def api_extract_invoice(text: str):
    try:
        invoice, stats = factory.get_structured(Invoice, text)
        return {"data": invoice, "metrics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# DAY 8, 9, 10, 11: CONSOLIDATED TOOL CHAT
# ==========================================
# What: Multi-step tool loop with error handling and observability.
# Why: This allows the AI to decide, act, fail, and self-correct.
@app.post("/chat/tools")
async def api_tool_chat(user_query: str):
    # --- DAY 11: THE SECURITY GATE ---
    # What: Intercepting the query BEFORE the LLM sees it.
    is_safe, message = security_scanner(user_query)
    if not is_safe:
        return {"response": message, "security_status": "REJECTED"}

    messages = [
        {"role": "system", "content": "You are a DevOps Assistant. If a tool fails, explain the error."},
        {"role": "user", "content": user_query}
    ]
    
    start_total = time.time()
    
    # [Day 8] Step 1: Decision Phase
    response_message = factory.chat_with_tools(messages, TOOLS_SCHEMA)
    
    if response_message.tool_calls:
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            f_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # [Day 9] Step 2: Resilient Execution
            try:
                if f_name == "get_server_status":
                    result = get_server_status(args['hostname'])
                elif f_name == "restart_server":
                    result = restart_server(args['hostname'])
                elif f_name == "query_database":
                    result = query_database(args['query_string'])
                else:
                    result = f"Error: Tool {f_name} not found."
            except Exception as e:
                # [Day 9] Step 3: Self-Correction Loop
                result = f"TOOL_EXECUTION_ERROR: {str(e)}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": f_name,
                "content": result
            })
        
        # [Day 8] Final Summary Phase
        final_answer = factory.chat_with_tools(messages, TOOLS_SCHEMA)
        
        # [Day 10] Metadata Return
        return {
            "response": final_answer.content,
            "latency_ms": round((time.time() - start_total) * 1000, 2),
            "security_status": "CLEAN",
            "version": "1.1.0-day11"
        }
    
    return {"response": response_message.content, "security_status": "CLEAN"}

# ==========================================
# UNWANTED / LEGACY CODE (KEEPING FOR HISTORY)
# ==========================================
"""
# DAY 4: CLI VERSION
def run_day4_chaos_test():
    factory = LLMFactory(provider="groq")
    raw_data = "I bought an AI chip from Silicon-Valley. Qty is 0, price is TBD."
    invoice, stats = factory.get_structured(Invoice, raw_data)
    print(f"✅ Success! Vendor: {invoice.vendor}")

# DAY 6: CLI CHUNKING TEST
def run_day6_chunking_test():
    mega_bill = "VENDOR: Silicon-Valley. " + ("ITEM: AI Chip, PRICE: 1200. " * 50)
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    chunks = processor.split_text(mega_bill)
    print(f"📦 Document split into {len(chunks)} chunks.")

# DAY 7: CONSOLIDATED CLI BOOTSTRAP
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
"""
def trim_history(history: list, max_messages: int = 10):
    """
    Keeps the System Prompt (index 0) and the last N-1 messages.
    Ensures the 'AI-Eng-Core' persona is never deleted.
    """
    if len(history) <= max_messages:
        return history
    
    # Preserve the System Prompt at [0]
    # Take the most recent messages to fill the rest of the limit
    # (max_messages - 1) because the system prompt takes 1 slot
    trimmed = [history[0]] + history[-(max_messages - 1):]
    
    print(f"✂️  Trimming history: Reduced from {len(history)} to {len(trimmed)} messages.")
    return trimmed

# In a real app, this would be in a DB (like Redis or PostgreSQL)
# For Day 12, we will use a simple in-memory store
#  DAY 13: THE AI ENGINEER PERSONA ---
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are 'AI-Eng-Core', a Senior AI Engineer specializing in Agentic Workflows. "
        "Your mission is to ensure the AI platform is efficient, resilient, and accurate. "
        "PRINCIPLES: "
        "1. Latency is the enemy. Be concise to save tokens and time. "
        "2. If an LLM call fails, suggest optimizations for the prompt or tool schema. "
        "3. No corporate fluff. Provide technical solutions and reasoning traces."
    )
}

@app.post("/chat")
async def chat_with_memory(user_input: str, user_id: str = "default_user"):
    # 1. Load history
    session_history = load_history(user_id)
    if not session_history:
        session_history = [{"role": "system", "content": f"You are AI-Eng-Core. The user is {user_id}."}]
    
    # 2. Append user input
    session_history.append({"role": "user", "content": user_input})
    
    # 3. Get LLM Response
    response = factory.chat_with_tools(session_history, TOOLS_SCHEMA)
    
    # Extract message properly (Fixes AttributeError)
    message = response.choices[0].message
    
    # 4. Handle Tool Calls
    if message.tool_calls:
        # Append the assistant's tool-calling message to history
        # IMPORTANT: Use .model_dump() to make it JSON serializable
        session_history.append(message.model_dump())
        
        for tool_call in message.tool_calls:
            f_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            try:
                # Execution logic
                if f_name == "get_server_status":
                    result = get_server_status(args.get("hostname"))
                elif f_name == "query_database":
                    result = query_database(args.get("query_string"))
                else:
                    result = f"Error: Tool {f_name} not found."
            except Exception as e:
                result = f"TOOL_ERROR: {str(e)}"

            # Append tool result
            session_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": f_name,
                "content": str(result)
            })

        # Final turn to get the summary
        final_turn = factory.chat_with_tools(session_history, TOOLS_SCHEMA)
        final_answer = final_turn.choices[0].message.content or "Analysis complete."
        session_history.append({"role": "assistant", "content": final_answer})
    
    else:
        # Standard chat case
        final_answer = message.content or "How can I help?"
        session_history.append({"role": "assistant", "content": final_answer})

    # 5. Save History (Fixes TypeError)
    # We ensure session_history is a list of dictionaries, not raw objects
    save_history(user_id, session_history)
    
    return {"response": final_answer}
@app.post("/engineer/embed")
async def api_generate_embedding(text: str):
    """
    Test endpoint to see how the AI 'digitizes' technical concepts.
    """
    vector = factory.get_embedding(text)
    return {
        "text": text,
        "vector_dimensions": len(vector),
        "sample": vector[:5], # Show the first 5 numbers
        "status": "SEMANTIC_READY"
    }
@app.post("/engineer/index")
async def api_index_document(text: str, doc_id: str):
    """
    Store technical knowledge in the long-term vector brain.
    """
    try:
        status = factory.add_to_library(text, doc_id, {"source": "manual_upload"})
        return {"status": status, "message": "Knowledge acquired."}
    except Exception as e:
        return {"error": str(e)}
# main.py
from src.core.orchestrator import run_agent_task

""" @app.post("/engineer/ask")
async def ask_agent(query: str):    
    Day 24 Integration Point: 
    Sends a query from the API to the Orchestrator.
    try:
        # We call the function we just refactored
        result = run_agent_task(query)
        return {"status": "success", "answer": result}
    except Exception as e:
        return {"status": "error", "message": str(e)} """

# --- MOCK STORAGE (Replace with Redis in production) ---
task_store = {}

@app.post("/engineer/ask")
async def ask_agent_async(query: str, background_tasks: BackgroundTasks):
    """
    Day 25: Returns a task_id immediately and runs the agent in the background.
    """
    task_id = str(uuid.uuid4())
    task_store[task_id] = {"status": "processing", "result": None}
    
    # Define the background worker
    def process_agent_task(tid, q):
        try:
            # Import and call your orchestrator function
            from src.core.orchestrator import run_agent_task
            result = run_agent_task(q)
            task_store[tid] = {"status": "completed", "result": result}
        except Exception as e:
            task_store[tid] = {"status": "failed", "error": str(e)}

    # Add the worker to background tasks
    background_tasks.add_task(process_agent_task, task_id, query)
    
    return {"status": "accepted", "task_id": task_id}

@app.get("/engineer/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Check the status of a long-running research task.
    """
    task = task_store.get(task_id)
    if not task:
        return {"status": "not_found"}
    return task

from src.core.vector_memory import init_qdrant, upsert_to_vector_db, search_memory
# 1. Run at startup
@app.on_event("startup")
def startup_event():
    init_qdrant()

@app.post("/chat")
async def chat_with_memory(user_input: str, user_id: str = "default_user"):
    # ... (your existing load/get response logic) ...
    
    # After you get the response from your LLM:
    # 1. Keep your working JSON save
    save_history(user_id, session_history)
    
    # 2. SHADOW SAVE to Vector DB
    # We save both the user input and the assistant response
    upsert_to_vector_db(user_id, "user", user_input)
    upsert_to_vector_db(user_id, "assistant", message.content)
    
    return {"response": message.content}
import os
import json
import time
from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv

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

# [Line 144] Initialize memory with the Senior AI Engineer persona
session_history = [SYSTEM_PROMPT]

@app.post("/chat")
async def chat_with_memory(user_input: str):
    global session_history
    # 1. Append user input to history
    session_history.append({"role": "user", "content": user_input})

    # 2. First call to LLM
    response_message = factory.chat_with_tools(session_history, TOOLS_SCHEMA)
    
    # CRITICAL: Append the full object to history (saves tool_calls metadata)
    session_history.append(response_message)

    # 3. Handle Tool Execution
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            f_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            try:
                # Logic for your specific tools
                if f_name == "get_server_status":
                    result = get_server_status(args["hostname"])
                elif f_name == "query_database":
                    result = query_database(args["query_string"])
                else:
                    result = f"Error: Tool {f_name} not found."
            except Exception as e:
                result = f"TOOL_EXECUTION_ERROR: {str(e)}"

            # 4. Append tool result to history
            session_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": f_name,
                "content": result
            })

        # 5. Final Turn: Ask the AI to summarize the results
        # This turn is what prevents the 'null' response!
        final_turn = factory.chat_with_tools(session_history, TOOLS_SCHEMA)
        
        # Ensure we have a string even if the model is being shy
        final_content = final_turn.content or "Task completed. No further technical notes."
        
        session_history.append({"role": "assistant", "content": final_content})
        # --- DAY 14: THE SLIDING WINDOW ---
        session_history = trim_history(session_history, max_messages=10)
        # ----------------------------------
        return {"response": final_content}
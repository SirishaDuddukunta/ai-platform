import os
import json
import time
from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv

# Core Platform Imports
from src.core.factory import LLMFactory
from src.core.processor import DocumentProcessor
from src.core.tools import get_server_status, restart_server, query_database, TOOLS_SCHEMA
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
# What: Extracts structured JSON from messy text using Pydantic. [cite: 17]
@app.post("/extract/invoice")
async def api_extract_invoice(text: str):
    try:
        invoice, stats = factory.get_structured(Invoice, text)
        return {"data": invoice, "metrics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# DAY 8, 9, 10: CONSOLIDATED TOOL CHAT
# ==========================================
# What: Multi-step tool loop with error handling and observability.
# Why: This allows the AI to decide, act, fail, and self-correct.
@app.post("/chat/tools")
async def api_tool_chat(user_query: str):
    messages = [
        {"role": "system", "content": "You are a DevOps Assistant. If a tool fails, explain the error."},
        {"role": "user", "content": user_query}
    ]
    
    start_total = time.time()
    # [Day 8] Step 1: Decision
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
                # [Day 9] Step 3: Feedback Loop
                result = f"TOOL_EXECUTION_ERROR: {str(e)}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": f_name,
                "content": result
            })
        
        # [Day 8] Final Summary
        final_answer = factory.chat_with_tools(messages, TOOLS_SCHEMA)
        
        # [Day 10] Metadata Return
        return {
            "response": final_answer.content,
            "latency_ms": round((time.time() - start_total) * 1000, 2),
            "version": "1.0.1-day10"
        }
    
    return {"response": response_message.content}

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
    # run_day4_chaos_test()
    # run_day6_chunking_test()
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
"""
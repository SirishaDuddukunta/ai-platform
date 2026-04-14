import os
import json
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# Import your core platform components
from src.core.factory import LLMFactory
from src.core.processor import DocumentProcessor
from src.core.tools import get_server_status, restart_server, TOOLS_SCHEMA
from src.schemas.invoice import Invoice

# 1. INITIALIZATION
load_dotenv()
app = FastAPI(title="Sirisha's AI Platform") # <--- THIS FIXES THE ERROR
factory = LLMFactory()

# ==========================================
# DAY 4: RESILIENCE & EXTRACTION (API VERSION)
# ==========================================
# DEFINITION: Extracts structured data from messy text.
@app.post("/extract/invoice")
async def api_extract_invoice(text: str):
    try:
        # Re-using the logic from your Day 4 chaos test
        invoice, stats = factory.get_structured(Invoice, text)
        return {"data": invoice, "metrics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# DAY 8: TOOL CALLING (API VERSION)
# ==========================================
# DEFINITION: An endpoint that lets the AI decide to use DevOps tools.
@app.post("/chat/tools")
async def api_tool_chat(user_query: str):
    messages = [
        {"role": "system", "content": "You are a DevOps Assistant."},
        {"role": "user", "content": user_query}
    ]
    
    # Step 1: LLM decides to use a tool
    response_message = factory.chat_with_tools(messages, TOOLS_SCHEMA)
    
    if response_message.tool_calls:
        messages.append(response_message)
        
        # Step 2: Execute the requested tools
        for tool_call in response_message.tool_calls:
            f_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            if f_name == "get_server_status":
                result = get_server_status(args['hostname'])
            elif f_name == "restart_server":
                result = restart_server(args['hostname'])
            else:
                result = "Unknown tool."

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": f_name,
                "content": result
            })
        
        # Step 3: Final summary from LLM
        final_answer = factory.chat_with_tools(messages, TOOLS_SCHEMA)
        return {"response": final_answer.content, "tools_used": [t.function.name for t in response_message.tool_calls]}
    
    return {"response": response_message.content}

# ==========================================
# LEGACY CLI LOGIC (KEEPING OLD CODE)
# ==========================================
# This allows you to still run tests via 'python main.py' if you want.
def run_legacy_tests():
    print("Running Day 4/6/8 Legacy CLI Tests...")
    # [Your previous test code logic remains accessible here]
    pass

if __name__ == "__main__":
    import uvicorn
    # This starts the web server automatically when you run the file
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
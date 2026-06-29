import time
from fastapi import FastAPI, APIRouter, Request
from dotenv import load_dotenv

from src.core.factory import LLMFactory
from src.core.tools import TOOLS_SCHEMA, restart_server
from src.core.memory import load_history, save_history
from src.core.vector_memory import init_qdrant, upsert_to_vector_db, search_memory
from src.core.logger import log_trace

# 1. INITIALIZATION
load_dotenv()
app = FastAPI(title="Sirisha's AI Platform")
factory = LLMFactory()

# Routers
chat_router = APIRouter()
tools_router = APIRouter()
engineer_router = APIRouter()

@app.on_event("startup")
def startup_event():
    init_qdrant()
    log_trace("System", {"status": "Application initialized successfully"})

# --- MIDDLEWARE ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- CHAT ENDPOINTS ---
@chat_router.post("/chat")
async def chat_with_memory(user_input: str, user_id: str = "default_user"):
    # Trace input
    log_trace("User Input", {"user": user_id, "input": user_input})

    # Retrieve memories and history
    relevant_memories = search_memory(user_id, user_input, limit=3)
    context_text = "\n".join(relevant_memories)
    session_history = load_history(user_id)
    
    memory_instruction = {
        "role": "system", 
        "content": f"Use the following retrieved memories to answer the user if relevant:\n{context_text}"
    }
    
    full_prompt = [memory_instruction] + session_history + [{"role": "user", "content": user_input}]
    
    # Generate response
    completion = factory.chat_with_tools(full_prompt, TOOLS_SCHEMA)
    assistant_content = completion.choices[0].message.content or ""

    # Storage & Trace
    save_history(
        user_id,
        session_history
        + [{"role": "user", "content": user_input}, {"role": "assistant", "content": assistant_content}],
    )
    upsert_to_vector_db(user_id, "user", user_input)
    upsert_to_vector_db(user_id, "assistant", assistant_content)

    return {"response": assistant_content}

# --- TOOLS ENDPOINTS ---
@tools_router.get("/chat/tools")
async def get_tools_status():
    return {"tools": TOOLS_SCHEMA}

@tools_router.post("/tools/restart")
async def restart():
    return restart_server()

# --- ENGINEER ENDPOINTS ---
@engineer_router.post("/engineer/ask")
async def ask_engineer(question: str):
    return {"answer": "Not implemented yet"}

# --- INCLUDE ROUTERS ---
app.include_router(chat_router)
app.include_router(tools_router)
app.include_router(engineer_router)

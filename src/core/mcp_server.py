# src/core/mcp_server.py
#
# WHAT IS MCP?
# MCP (Model Context Protocol) is a standard way for an AI app (the "host",
# e.g. Claude Desktop) to talk to a separate program (the "server") that
# exposes tools/data. The host launches a "client" internally, the client
# talks to our server, and our server tells it: "here are the tools you can
# call, and here's what happens when you call them."
#
# This file IS the server. It does NOT connect out to anything by itself —
# it just sits and waits to be asked questions by whatever MCP client
# launches it (Claude Desktop, or our own test client in
# scripts/test_mcp_client.py).
#
# TRANSPORT: we use "stdio" transport, meaning the client and server talk to
# each other over stdin/stdout of a subprocess, not over the network. That's
# why there's no "port" anywhere in this file.

# --- Standard library imports ---
import os              # lets us read environment variables (like our API key)
import sys             # used below to fix a Windows console encoding quirk
import time            # lets us measure time, used for the rate limiter

# stdout is the actual MCP protocol channel for this server (see STEP 5 below),
# so we must never touch its encoding. Debug prints elsewhere in the codebase
# go to stderr instead — this line just makes sure stderr can safely print
# emoji on Windows, where the default console encoding sometimes can't.
sys.stderr.reconfigure(encoding="utf-8")

# --- Third-party imports ---
from dotenv import load_dotenv          # reads the .env file into environment variables
from mcp.server.fastmcp import FastMCP  # the high-level MCP server class we build on

# --- Our own project code ---
# We reuse the SAME LLMFactory class that main.py's FastAPI app uses, so the
# MCP server searches/writes to the exact same document library.
from src.core.factory import LLMFactory

# Load variables from .env (like MCP_API_KEY) into os.environ so os.getenv() can see them.
load_dotenv()

# ==========================================
# STEP 1: CREATE THE SERVER
# ==========================================
# FastMCP is a helper class that hides most of the raw MCP protocol plumbing.
# We just give it a name, and then use @mcp.tool() below to register functions.
mcp = FastMCP(name="rag-platform")

# One shared LLMFactory instance for this whole server process. It loads the
# embedding model and opens the Chroma database ONCE when the server starts,
# instead of once per tool call (loading a model is slow — do it once).
factory = LLMFactory()

# ==========================================
# STEP 2: A TINY "AUTH" LAYER
# ==========================================
# Real HTTP-based MCP servers get an API key from a request header. Our
# server uses stdio (no HTTP requests, no headers), so instead we ask every
# tool call to include an `api_key` argument and we check it here.
#
# EXPECTED_API_KEY is read once at startup from the .env file.
EXPECTED_API_KEY = os.getenv("MCP_API_KEY")


def check_api_key(api_key: str) -> None:
    """
    Raise an error if the caller didn't send the right key.
    Every tool function below calls this FIRST, before doing any real work,
    so an unauthenticated caller can never reach our data.
    """
    if not EXPECTED_API_KEY:
        # If the server operator never set MCP_API_KEY, fail loudly instead
        # of silently allowing everyone in — a missing key is a config bug,
        # not "no auth needed".
        raise ValueError("Server misconfigured: MCP_API_KEY is not set in .env")

    if api_key != EXPECTED_API_KEY:
        raise ValueError("Unauthorized: invalid api_key")


# ==========================================
# STEP 3: A TINY RATE LIMITER
# ==========================================
# Goal: stop one caller from hammering a tool hundreds of times per second.
# We keep, per tool name, a list of the timestamps of recent calls. Before
# doing work, we throw away timestamps older than the time window and check
# how many calls are left in that window.
MAX_CALLS_PER_WINDOW = 10   # how many calls...
RATE_LIMIT_WINDOW_SECONDS = 60  # ...allowed per this many seconds

# call_history looks like: {"search_docs": [1719999000.1, 1719999001.4, ...]}
call_history: dict[str, list[float]] = {}


def check_rate_limit(tool_name: str) -> None:
    """Raise an error if `tool_name` has been called too many times recently."""
    now = time.time()

    # Get this tool's past call timestamps (or an empty list if it's the first call).
    timestamps = call_history.get(tool_name, [])

    # Keep only timestamps that fall inside our rolling time window.
    recent_timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]

    if len(recent_timestamps) >= MAX_CALLS_PER_WINDOW:
        raise ValueError(
            f"Rate limit exceeded for '{tool_name}': "
            f"max {MAX_CALLS_PER_WINDOW} calls per {RATE_LIMIT_WINDOW_SECONDS}s"
        )

    # Record this call and save the trimmed list back.
    recent_timestamps.append(now)
    call_history[tool_name] = recent_timestamps


def authorize(api_key: str, tool_name: str) -> None:
    """Runs both guards every tool needs, in the cheap-first order: auth, then rate limit."""
    # Auth check first — never touch the database for an unauthorized caller.
    check_api_key(api_key)
    # Then the rate limit check — cheap, so it's fine to do it right after auth
    # but before any real (slower) work like a DB query or file read.
    check_rate_limit(tool_name)


# ==========================================
# STEP 4: THE ACTUAL TOOLS
# ==========================================
# @mcp.tool() registers the function below as something the AI host can call.
# FastMCP reads the function's type hints and docstring to automatically
# build the tool's schema (name, parameters, description) — we don't have to
# write that JSON schema by hand like we did in tools.py.

@mcp.tool()
def search_docs(query: str, api_key: str) -> list[dict]:
    """
    Search the indexed document library for text relevant to `query`.

    Args:
        query: the natural-language question or topic to search for.
        api_key: shared secret proving the caller is allowed to use this server.

    Returns:
        A list of {"id": ..., "text": ...} matches, most relevant first.
    """
    # Runs check_api_key() + check_rate_limit() together — every tool needs
    # both, so this one call replaces what used to be two lines here.
    authorize(api_key, "search_docs")

    # Delegate to the method we added on LLMFactory: embed the query, ask
    # Chroma for the nearest neighbours, return them as plain dicts.
    return factory.search_library(query)


@mcp.tool()
def add_document(path: str, api_key: str) -> str:
    """
    Read a local text file and index its contents into the document library
    so future search_docs() calls can find it.

    Args:
        path: filesystem path to a .txt file to ingest.
        api_key: shared secret proving the caller is allowed to use this server.

    Returns:
        A confirmation message, or an error message if the file couldn't be read.
    """
    # Same auth + rate-limit pair as search_docs, just tagged with this tool's name
    # so its call count is tracked separately in call_history.
    authorize(api_key, "add_document")

    # Guard clause: fail with a clear message instead of crashing if the
    # path doesn't exist (a very common mistake when a beginner tests this).
    if not os.path.isfile(path):
        return f"Error: no file found at '{path}'"

    # Read the whole file as plain text.
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Use the file name (without extension) as the document's unique id,
    # e.g. "runbook.txt" -> "runbook".
    doc_id = os.path.splitext(os.path.basename(path))[0]

    # This calls the SAME add_to_library() method main.py's /engineer/index
    # route uses — one shared code path for indexing documents, whether the
    # request comes from the REST API or from an MCP client.
    return factory.add_to_library(text, doc_id, metadata={"source": path})


@mcp.tool()
def get_metrics(api_key: str) -> dict:
    """
    Report basic health/usage stats about this MCP server.

    Args:
        api_key: shared secret proving the caller is allowed to use this server.

    Returns:
        A dict with the document count in the library and recent call counts
        per tool (handy for spotting who's close to the rate limit).
    """
    # Same guard as the other tools; tracked in call_history under "get_metrics".
    authorize(api_key, "get_metrics")

    # collection.count() is a built-in Chroma method: how many documents are
    # currently stored in the "engineer_docs" collection.
    doc_count = factory.collection.count()

    # Report how many calls each tool has made inside the current rate-limit
    # window, so an operator can see usage at a glance.
    recent_call_counts = {name: len(times) for name, times in call_history.items()}

    return {
        "documents_indexed": doc_count,
        "recent_call_counts": recent_call_counts,
        "mcp_spec_version": "2025-11-25",
    }


# ==========================================
# STEP 5: RUN THE SERVER
# ==========================================
# This block only runs when the file is executed directly, e.g.
#   python -m src.core.mcp_server
# It does NOT run when this file is merely imported by something else.
if __name__ == "__main__":
    # mcp.run() starts the stdio event loop: it reads MCP requests from
    # stdin, dispatches them to the @mcp.tool() functions above, and writes
    # responses to stdout. It blocks forever until the process is killed —
    # that's expected, this IS the server.
    mcp.run(transport="stdio")

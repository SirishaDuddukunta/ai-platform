# WHAT IS THIS FILE?
# This is an MCP *client* — the opposite role from src/core/mcp_server.py.
# A client's job is to launch a server, ask it "what tools do you have?",
# and then call those tools. In real life, Claude Desktop plays this role
# for you. This script exists so we can test our server WITHOUT needing
# Claude Desktop open — just run this file from the terminal.

import asyncio  # MCP communication is asynchronous, so we need asyncio to run it
import os       # used to read our own local .env for the api_key in this test
import sys      # sys.executable = path to the exact Python we're running right now

# Windows terminals sometimes default to an old encoding (cp1252) that can't
# print emoji like checkmarks. Force stdout to UTF-8 so our print()s below
# don't crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters  # client-side building blocks
from mcp.client.stdio import stdio_client               # launches a server over stdio

# Load MCP_API_KEY from .env so this test client can authenticate,
# exactly like a real caller of our server would have to.
load_dotenv()
API_KEY = os.getenv("MCP_API_KEY")


async def run():
    # StdioServerParameters describes HOW to launch the server: which
    # command to run, what arguments to pass, and which folder to run it
    # from. This is exactly the command you'd type in a terminal to start
    # the server yourself.
    #
    # We use sys.executable (the exact Python interpreter running THIS
    # script) instead of the plain word "python", so the server subprocess
    # uses the same virtual environment — and therefore has mcp, chromadb,
    # etc. already installed.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.core.mcp_server"],
        cwd=project_root,
    )

    # stdio_client starts the server as a subprocess and gives us back
    # `read`/`write` streams connected to its stdin/stdout.
    async with stdio_client(server_params) as (read, write):
        # ClientSession wraps those raw streams with the actual MCP protocol
        # logic (handshakes, request/response matching, etc).
        async with ClientSession(read, write) as session:
            # Every MCP session must be initialized before you can do anything else —
            # this is the client/server handshake.
            await session.initialize()

            # Ask the server what tools it exposes. This should show
            # search_docs, add_document, and get_metrics.
            tools = await session.list_tools()
            print(f"✅ Connected! Available tools: {[t.name for t in tools.tools]}")

            # --- Try calling get_metrics() ---
            # call_tool() takes the tool's name and a dict of arguments,
            # matching the parameters we defined with @mcp.tool() in the server.
            metrics_result = await session.call_tool("get_metrics", {"api_key": API_KEY})
            print(f"\n📊 get_metrics() ->")
            for content_block in metrics_result.content:
                print(content_block.text)

            # --- Try calling search_docs() ---
            # This will return an empty list until you've added at least one
            # document with add_document(), which is expected on a fresh DB.
            search_result = await session.call_tool(
                "search_docs", {"query": "database optimization", "api_key": API_KEY}
            )
            print(f"\n🔍 search_docs('database optimization') ->")
            for content_block in search_result.content:
                print(content_block.text)


# This only runs when you execute this file directly:
#   python scripts/test_mcp_client.py
if __name__ == "__main__":
    asyncio.run(run())

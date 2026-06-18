from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    # Tell the client how to launch your server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.core.mcp_server"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List the tools available on your server
            tools = await session.list_tools()
            print(f"✅ Connected! Available tools: {tools}")

import asyncio
asyncio.run(run())
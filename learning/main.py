import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def main():
    print("Hello from learn-ai!")
    """Connect to an MCP server running with SSE transport"""
    # Store the context managers so they stay alive
    _streams_context = sse_client(url="http://localhost:8080/sse")
    streams = await _streams_context.__aenter__()

    _session_context = ClientSession(*streams)
    session: ClientSession = await _session_context.__aenter__()

    # Initialize
    await session.initialize()

    # List available tools to verify connection
    print("Initialized SSE client...")
    print("Listing tools...")
    response = await session.list_tools()
    tools = response.tools
    print("\nConnected to server with tools:", [tool.name for tool in tools])


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Simple MCP client example with OAuth authentication support.

This client connects to an MCP server using streamable HTTP transport with OAuth.
"""

import asyncio
import os
import threading
import time
import webbrowser
import logging
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InMemoryTokenStorage(TokenStorage):
    """Simple in-memory token storage implementation."""

    def __init__(self):
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


class CallbackHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler to capture OAuth callback."""

    def __init__(self, request, client_address, server, callback_data):
        """Initialize with callback data storage."""
        self.callback_data = callback_data
        super().__init__(request, client_address, server)

    def do_GET(self):
        """Handle GET request from OAuth redirect."""
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)

        if "code" in query_params:
            self.callback_data["authorization_code"] = query_params["code"][0]
            self.callback_data["state"] = query_params.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
            <html>
            <body>
                <h1>Authorization Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                <script>setTimeout(() => window.close(), 2000);</script>
            </body>
            </html>
            """
            )
        elif "error" in query_params:
            self.callback_data["error"] = query_params["error"][0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"""
            <html>
            <body>
                <h1>Authorization Failed</h1>
                <p>Error: {query_params["error"][0]}</p>
                <p>You can close this window and return to the terminal.</p>
            </body>
            </html>
            """.encode()
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class CallbackServer:
    """Simple server to handle OAuth callbacks."""

    def __init__(self, port=3030):
        self.port = port
        self.server = None
        self.thread = None
        self.callback_data = {"authorization_code": None, "state": None, "error": None}

    def _create_handler_with_data(self):
        """Create a handler class with access to callback data."""
        callback_data = self.callback_data

        class DataCallbackHandler(CallbackHandler):
            def __init__(self, request, client_address, server):
                super().__init__(request, client_address, server, callback_data)

        return DataCallbackHandler

    def start(self):
        """Start the callback server in a background thread."""
        try:
            handler_class = self._create_handler_with_data()
            self.server = HTTPServer(("localhost", self.port), handler_class)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            logger.info(f"Started callback server on http://localhost:{self.port}")
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {self.port} already in use, trying next port")
                self.port += 1
                self.start()
            else:
                raise

    def stop(self):
        """Stop the callback server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    def wait_for_callback(self, timeout=300):
        """Wait for OAuth callback with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.callback_data["authorization_code"]:
                return self.callback_data["authorization_code"]
            elif self.callback_data["error"]:
                raise Exception(f"OAuth error: {self.callback_data['error']}")
            time.sleep(0.1)
        raise Exception("Timeout waiting for OAuth callback")

    def get_state(self):
        """Get the received state parameter."""
        return self.callback_data["state"]


class SimpleAuthClient:
    """Simple MCP client with auth support."""

    def __init__(self, server_url: str, transport_type: str = "streamable_http"):
        self.server_url = server_url
        self.transport_type = transport_type
        self.session: Optional[ClientSession] = None
        self.is_connected = False

    async def connect(self, enable_interactive: bool = True):
        """Connect to the MCP server."""
        logger.info(f"Attempting to connect to {self.server_url}")

        try:
            if self.transport_type in ["streamable_http", "sse"]:
                # OAuth-based transports
                await self._connect_with_oauth(enable_interactive)
            else:
                # STDIO transport
                await self._connect_stdio(enable_interactive)
            
            self.is_connected = True
            logger.info(f"Successfully connected to {self.server_url}")
                
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.is_connected = False
            raise

    async def _connect_with_oauth(self, enable_interactive: bool):
        """Connect using OAuth authentication."""
        callback_server = CallbackServer(port=3030)
        callback_server.start()

        try:
            async def callback_handler() -> tuple[str, str | None]:
                """Wait for OAuth callback and return auth code and state."""
                logger.info("Waiting for authorization callback...")
                auth_code = callback_server.wait_for_callback(timeout=300)
                return auth_code, callback_server.get_state()

            client_metadata_dict = {
                "client_name": "Simple Auth Client",
                "redirect_uris": [f"http://localhost:{callback_server.port}/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "client_secret_post",
            }

            async def _default_redirect_handler(authorization_url: str) -> None:
                """Default redirect handler that opens the URL in a browser."""
                logger.info(f"Opening browser for authorization: {authorization_url}")
                webbrowser.open(authorization_url)

            # Create OAuth authentication handler
            oauth_auth = OAuthClientProvider(
                server_url=self.server_url.replace("/mcp", ""),
                client_metadata=OAuthClientMetadata.model_validate(client_metadata_dict),
                storage=InMemoryTokenStorage(),
                redirect_handler=_default_redirect_handler,
                callback_handler=callback_handler,
            )

            # Create transport based on type
            if self.transport_type == "sse":
                async with sse_client(
                    url=self.server_url,
                    auth=oauth_auth,
                    timeout=60,
                ) as (read_stream, write_stream):
                    await self._run_session(read_stream, write_stream, None, enable_interactive)
            else:
                async with streamablehttp_client(
                    url=self.server_url,
                    auth=oauth_auth,
                    timeout=timedelta(seconds=60),
                ) as (read_stream, write_stream, get_session_id):
                    await self._run_session(read_stream, write_stream, get_session_id, enable_interactive)

        finally:
            callback_server.stop()

    async def _connect_stdio(self, enable_interactive: bool):
        """Connect using STDIO transport."""
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", self.server_url, "stdio"],
            env={"UV_INDEX": os.environ.get("UV_INDEX", "")},
        )
        
        async with stdio_client(server=server_params) as (read_stream, write_stream):
            await self._run_session(read_stream, write_stream, None, enable_interactive)

    async def _run_session(self, read_stream, write_stream, get_session_id, enable_interactive: bool = True):
        """Run the MCP session with the given streams."""
        logger.info("Initializing MCP session...")
        
        async with ClientSession(read_stream, write_stream) as session:
            self.session = session
            logger.info("Starting session initialization...")
            await session.initialize()
            logger.info("Session initialization complete!")

            if get_session_id:
                session_id = get_session_id()
                if session_id:
                    logger.info(f"Session ID: {session_id}")
                    
            if enable_interactive:
                await self.interactive_loop()

    async def list_tools(self):
        """List available tools from the server."""
        if not self.session:
            logger.error("Not connected to server")
            return

        try:
            result = await self.session.list_tools()
            if hasattr(result, "tools") and result.tools:
                logger.info("Available tools:")
                for i, tool in enumerate(result.tools, 1):
                    logger.info(f"{i}. {tool.name}")
                    if tool.description:
                        logger.info(f"   Description: {tool.description}")
            else:
                logger.info("No tools available")
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None):
        """Call a specific tool."""
        if not self.session:
            logger.error("Not connected to server")
            return

        try:
            result = await self.session.call_tool(tool_name, arguments or {})
            logger.info(f"Tool '{tool_name}' result:")
            if hasattr(result, "content"):
                for content in result.content:
                    if content.type == "text":
                        logger.info(content.text)
                    else:
                        logger.info(str(content))
            else:
                logger.info(str(result))
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")

    async def interactive_loop(self):
        """Run interactive command loop."""
        print("\n🎯 Interactive MCP Client")
        print("Commands:")
        print("  list - List available tools")
        print("  call <tool_name> [args] - Call a tool")
        print("  quit - Exit the client")
        print()

        while True:
            try:
                command = input("mcp> ").strip()

                if not command:
                    continue

                if command == "quit":
                    break

                elif command == "list":
                    await self.list_tools()

                elif command.startswith("call "):
                    parts = command.split(maxsplit=2)
                    tool_name = parts[1] if len(parts) > 1 else ""

                    if not tool_name:
                        print("❌ Please specify a tool name")
                        continue

                    # Parse arguments (simple JSON-like format)
                    arguments = {}
                    if len(parts) > 2:
                        import json
                        try:
                            arguments = json.loads(parts[2])
                        except json.JSONDecodeError:
                            print("❌ Invalid arguments format (expected JSON)")
                            continue

                    await self.call_tool(tool_name, arguments)

                else:
                    print("❌ Unknown command. Try 'list', 'call <tool_name>', or 'quit'")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                break

    # API methods for FastAPI integration
    async def connect_async(self, enable_interactive: bool = False) -> bool:
        """Async version of connect for API use"""
        try:
            await self.connect(enable_interactive)
            return self.is_connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def list_tools_async(self) -> list[dict[str, str]]:
        """Get tools list for API"""
        if not self.session:
            return []
        
        try:
            result = await self.session.list_tools()
            if hasattr(result, "tools") and result.tools:
                return [
                    {
                        "name": tool.name, 
                        "description": tool.description or "No description available"
                    }
                    for tool in result.tools
                ]
            return []
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

    async def call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call tool for API"""
        if not self.session:
            raise Exception("Not connected to server")
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            if hasattr(result, "content") and result.content:
                return [
                    {
                        "type": content.type,
                        "text": getattr(content, 'text', str(content))
                    }
                    for content in result.content
                ]
            else:
                return {"result": str(result)}
                
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the server"""
        if self.session:
            # The session will be closed by the context manager
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from server")


async def main():
    """Main entry point."""
    server_url = os.getenv(
        "MCP_SERVER_URL",
        "/Users/rajubhimani/code/learn_ai/mcp_server/time/src/mcp_server_time/server.py"
    )
    transport_type = os.getenv("MCP_TRANSPORT_TYPE", "stdio")

    # Adjust URL based on transport type
    if transport_type == "stdio":
        # server_url is the path to the server script
        pass
    elif transport_type == "streamable_http":
        port = os.getenv("MCP_SERVER_PORT", "8000")
        server_url = f"http://localhost:{port}/mcp"
    elif transport_type == "sse":
        port = os.getenv("MCP_SERVER_PORT", "8000")
        server_url = f"http://localhost:{port}/sse"
    else:
        raise ValueError(
            f"Invalid transport type '{transport_type}'. "
            "Supported types: streamable_http, stdio, sse"
        )

    print("🚀 Simple MCP Auth Client")
    print(f"Connecting to: {server_url}")
    print(f"Transport type: {transport_type}")

    client = SimpleAuthClient(server_url, transport_type)
    await client.connect()
    print("\n👋 Goodbye!")


def cli():
    """CLI entry point for uv script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
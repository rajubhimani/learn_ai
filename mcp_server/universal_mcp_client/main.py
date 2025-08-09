#!/usr/bin/env python3
"""
FastAPI backend for MCP Server Manager UI
Wraps the MCP client functionality in REST endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
from datetime import datetime

# Import your existing MCP client class
from src.client import SimpleAuthClient  # Adjust import path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Server Manager API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ServerConfig(BaseModel):
    name: str
    url: str
    transport: str = "streamable_http"

class ServerResponse(BaseModel):
    id: str
    name: str
    url: str
    transport: str
    status: str
    tools: List[Dict[str, Any]] = []
    error: Optional[str] = None

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}

class ToolCallResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str

# In-memory storage (use database in production)
servers: Dict[str, Dict] = {}
mcp_clients: Dict[str, SimpleAuthClient] = {}

@app.post("/api/servers", response_model=ServerResponse)
async def add_server(server_config: ServerConfig):
    """Add a new MCP server"""
    server_id = str(len(servers) + 1)
    
    server = {
        "id": server_id,
        "name": server_config.name,
        "url": server_config.url,
        "transport": server_config.transport,
        "status": "disconnected",
        "tools": [],
        "created_at": datetime.now().isoformat()
    }
    
    servers[server_id] = server
    logger.info(f"Added server {server_id}: {server_config.name}")
    return ServerResponse(**server)

@app.get("/api/servers", response_model=List[ServerResponse])
async def list_servers():
    """List all configured servers"""
    return [ServerResponse(**server) for server in servers.values()]

@app.delete("/api/servers/{server_id}")
async def remove_server(server_id: str):
    """Remove a server"""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # Disconnect if connected
    if server_id in mcp_clients:
        try:
            await disconnect_client(server_id)
        except Exception as e:
            logger.error(f"Error disconnecting server {server_id}: {e}")
    
    del servers[server_id]
    logger.info(f"Removed server {server_id}")
    return {"message": "Server removed successfully"}

@app.post("/api/servers/{server_id}/connect")
async def connect_server(server_id: str, background_tasks: BackgroundTasks):
    """Connect to an MCP server"""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    server = servers[server_id]
    
    # Check if already connected
    if server["status"] == "connected":
        return {"message": "Already connected", "status": "connected"}
    
    # Update status to connecting
    server["status"] = "connecting"
    server.pop("error", None)  # Clear any previous errors
    
    # Start connection in background
    background_tasks.add_task(perform_connection, server_id)
    
    logger.info(f"Starting connection to server {server_id}")
    return {"message": "Connection started", "status": "connecting"}

async def perform_connection(server_id: str):
    """Perform the actual connection (runs in background)"""
    server = servers[server_id]
    
    try:
        logger.info(f"Connecting to server {server_id} at {server['url']}")
        
        # Create MCP client instance
        client = SimpleAuthClient(
            server_url=server["url"],
            transport_type=server["transport"]
        )
        
        # Attempt connection
        success = await client.connect_async()
        
        if not success:
            raise Exception("Failed to establish connection")

        # Get tools
        tools = await client.list_tools_async()
        
        # Update server status
        server["status"] = "connected"
        server["tools"] = tools
        server.pop("error", None)  # Clear any errors
        
        # Store client instance
        mcp_clients[server_id] = client
        
        logger.info(f"Successfully connected to server {server_id}")
        
    except Exception as e:
        logger.error(f"Failed to connect to server {server_id}: {e}")
        # Update status to error
        server["status"] = "error"
        server["error"] = str(e)
        server["tools"] = []
        
        # Clean up any partial client
        if server_id in mcp_clients:
            del mcp_clients[server_id]

async def disconnect_client(server_id: str):
    """Helper to disconnect a client with cleanup"""
    if server_id in mcp_clients:
        client = mcp_clients[server_id]
        try:
            # Add any cleanup logic here if your client has disconnect method
            if hasattr(client, 'disconnect'):
                await client.disconnect()
        except Exception as e:
            logger.error(f"Error during client cleanup for {server_id}: {e}")
        finally:
            del mcp_clients[server_id]

@app.post("/api/servers/{server_id}/disconnect")
async def disconnect_server(server_id: str):
    """Disconnect from an MCP server"""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # Clean up client
    await disconnect_client(server_id)
    
    # Update server status
    servers[server_id]["status"] = "disconnected"
    servers[server_id]["tools"] = []
    servers[server_id].pop("error", None)
    
    logger.info(f"Disconnected from server {server_id}")
    return {"message": "Server disconnected"}

@app.get("/api/servers/{server_id}/tools")
async def get_server_tools(server_id: str):
    """Get tools for a specific server"""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    server = servers[server_id]
    if server["status"] != "connected":
        raise HTTPException(status_code=400, detail="Server not connected")
    
    return {"tools": server["tools"]}

@app.post("/api/servers/{server_id}/tools/call", response_model=ToolCallResponse)
async def call_tool(server_id: str, tool_call: ToolCall):
    """Call a tool on a specific server"""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    if server_id not in mcp_clients:
        raise HTTPException(status_code=400, detail="Server not connected")
    
    client = mcp_clients[server_id]
    
    try:
        logger.info(f"Calling tool {tool_call.tool_name} on server {server_id}")
        
        # Call the tool
        result = await client.call_tool_async(
            tool_call.tool_name, 
            tool_call.arguments
        )
        
        return ToolCallResponse(
            success=True,
            result=result,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error calling tool {tool_call.tool_name} on server {server_id}: {e}")
        return ToolCallResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@app.get("/api/servers/{server_id}/status")
async def get_server_status(server_id: str):
    """Get current status of a server"""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    server = servers[server_id]
    return {
        "status": server["status"],
        "error": server.get("error"),
        "tool_count": len(server.get("tools", []))
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "servers_count": len(servers),
        "connected_count": sum(1 for s in servers.values() if s["status"] == "connected")
    }

# Graceful shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    logger.info("Shutting down, cleaning up connections...")
    for server_id in list(mcp_clients.keys()):
        try:
            await disconnect_client(server_id)
        except Exception as e:
            logger.error(f"Error during shutdown cleanup for {server_id}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
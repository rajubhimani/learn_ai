import asyncio
import os
from datetime import timedelta
import litellm
from litellm.experimental_mcp_client import load_mcp_tools
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import StdioServerParameters, stdio_client
import json
from typing import List

# Global variable to store active sessions for tool execution
active_sessions = {}

async def gather_all_tools(mcp_urls: List[str]):
    """Gather tools from multiple MCP servers and keep sessions active"""
    print(f"Gathering tools from MCP servers: {mcp_urls}")
    tool_lists = []
    
    for mcp_url in mcp_urls:
        try:
            if mcp_url.startswith(('http://', 'https://')) and mcp_url.endswith('/sse'):
                # SSE-based MCP server
                print(f"Connecting to SSE MCP server: {mcp_url}")
                # Keep the session active by storing it
                session_data = await create_sse_session(mcp_url)
            elif mcp_url.startswith(('http://', 'https://')) and not mcp_url.endswith('/sse'):
                # HTTP-based MCP server
                print(f"Connecting to HTTP MCP server: {mcp_url}")
                session_data = await create_http_session(mcp_url)
            else:
                # Local stdio-based MCP servers
                print(f"Attempting stdio connection to: {mcp_url}")
                session_data = await create_stdio_session(mcp_url)
            if session_data:
                active_sessions[mcp_url] = session_data
                tools = session_data['tools']
                if tools:
                    tool_lists.append(tools)
                    print(f"Loaded {len(tools)} tools from {mcp_url}")
            else:
                print(f"Unknown MCP server type or failed to connect: {mcp_url}")
                continue
                
        except Exception as e:
            print(f"Failed to connect to {mcp_url}: {str(e)}")
            # Print more detailed error info for debugging
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            continue
    
    # Flatten all tools into one list
    all_tools = [tool for tools in tool_lists for tool in tools]
    return all_tools

async def create_sse_session(mcp_url: str):
    """Create and maintain an SSE session"""
    try:
        # Create connection context managers
        sse_context = sse_client(url=mcp_url, timeout=60)
        read, write = await sse_context.__aenter__()
        
        session_context = ClientSession(read, write)
        session = await session_context.__aenter__()
        
        # Initialize the session
        result = await session.initialize()
        print(f"Session initialized successfully for {mcp_url}")
        
        # Load tools using the session
        tools = await load_mcp_tools(session=session, format="openai")
        
        return {
            'session': session,
            'tools': tools,
            'sse_context': sse_context,
            'session_context': session_context
        }
    except Exception as e:
        print(f"Failed to create SSE session for {mcp_url}: {e}")
        return None

async def create_http_session(mcp_url: str):
    """Create and maintain an HTTP session"""
    try:
        # Create connection context managers
        http_context = streamablehttp_client(url=mcp_url, timeout=60)
        read, write, _ = await http_context.__aenter__()
        
        session_context = ClientSession(read, write)
        session = await session_context.__aenter__()
        
        # Initialize the session
        result = await session.initialize()
        print(f"Session initialized successfully for {mcp_url}")
        
        # Load tools using the session
        tools = await load_mcp_tools(session=session, format="openai")
        
        return {
            'session': session,
            'tools': tools,
            'http_context': http_context,
            'session_context': session_context
        }
    except Exception as e:
        print(f"Failed to create HTTP session for {mcp_url}: {e}")
        return None

async def create_stdio_session(mcp_url: str):
    """Create and maintain a stdio session"""
    try:
        # Assume it's a command to run a local MCP server
        server_params = StdioServerParameters(
            command=mcp_url,
            args=[]
        )
        # Create connection context managers
        stdio_context = stdio_client(server_params)
        read, write = await stdio_context.__aenter__()
        
        session_context = ClientSession(read, write)
        session = await session_context.__aenter__()
        
        # Initialize the session
        result = await session.initialize()
        print(f"Session initialized successfully for {mcp_url}")
        
        # Load tools using the session
        tools = await load_mcp_tools(session=session, format="openai")
        
        return {
            'session': session,
            'tools': tools,
            'stdio_context': stdio_context,
            'session_context': session_context
        }
    except Exception as stdio_error:
        print(f"Failed to connect via stdio to {mcp_url}: {stdio_error}")
        return None

async def execute_tool_call(tool_call, tools):
    """Execute a single tool call using the appropriate MCP session"""
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    
    print(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
    
    # Find which session has this tool
    for mcp_url, session_data in active_sessions.items():
        session = session_data['session']
        session_tools = session_data['tools']
        
        # Check if this tool exists in this session
        tool_exists = any(
            tool.get('function', {}).get('name') == tool_name 
            for tool in session_tools
        )
        
        if tool_exists:
            try:
                # Execute the tool call
                result = await session.call_tool(tool_name, tool_args)
                print(f"✅ Tool {tool_name} executed successfully")
                print(f"🔍 Raw result type: {type(result)}")
                print(f"🔍 Raw result: {result}")
                
                # Extract content from MCP result
                content = extract_mcp_content(result)
                
                return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": content
                }
            except Exception as e:
                print(f"❌ Error executing tool {tool_name}: {e}")
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
                return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Error: {str(e)}"
                }
    
    # Tool not found in any session
    print(f"❌ Tool {tool_name} not found in any active session")
    return {
        "tool_call_id": tool_call.id,
        "role": "tool",
        "name": tool_name,
        "content": f"Error: Tool {tool_name} not found"
    }

def extract_mcp_content(result):
    """Extract content from MCP result object"""
    try:
        # If result has content attribute
        if hasattr(result, 'content'):
            content = result.content
            
            # If content is a list, extract from each item
            if isinstance(content, list):
                extracted_parts = []
                for item in content:
                    if hasattr(item, 'text'):
                        # TextContent object
                        extracted_parts.append(item.text)
                    elif hasattr(item, 'data'):
                        # ImageContent or other content types
                        extracted_parts.append(str(item.data))
                    elif isinstance(item, dict):
                        # Dictionary content
                        extracted_parts.append(json.dumps(item))
                    else:
                        # Fallback to string representation
                        extracted_parts.append(str(item))
                return "\n".join(extracted_parts) if extracted_parts else "No content"
            
            # If content is a single object
            elif hasattr(content, 'text'):
                return content.text
            elif hasattr(content, 'data'):
                return str(content.data)
            elif isinstance(content, (dict, list)):
                return json.dumps(content)
            else:
                return str(content)
        
        # If result is directly the content
        elif hasattr(result, 'text'):
            return result.text
        elif isinstance(result, (dict, list)):
            return json.dumps(result)
        else:
            return str(result)
            
    except Exception as e:
        print(f"❌ Error extracting content: {e}")
        return f"Error extracting content: {str(e)}"

async def chat_with_tools(messages, tools, max_iterations=1):
    """Handle a complete conversation with tool calling"""
    conversation = messages.copy()
    
    for iteration in range(max_iterations):
        # Don't print iteration info in interactive mode to keep it clean
        
        try:
            response = await litellm.acompletion(
                model="ollama/llama3.1",
                messages=conversation,
                api_base="http://localhost:11434",
                tools=tools if tools else None,
                stream=False
            )
            
            if not (hasattr(response, 'choices') and response.choices):
                print("❌ Unexpected response format:", response)
                break
                
            message = response.choices[0].message
            
            # Add the assistant's message to conversation
            assistant_msg = {
                "role": "assistant",
                "content": message.content
            }
            if hasattr(message, 'tool_calls') and message.tool_calls:
                assistant_msg["tool_calls"] = message.tool_calls
            
            conversation.append(assistant_msg)
            
            # Display the response
            if message.content:
                print(f"🤖 Assistant: {message.content}")
            
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"🔧 Using {len(message.tool_calls)} tool(s)...")
                
                # Execute all tool calls
                for tool_call in message.tool_calls:
                    result = await execute_tool_call(tool_call, tools)
                    conversation.append(result)
                    
                    # Show tool execution (optional, can be commented out for cleaner output)
                    tool_name = result.get('name', 'unknown')
                    tool_content = result.get('content', '')
                    if len(tool_content) > 100:
                        tool_content = tool_content[:100] + "..."
                    print(f"  ⚙️ {tool_name}: {tool_content}")
                
                # Continue the conversation with tool results
                continue
            else:
                # No more tool calls, conversation completed
                break
                
        except Exception as e:
            print(f"❌ Error in completion: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            break
    
    return conversation

async def cleanup_sessions():
    """Clean up active sessions"""
    for mcp_url, session_data in active_sessions.items():
        try:
            if 'session_context' in session_data:
                await session_data['session_context'].__aexit__(None, None, None)
            if 'sse_context' in session_data:
                await session_data['sse_context'].__aexit__(None, None, None)
        except Exception as e:
            print(f"Error cleaning up session for {mcp_url}: {e}")

async def interactive_chat():
    """Interactive chat session with MCP tools"""
    print("\n💬 Interactive Chat Mode")
    print("=" * 40)
    print("Type your messages and press Enter.")
    print("Use '/quit' or '/bye' to exit.")
    print("Use '/help' to see available tools.")
    print("=" * 40)
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for quit commands
            if user_input.lower() in ['/quit', '/bye', '/exit']:
                print("👋 Goodbye!")
                break
            
            # Check for help command
            if user_input.lower() == '/help':
                print("\n🔧 Available tools:")
                if not active_sessions:
                    print("  No tools loaded")
                else:
                    tool_count = 0
                    for session_data in active_sessions.values():
                        for tool in session_data['tools']:
                            tool_count += 1
                            tool_name = tool.get('function', {}).get('name', 'Unknown')
                            tool_desc = tool.get('function', {}).get('description', 'No description')
                            print(f"  {tool_count}. {tool_name}: {tool_desc}")
                print("\nCommands:")
                print("  /help - Show this help")
                print("  /quit, /bye, /exit - Exit chat")
                print("  /clear - Clear conversation history")
                continue
            
            # Check for clear command
            if user_input.lower() == '/clear':
                conversation_history = []
                print("🧹 Conversation history cleared!")
                continue
            
            # Skip empty messages
            if not user_input:
                continue
            
            # Add user message to conversation
            user_message = {"role": "user", "content": user_input}
            current_conversation = conversation_history + [user_message]
            
            # Get all available tools
            all_tools = []
            for session_data in active_sessions.values():
                all_tools.extend(session_data['tools'])
            
            print(f"\n🤖 Assistant is thinking... (with {len(all_tools)} tools available)")
            
            # Process the conversation with tools
            updated_conversation = await chat_with_tools(current_conversation, all_tools)
            
            # Update conversation history
            conversation_history = updated_conversation
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")

async def main():
    """Main function to demonstrate MCP tool loading and usage"""
    
    # Configure your MCP servers here
    mcp_urls = [
        "http://localhost:8080/sse",                    # SSE-based MCP server
        # "/path/to/local/mcp/server",              # Local executable MCP server
        # "python -m some_mcp_server",              # Python module MCP server
    ]
    
    print("=" * 50)
    print("MCP Interactive Chat Demo")
    print("=" * 50)
    
    try:
        print("Loading tools from MCP servers...")
        tools = await gather_all_tools(mcp_urls)
        
        if not tools:
            print("\n⚠️  No tools loaded from MCP servers")
            print("Proceeding without MCP tools...")
        else:
            print(f"\n✅ Successfully loaded {len(tools)} tools from MCP servers")
            print("\nAvailable tools:")
            for i, tool in enumerate(tools):
                tool_name = tool.get('function', {}).get('name', 'Unknown')
                tool_desc = tool.get('function', {}).get('description', 'No description')
                print(f"  {i+1}. {tool_name}: {tool_desc[:60]}{'...' if len(tool_desc) > 60 else ''}")
        
        # Start interactive chat
        await interactive_chat()
        
    finally:
        # Clean up sessions
        print("\n🧹 Cleaning up sessions...")
        await cleanup_sessions()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        print(traceback.format_exc())
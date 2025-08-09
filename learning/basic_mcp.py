"""
FastMCP quickstart example.
cd to the `examples/snippets/clients` directory and run:
 uv run server fastmcp_quickstart stdio
"""

from mcp.server.fastmcp import FastMCP
import mcp.types as types
from mcp.shared._httpx_utils import create_mcp_http_client
import math
# Create an MCP server
mcp = FastMCP("Demo", port=8080, host="localhost")

# Add an addition tool
@mcp.tool()
def add(a: float | int, b: float | int) -> float | int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def subtract(a: float | int, b: float | int) -> float | int:
    """Subtract two numbers"""
    return a - b

@mcp.tool()
def multiply(a: float | int, b: float | int) -> float | int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide(a: float | int, b: float | int) -> float | int:
    """Divide two numbers"""
    return a / b

@mcp.tool()
def square(a: float | int) -> float | int:
    """Square a number"""
    return a * a

@mcp.tool()
def cube(a: float | int) -> float | int:
    """Cube a number"""
    return a * a * a

@mcp.tool()
def sqrt(a: float | int) -> float | int:
    """Square root of a number"""
    return math.sqrt(a)

@mcp.tool()
def sin(a: float | int) -> float | int:
    """Sine of a number"""
    return math.sin(a)

@mcp.tool()
def cos(a: float | int) -> float | int:
    """Cosine of a number"""
    return math.cos(a)

@mcp.tool()
def tan(a: float | int) -> float | int:
    """Tangent of a number"""
    return math.tan(a)

@mcp.tool()
def log(a: float | int) -> float | int:
    """Natural logarithm of a number"""
    return math.log(a)

@mcp.tool()
def exp(a: float | int) -> float | int:
    """Exponential of a number"""
    return math.exp(a)

@mcp.tool()
def pow(a: float | int, b: float | int) -> float | int:
    """Power of a number"""
    return math.pow(a, b)

@mcp.tool()
def factorial(a: float | int) -> float | int:
    """Factorial of a number"""
    return math.factorial(a)

@mcp.tool()
def pi() -> float:
    """Return the value of pi"""
    return math.pi

@mcp.tool()
def e() -> float:
    """Return the value of e"""
    return math.e

@mcp.tool()
def phi() -> float:
    """Return the value of phi"""
    return (1 + math.sqrt(5)) / 2

@mcp.tool()
def fibonacci(n: float | int) -> float | int:
    """Return the nth Fibonacci number"""
    return math.fibonacci(n)

@mcp.tool()
def is_prime(n: float | int) -> bool:
    """Check if a number is prime"""
    return math.isprime(n)

@mcp.tool()
def is_even(n: float | int) -> bool:
    """Check if a number is even"""
    return n % 2 == 0

@mcp.tool()
def is_odd(n: float | int) -> bool:
    """Check if a number is odd"""
    return n % 2 != 0


@mcp.tool()
async def fetch_website(url: str) -> list[types.ContentBlock]:
    """Fetch a website and return its content as a list of strings"""
    headers = {
        "User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"
    }
    async with create_mcp_http_client(headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return [types.TextContent(type="text", text=response.text)]

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# Add a prompt
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }
    return f"{styles.get(style, styles['friendly'])} for someone named {name}."

# Run server with stdio transport
if __name__ == "__main__":
    mcp.run(transport="sse")
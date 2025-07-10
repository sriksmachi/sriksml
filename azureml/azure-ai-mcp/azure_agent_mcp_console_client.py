#!/usr/bin/env python
"""
MCP client that connects to an MCP server, loads tools, and runs a chat loop using Google Gemini LLM.
"""

import asyncio
import os
import sys
import json
from contextlib import AsyncExitStack
from typing import Optional, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env (e.g., GOOGLE_API_KEY)

# Custom JSON encoder for objects with 'content' attribute


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "content"):
            return {"type": o.__class__.__name__, "content": o.content}
        return super().default(o)


print(os.environ["AZURE_OPENAI_ENDPOINT"])

# Instantiate Google Gemini LLM with deterministic output and retry logic
llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

# Require server script path as command-line argument
if len(sys.argv) < 2:
    print("Usage: python client_langchain_google_genai_bind_tools.py <path_to_server_script>")
    sys.exit(1)

server_script = sys.argv[1]

# Configure MCP server startup parameters
server_params = StdioServerParameters(
    command="python" if server_script.endswith(".py") else "node",
    args=[server_script],
)

# Global holder for the active MCP session (used by tool adapter)
mcp_client = None

# Main async function: connect, load tools, create agent, run chat loop


async def run_agent():
    global mcp_client
    print("Starting MCP Client...")
    async with stdio_client(server_params) as (read, write):
        print("MCP Client connected!")
        print("Server parameters:")
        print(f"  Command: {server_params}")
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_client = type("MCPClientHolder", (), {"session": session})()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, tools)
            print("MCP Client Started! Type 'quit' to exit.")
            while True:
                query = input("\\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                # Send user query to agent and print formatted response
                response = await agent.ainvoke({"messages": query})
                try:
                    formatted = json.dumps(
                        response, indent=2, cls=CustomEncoder)
                except Exception:
                    formatted = str(response)
                print("\\nResponse:")
                print(formatted)
    return

# Entry point: run the async agent loop
if __name__ == "__main__":
    asyncio.run(run_agent())

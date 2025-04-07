"""Azure AI Agent Service MCP Server"""

import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import MessageRole, Agent
from azure.identity.aio import DefaultAzureCredential

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("azure_agent_mcp")

# Global variables for client and agent cache
ai_client = None
agent_cache = {}
default_agent_id = None


def initialize_server():
    """Initialize the Azure AI Agent client asynchronously."""
    global ai_client, default_agent_id

    # Load environment variables
    project_connection_string = os.getenv("PROJECT_CONNECTION_STRING")
    default_agent_id = os.getenv("DEFAULT_AGENT_ID")

    # Validate essential environment variables
    if not project_connection_string:
        logger.error(
            "Missing required environment variable: PROJECT_CONNECTION_STRING")
        return False

    try:
        credential = DefaultAzureCredential()
        ai_client = AIProjectClient.from_connection_string(
            credential=credential,
            conn_str=project_connection_string,
            user_agent="mcp-azure-ai-agent",
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize AIProjectClient: {str(e)}")
        return False


async def get_agent(agent_id: str) -> Agent:
    """Get an agent by ID with simple caching."""
    global agent_cache

    # Check cache first
    if agent_id in agent_cache:
        return agent_cache[agent_id]

    # Fetch agent if not in cache
    try:
        agent = await ai_client.agents.get_agent(agent_id=agent_id)
        agent_cache[agent_id] = agent
        return agent
    except Exception as e:
        logger.error(
            f"Agent retrieval failed - ID: {agent_id}, Error: {str(e)}")
        raise ValueError(f"Agent not found or inaccessible: {agent_id}")


async def query_agent(agent_id: str, query: str) -> str:
    """Query an Azure AI Agent and get the response."""
    try:
        # Get agent (from cache or fetch it)
        agent = await get_agent(agent_id)

        # Always create a new thread
        thread = await ai_client.agents.create_thread()
        thread_id = thread.id

        # Add message to thread
        await ai_client.agents.create_message(
            thread_id=thread_id, role=MessageRole.USER, content=query
        )

        # Process the run asynchronously
        run = await ai_client.agents.create_run(thread_id=thread_id, agent_id=agent_id)

        # Poll until the run is complete
        while run.status in ["queued", "in_progress", "requires_action"]:
            await asyncio.sleep(1)  # Non-blocking sleep
            run = await ai_client.agents.get_run(thread_id=thread_id, run_id=run.id)

        if run.status == "failed":
            error_msg = f"Agent run failed: {run.last_error}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        # Get the agent's response
        response_messages = await ai_client.agents.list_messages(thread_id=thread_id)
        response_message = response_messages.get_last_message_by_role(
            MessageRole.AGENT)

        result = ""
        citations = []

        if response_message:
            # Collect text content
            for text_message in response_message.text_messages:
                result += text_message.text.value + "\n"

            # Collect citations
            for annotation in response_message.url_citation_annotations:
                citation = (
                    f"[{annotation.url_citation.title}]({annotation.url_citation.url})"
                )
                if citation not in citations:
                    citations.append(citation)

        # Add citations if any
        if citations:
            result += "\n\n## Sources\n"
            for citation in citations:
                result += f"- {citation}\n"

        return result.strip()

    except Exception as e:
        logger.error(f"Agent query failed - ID: {agent_id}, Error: {str(e)}")
        raise


# Initialize MCP and server
load_dotenv()
server_initialized = initialize_server()
mcp = FastMCP(
    "azure-agent",
    description="MCP server for Azure AI Agent Service integration",
    dependencies=["azure-identity", "python-dotenv",
                  "azure-ai-projects", "aiohttp"],
)


@mcp.tool()
async def connect_agent(agent_id: str, query: str, ctx: Context = None) -> str:
    """Connect to a specific Azure AI Agent."""
    if not server_initialized:
        return "Error: Azure AI Agent server is not initialized. Check server logs for details."

    try:
        response = await query_agent(agent_id, query)
        return f"## Response from Azure AI Agent\n\n{response}"
    except Exception as e:
        return f"Error connecting to agent: {str(e)}"


@mcp.tool()
async def query_default_agent(query: str, ctx: Context = None) -> str:
    """Send a query to the default configured Azure AI Agent."""
    if not server_initialized:
        return "Error: Azure AI Agent server is not initialized. Check server logs for details."

    if not default_agent_id:
        return "Error: No default agent configured. Set DEFAULT_AGENT_ID environment variable or use connect_agent tool."

    try:
        response = await query_agent(default_agent_id, query)
        return f"## Response from Default Azure AI Agent\n\n{response}"
    except Exception as e:
        return f"Error querying default agent: {str(e)}"


@mcp.tool()
async def list_agents() -> str:
    """List available agents in the Azure AI Agent Service."""
    if not server_initialized:
        return "Error: Azure AI Agent server is not initialized. Check server logs for details."

    try:
        agents = await ai_client.agents.list_agents()
        if not agents or not agents.data:
            return "No agents found in the Azure AI Agent Service."

        result = "## Available Azure AI Agents\n\n"
        for agent in agents.data:
            result += f"- **{agent.name}**: `{agent.id}`\n"

        if default_agent_id:
            result += f"\n**Default Agent ID**: `{default_agent_id}`"

        return result
    except Exception as e:
        return f"Error listing agents: {str(e)}"


if __name__ == "__main__":
    status = "successfully initialized" if server_initialized else "initialization failed"
    print(
        f"\n{'='*50}\nAzure AI Agent MCP Server {status}\nStarting server...\n{'='*50}\n")
    mcp.run()

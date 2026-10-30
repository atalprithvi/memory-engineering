"""Talk to the local MCP tool server."""

from pathlib import Path

from fastmcp import Client

from memory_agent.agent.state import AgentState

TOOL_SERVER = Path(__file__).resolve().parent / "server.py"


def open_client() -> Client:
    return Client(TOOL_SERVER)


async def execute_action(action: str, _state: AgentState, client: Client) -> str:
    """Run one tool. Failures come back as text so the planner can recover."""
    try:
        result = await client.call_tool(action, {})
    except Exception as exc:
        return f"Error: {exc}"

    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            return text
    return str(result)

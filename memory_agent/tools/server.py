"""Local MCP tools the agent can call."""

import secrets
import string
from datetime import datetime

from fastmcp import FastMCP

mcp = FastMCP("Utility Tools")

_MIN_PASSWORD_LENGTH = 8
_MAX_PASSWORD_LENGTH = 128


@mcp.tool()
def current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().astimezone().strftime("%d-%m-%Y %H:%M:%S")


@mcp.tool()
def roll_dice() -> int:
    """Roll a six-sided die."""
    return secrets.randbelow(6) + 1


@mcp.tool()
def generate_password(length: int = 12) -> str:
    """Generate a secure random password."""
    if not _MIN_PASSWORD_LENGTH <= length <= _MAX_PASSWORD_LENGTH:
        raise ValueError(
            f"length must be between {_MIN_PASSWORD_LENGTH} and {_MAX_PASSWORD_LENGTH}"
        )

    alphabet = string.ascii_letters + string.digits + string.punctuation
    return "".join(secrets.choice(alphabet) for _ in range(length))


if __name__ == "__main__":
    mcp.run()

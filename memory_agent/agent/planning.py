"""Turn a user request into tasks, then choose the next tool."""

import re
from collections.abc import Sequence
from textwrap import dedent
from typing import Protocol

from memory_agent.llm import chat
from memory_agent.agent.state import AgentState

_TASK_PREFIX = "TASK:"
_TOOL_PREFIX = "TOOL:"
_LIST_MARKER = re.compile(r"^(?:[-*]|\d+[.)])\s+")


class ToolInfo(Protocol):
    """Fields the planner reads from an MCP tool."""

    name: str
    description: str | None


def parse_task_list(raw: str) -> list[str]:
    tasks = []
    for line in raw.splitlines():
        cleaned = _LIST_MARKER.sub("", line.strip()).strip()
        if not cleaned or cleaned.upper() == "NONE":
            continue
        tasks.append(cleaned)
    return tasks


def parse_decision(decision: str) -> tuple[str | None, str | None]:
    """Read TASK and TOOL lines. Accepts both fields on one line."""
    task = None
    tool = None

    for line in decision.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith(_TASK_PREFIX):
            task = stripped.split(":", 1)[1].strip()
        elif upper.startswith(_TOOL_PREFIX):
            tool = stripped.split(":", 1)[1].strip()

    if task:
        tool_at = task.upper().find(_TOOL_PREFIX)
        if tool_at != -1:
            inline_tool = task[tool_at + len(_TOOL_PREFIX) :].strip()
            task = task[:tool_at].strip()
            tool = tool or inline_tool

    return task or None, tool or None


def is_finish_decision(decision: str) -> bool:
    normalized = decision.strip().upper()
    return normalized in {"FINISH", "TASK: FINISH"}


def decompose_task(user_request: str) -> list[str]:
    prompt = dedent(
        f"""
        You are an AI task planner.

        Break the user's request into smaller tasks.

        Rules:
        1. Create only necessary tasks.
        2. Put tasks in execution order.
        3. Each task should be a short action.
        4. Return one task per line.
        5. Do not number the tasks.
        6. Do not explain anything.

        Return NONE if the request belongs to any of these categories:
        1. Greetings or casual conversation (for example "hi", "hello", "how are you?").
        2. Introductions or personal details (for example "my name is Pankaj").
        3. Questions about past interactions (for example "what is my name?").
        4. Questions that can be answered directly without tools.

        User request:
        {user_request}
        """
    ).strip()

    return parse_task_list(
        chat(prompt, system_prompt="You are an AI task planner.")
    )


def _tool_catalog(tools: Sequence[ToolInfo]) -> str:
    blocks = []
    for tool in tools:
        blocks.append(
            "\n".join(
                [
                    "Tool:",
                    tool.name,
                    "",
                    "Description:",
                    (tool.description or "").strip(),
                    "----------------------------",
                ]
            )
        )
    return "\n".join(blocks) if blocks else "NONE"


def plan_next_action(state: AgentState, tools: Sequence[ToolInfo]) -> str:
    prompt = dedent(
        f"""
        You are an AI planner.

        Select the next unfinished task and the best tool to complete it.

        User request:
        {state.user_request}

        Relevant previous memories:
        {state.format_memories()}

        Tasks:
        {state.format_tasks()}

        Pending tasks:
        {state.format_pending_tasks()}

        Completed actions:
        {state.format_list(state.actions)}

        Observations:
        {state.format_observations()}

        Intermediate results:
        {state.format_list(state.results)}

        Available tools:
        {_tool_catalog(tools)}

        Rules:
        1. Select only ONE pending task.
        2. The TASK must be copied exactly from the pending task list.
        3. Do not change the task wording.
        4. Do not select a completed task.
        5. Select the best available tool for that task.
        6. Use previous observations, results, and relevant memories.
        7. If an action failed, reconsider the task.
        8. If there are no pending tasks, return only the word FINISH.
        9. If tasks are still pending, return exactly two lines:
           TASK: <exact pending task>
           TOOL: <tool_name>
        10. Do not explain anything.
        11. If the request can be answered from memory or general knowledge, or the task list is empty, return only FINISH.

        Next decision:
        """
    ).strip()

    return chat(prompt, system_prompt="You are an AI planner.")

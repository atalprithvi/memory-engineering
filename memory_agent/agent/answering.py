"""Final user-facing answer and long-term memory extraction."""

from textwrap import dedent

from memory_agent.llm import chat
from memory_agent.agent.state import AgentState

_NO_MEMORY = "NONE"


def format_answer(state: AgentState) -> str:
    prompt = dedent(
        f"""
        The user asked:
        {state.user_request}

        Relevant previous memories:
        {state.format_memories()}

        Tasks:
        {state.format_tasks()}

        Actions:
        {state.format_list(state.actions)}

        Observations:
        {state.format_observations()}

        Intermediate results:
        {state.format_list(state.results)}

        Give the user a natural and concise answer.
        Use relevant previous memories when they help answer the question.

        Do not mention the planner, the agent loop, MCP, internal state, tools, or internal reasoning.
        """
    ).strip()
    return chat(prompt)


def extract_memory(user_message: str, assistant_message: str) -> str:
    prompt = dedent(
        f"""
        You are a memory extraction system.

        Read the conversation below and decide whether it contains useful long-term information about the user.

        User:
        {user_message}

        Assistant:
        {assistant_message}

        Store information only if it will be useful in future conversations.

        Useful information includes preferences, interests, goals, skills, projects, and important user facts.

        Do not store temporary questions, the current time, one-off calculations, one-time requests, tool results, or casual conversation.

        If useful information exists, return only the fact that should be remembered.
        If there is nothing useful to remember, return {_NO_MEMORY}.
        """
    ).strip()
    return chat(prompt, system_prompt="You extract useful long-term memories.")


def is_empty_memory(fact: str) -> bool:
    return not fact.strip() or fact.strip().upper() == _NO_MEMORY

"""Run the agent from the command line: python -m memory_agent"""

import asyncio

from memory_agent.agent.answering import extract_memory, format_answer
from memory_agent.agent.loop import run_agent_loop
from memory_agent.agent.planning import decompose_task, plan_next_action
from memory_agent.config import load_settings
from memory_agent.memory.store import MemoryStore
from memory_agent.tools.client import execute_action, open_client


async def main() -> None:
    settings = load_settings()
    store = MemoryStore(settings.database_path)
    store.initialize()

    async with open_client() as client:
        tools = await client.list_tools()
        _print_tools(tools)

        user_request = input("\nUser: ").strip()
        if not user_request:
            print("No request provided.")
            return

        state = await run_agent_loop(
            user_request,
            planner=plan_next_action,
            decompose=decompose_task,
            execute_action=execute_action,
            format_answer=format_answer,
            extract_memory=extract_memory,
            store=store,
            tools=tools,
            client=client,
        )

    print("\nFinal answer")
    print("------------")
    print(state.final_answer)


def _print_tools(tools) -> None:
    print("\nAvailable tools")
    print("---------------")
    for tool in tools:
        print(f"- {tool.name}")


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()

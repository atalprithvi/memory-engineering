"""Plan, act, and stop once the request is done or the step budget is spent."""

from collections.abc import Awaitable, Callable, Sequence

from memory_agent.agent.answering import is_empty_memory
from memory_agent.agent.planning import is_finish_decision, parse_decision
from memory_agent.agent.state import AgentState, Observation
from memory_agent.memory.store import MemoryStore

Planner = Callable[[AgentState, Sequence], str]
Decomposer = Callable[[str], list[str]]
Executor = Callable[[str, AgentState, object], Awaitable[str]]
Answerer = Callable[[AgentState], str]
Extractor = Callable[[str, str], str]

_STEP_LIMIT_MESSAGE = "I could not complete the request within the step limit."
_FAILED_ACTION = "The previous action failed. The planner must reconsider."


async def run_agent_loop(
    user_request: str,
    *,
    planner: Planner,
    decompose: Decomposer,
    execute_action: Executor,
    format_answer: Answerer,
    extract_memory: Extractor,
    store: MemoryStore,
    tools: Sequence,
    client: object,
    max_steps: int = 5,
) -> AgentState:
    state = AgentState(user_request=user_request, max_steps=max_steps)
    state.memories.extend(store.search(user_request))

    for description in decompose(user_request):
        state.add_task(description)
    _print_plan(state)

    while state.can_continue():
        if state.all_tasks_completed():
            _print_step(state)
            print("Planner Decision:\nTASK: FINISH")
            _finish(state, format_answer, extract_memory, store)
            break

        _print_step(state)
        decision = planner(state, tools)
        print("Planner Decision:")
        print(decision)

        if is_finish_decision(decision):
            _finish(state, format_answer, extract_memory, store)
            break

        task, tool = parse_decision(decision)
        if not task or not tool:
            state.results.append("Invalid planner decision.")
            state.advance()
            continue

        await _run_tool(state, task, tool, execute_action, client)
        if state.all_tasks_completed():
            print("Planner Decision:\nTASK: FINISH")
            _finish(state, format_answer, extract_memory, store)
            break

    if not state.finished and state.all_tasks_completed():
        _finish(state, format_answer, extract_memory, store)
    elif not state.finished:
        state.final_answer = _STEP_LIMIT_MESSAGE
    return state


def _finish(
    state: AgentState,
    format_answer: Answerer,
    extract_memory: Extractor,
    store: MemoryStore,
) -> None:
    answer = format_answer(state)
    state.finish(answer)
    store.save_conversation(state.user_request, answer)

    fact = extract_memory(state.user_request, answer)
    if not is_empty_memory(fact):
        store.save_fact(fact)


async def _run_tool(
    state: AgentState,
    task: str,
    tool: str,
    execute_action: Executor,
    client: object,
) -> None:
    state.current_task = task
    observation = await execute_action(tool, state, client)
    print("Observation:", observation)

    state.actions.append(tool)
    state.observations.append(Observation(state.current_step, tool, observation))
    state.results.append(observation)

    if observation.lower().startswith("error"):
        state.results.append(_FAILED_ACTION)
    else:
        state.complete_task(task)
        print("Completed:", task)

    state.advance()


def _print_plan(state: AgentState) -> None:
    print("\nTask plan")
    print("---------")
    if not state.tasks:
        print("- none")
        return
    for task in state.tasks:
        print(f"- {task.description}")


def _print_step(state: AgentState) -> None:
    print(f"\n--- Step {state.current_step + 1} ---")

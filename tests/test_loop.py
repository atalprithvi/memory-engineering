import tempfile
import unittest
from pathlib import Path

from memory_agent.agent.loop import run_agent_loop
from memory_agent.agent.state import AgentState
from memory_agent.memory.store import MemoryStore


def _embed(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "python" in lowered else 0.0,
        1.0 if "time" in lowered else 0.0,
        1.0,
    ]


class _Client:
    pass


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.directory.name) / "memory.db", embed=_embed)
        self.store.initialize()

    def tearDown(self):
        self.directory.cleanup()

    async def test_finished_work_is_saved(self):
        async def execute(action, state, client):
            return "30-10-2026 23:00:00"

        state = await run_agent_loop(
            "What time is it?",
            planner=lambda state, tools: "TASK: Get the current time\nTOOL: current_time",
            decompose=lambda request: ["Get the current time"],
            execute_action=execute,
            format_answer=lambda state: "It is 11 PM.",
            extract_memory=lambda user, assistant: "NONE",
            store=self.store,
            tools=[],
            client=_Client(),
        )

        self.assertEqual(state.final_answer, "It is 11 PM.")
        self.assertTrue(state.finished)
        saved = self.store.search("What time is it?")
        self.assertEqual(saved[0].assistant_message, "It is 11 PM.")

    async def test_useful_fact_is_stored_on_finish(self):
        state = await run_agent_loop(
            "I am learning Python.",
            planner=lambda state, tools: "FINISH",
            decompose=lambda request: [],
            execute_action=self._unused_tool,
            format_answer=lambda state: "Got it. You are learning Python.",
            extract_memory=lambda user, assistant: "The user is learning Python.",
            store=self.store,
            tools=[],
            client=_Client(),
        )

        self.assertTrue(state.finished)
        facts = self.store.search("python")
        self.assertTrue(any("learning Python" in item.user_message for item in facts))

    async def test_repeated_tasks_finish_inside_the_step_budget(self):
        calls = []

        async def execute(action, state, client):
            calls.append(action)
            return "ok"

        def planner(state, tools):
            task = state.pending_tasks()[0]
            tool = "generate_password" if "password" in task else "roll_dice"
            return f"TASK: {task}\nTOOL: {tool}"

        state = await run_agent_loop(
            "Roll the dice three times, then generate a password.",
            planner=planner,
            decompose=lambda request: [
                "roll dice",
                "roll dice",
                "roll dice",
                "generate password",
            ],
            execute_action=execute,
            format_answer=lambda state: "Rolled three times. Password is ready.",
            extract_memory=lambda user, assistant: "NONE",
            store=self.store,
            tools=[],
            client=_Client(),
            max_steps=4,
        )

        self.assertEqual(
            calls,
            ["roll_dice", "roll_dice", "roll_dice", "generate_password"],
        )
        self.assertTrue(state.all_tasks_completed())
        self.assertTrue(state.finished)
        self.assertEqual(state.final_answer, "Rolled three times. Password is ready.")

    async def test_step_limit_does_not_invent_an_answer_from_tools(self):
        state = await run_agent_loop(
            "Do something",
            planner=lambda state, tools: "not a decision",
            decompose=lambda request: ["Do something"],
            execute_action=self._unused_tool,
            format_answer=lambda state: "unused",
            extract_memory=lambda user, assistant: "NONE",
            store=self.store,
            tools=[],
            client=_Client(),
            max_steps=1,
        )

        self.assertFalse(state.finished)
        self.assertIn("step limit", state.final_answer)
        self.assertIsInstance(state, AgentState)

    async def _unused_tool(self, action, state, client):
        raise AssertionError("tool should not run")

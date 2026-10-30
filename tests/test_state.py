import unittest

from memory_agent.agent.state import AgentState


class AgentStateTests(unittest.TestCase):
    def test_complete_task_ignores_case_and_underscores(self):
        state = AgentState(user_request="make a password")
        state.add_task("Generate password")

        self.assertTrue(state.complete_task("generate_password"))
        self.assertEqual(state.tasks[0].status, "completed")
        self.assertTrue(state.all_tasks_completed())

    def test_complete_task_closes_one_copy_at_a_time(self):
        state = AgentState(user_request="roll the dice three times")
        state.add_task("roll dice")
        state.add_task("roll dice")
        state.add_task("roll dice")

        self.assertTrue(state.complete_task("roll dice"))
        self.assertEqual(
            [task.status for task in state.tasks],
            ["completed", "pending", "pending"],
        )
        self.assertTrue(state.complete_task("roll dice"))
        self.assertEqual(
            [task.status for task in state.tasks],
            ["completed", "completed", "pending"],
        )

    def test_empty_task_list_is_not_complete(self):
        state = AgentState(user_request="hello")
        self.assertFalse(state.all_tasks_completed())

    def test_step_budget_stops_the_loop(self):
        state = AgentState(user_request="hello", max_steps=1)
        self.assertTrue(state.can_continue())
        state.advance()
        self.assertFalse(state.can_continue())

    def test_finish_records_the_answer(self):
        state = AgentState(user_request="hello")
        state.finish("Hi there.")
        self.assertEqual(state.final_answer, "Hi there.")
        self.assertFalse(state.can_continue())

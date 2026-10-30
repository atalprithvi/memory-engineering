import unittest

from memory_agent.agent.planning import is_finish_decision, parse_decision, parse_task_list


class PlanningParseTests(unittest.TestCase):
    def test_parse_decision_reads_two_lines(self):
        decision = "TASK: Get the current time\nTOOL: current_time"
        self.assertEqual(
            parse_decision(decision),
            ("Get the current time", "current_time"),
        )

    def test_parse_decision_reads_both_fields_on_one_line(self):
        decision = "TASK: Roll a die TOOL: roll_dice"
        self.assertEqual(parse_decision(decision), ("Roll a die", "roll_dice"))

    def test_parse_decision_rejects_incomplete_output(self):
        self.assertEqual(parse_decision("TASK: only a task"), ("only a task", None))

    def test_finish_decision(self):
        self.assertTrue(is_finish_decision("FINISH"))
        self.assertTrue(is_finish_decision("task: finish"))
        self.assertFalse(is_finish_decision("TASK: Get the time\nTOOL: current_time"))

    def test_parse_task_list_drops_none_and_numbering(self):
        raw = "NONE\n1. Get the current time\n- Roll a die\n"
        self.assertEqual(parse_task_list(raw), ["Get the current time", "Roll a die"])

    def test_parse_task_list_treats_none_as_no_work(self):
        self.assertEqual(parse_task_list("NONE"), [])

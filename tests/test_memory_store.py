import tempfile
import unittest
from pathlib import Path

from memory_agent.memory.store import MemoryStore, cosine_similarity


def _embed(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "name" in lowered or "pankaj" in lowered else 0.0,
        1.0 if "python" in lowered else 0.0,
        1.0 if "time" in lowered else 0.0,
    ]


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.directory.name) / "memory.db", embed=_embed)
        self.store.initialize()

    def tearDown(self):
        self.directory.cleanup()

    def test_search_returns_the_closest_memory(self):
        self.store.save_conversation("My name is Pankaj.", "Nice to meet you, Pankaj.")
        self.store.save_conversation("I am learning Python.", "Python is useful for AI.")

        results = self.store.search("What is my name?")

        self.assertEqual(len(results), 1)
        self.assertIn("Pankaj", results[0].user_message)
        self.assertGreater(results[0].score, 0.2)

    def test_fact_is_searchable(self):
        self.store.save_fact("The user is learning Python.")
        results = self.store.search("python")
        self.assertEqual(results[0].user_message, "The user is learning Python.")
        self.assertEqual(results[0].assistant_message, "")

    def test_zero_vectors_do_not_divide_by_zero(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [0.0, 0.0]), 0.0)

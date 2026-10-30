"""SQLite store with brute-force cosine search over embeddings.

Every row is scored in Python. That is fine for a small personal memory file
and avoids a separate vector database.
"""

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from memory_agent.agent.state import MemoryRecord

EmbedFn = Callable[[str], list[float]]

DEFAULT_TOP_K = 3
DEFAULT_SCORE_THRESHOLD = 0.20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT,
    assistant_message TEXT,
    embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


class MemoryStore:
    def __init__(self, database_path: Path | str, embed: EmbedFn | None = None) -> None:
        self.database_path = Path(database_path)
        self._embed = embed

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_SCHEMA)

    def save_conversation(self, user_message: str, assistant_message: str) -> None:
        text = f"{user_message}\n{assistant_message}"
        self._insert(user_message, assistant_message, text)

    def save_fact(self, fact: str) -> None:
        self._insert(fact, "", fact)

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> list[MemoryRecord]:
        query_embedding = self._embedding_for(query)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT user_message, assistant_message, embedding, created_at
                FROM memories
                """
            ).fetchall()

        matches = []
        for user_message, assistant_message, embedding_json, created_at in rows:
            score = cosine_similarity(query_embedding, json.loads(embedding_json))
            if score < threshold:
                continue
            matches.append(
                MemoryRecord(
                    user_message=user_message,
                    assistant_message=assistant_message or "",
                    created_at=str(created_at),
                    score=score,
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]

    def _insert(self, user_message: str, assistant_message: str, text: str) -> None:
        embedding = json.dumps(self._embedding_for(text))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (user_message, assistant_message, embedding)
                VALUES (?, ?, ?)
                """,
                (user_message, assistant_message, embedding),
            )

    def _embedding_for(self, text: str) -> list[float]:
        if self._embed is None:
            from memory_agent.llm import embed_text

            self._embed = embed_text
        return self._embed(text)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

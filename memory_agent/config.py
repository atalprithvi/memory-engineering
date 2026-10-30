"""Runtime settings loaded from the environment."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_REQUIRED = ("BASE_URL", "API_KEY", "MODEL")


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    embedding_model: str
    database_path: Path


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    load_dotenv()
    values = {name: os.getenv(name) for name in _REQUIRED}
    missing = [name for name, value in values.items() if not value]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {names}")

    return Settings(
        base_url=values["BASE_URL"],
        api_key=values["API_KEY"],
        model=values["MODEL"],
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        database_path=Path(os.getenv("DATABASE_PATH", "memory.db")),
    )

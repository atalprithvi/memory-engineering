"""Shared chat and embedding calls."""

from openai import OpenAI

from memory_agent.config import load_settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = load_settings()
        _client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    return _client


def chat(user_prompt: str, *, system_prompt: str | None = None) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = get_client().chat.completions.create(
        model=load_settings().model,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def embed_text(text: str) -> list[float]:
    response = get_client().embeddings.create(
        model=load_settings().embedding_model,
        input=text,
    )
    return list(response.data[0].embedding)

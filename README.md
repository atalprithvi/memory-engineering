# Memory Agent

A small agent that breaks a request into tasks, calls local tools, and keeps useful facts for later conversations.

It talks to a local model through an OpenAI-compatible API (Ollama by default) and calls tools over MCP.

## What you need

- Python 3.11 or newer
- [Ollama](https://ollama.com/) running on your machine
- These models pulled in Ollama:

```powershell
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
```

## Setup

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

`pip install -r requirements.txt` installs this project and its dependencies. Activate `.venv` in each new terminal before you run the agent.

## Configuration

Settings come from `.env`:

| Variable | Purpose | Default in `.env.example` |
|---|---|---|
| `BASE_URL` | Chat and embedding API | `http://localhost:11434/v1` |
| `API_KEY` | API key | `ollama` |
| `MODEL` | Chat model | `qwen3:1.7b` |
| `EMBEDDING_MODEL` | Model used to store and search memories | `nomic-embed-text` |
| `DATABASE_PATH` | SQLite file for memories | `memory.db` |

## Run

With the virtual environment active and Ollama running:

```powershell
python -m memory_agent
```

`memory-agent` does the same thing.

The agent can use these tools:

- `current_time` returns the local date and time
- `roll_dice` rolls a six-sided die
- `generate_password` creates a random password

Memories are written to `memory.db` after a finished run. Questions such as "what is my name?" can use facts saved from earlier conversations.

## Layout

```
memory_agent/
  agent/          state, planning, answers, and the loop
  memory/         SQLite memory store
  tools/          MCP tool server and client
  config.py       settings from the environment
  llm.py          shared chat and embedding client
tests/            unit tests
```

## Tests

```powershell
python -m unittest discover -s tests -t .
```

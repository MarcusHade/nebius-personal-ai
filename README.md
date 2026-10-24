# nebius-personal-ai

Email assistant POC for the NVIDIA x Nebius hackathon. Ask a question in plain English, a Nemotron model on
Nebius Token Factory decides which Gmail tools to call, and ChromaDB remembers what it has seen.

```
browser / curl
     |  POST /chat
     v
FastAPI (app/main.py)
     v
Agent loop (app/agent.py) --recall--> ChromaDB (app/memory.py)  <--index-- tool results + past Q&A
     |  chat + tools
     v
Nemotron Nano @ Nebius Token Factory (app/llm.py, OpenAI-compatible)
     |  tool_calls
     v
MCP client (app/mcp_client.py) --stdio--> Gmail MCP server (mcp_servers/gmail_server.py, read-only)
```

## Setup

1. `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. `cp .env.example .env`, then add your `NEBIUS_API_KEY`.
3. Gmail access (one time):
   - Google Cloud Console: new project, enable **Gmail API**.
   - OAuth consent screen: External, add your own address as a **test user**.
   - Credentials: create OAuth client ID, type **Desktop app**, download the JSON to `data/secrets/credentials.json`.
   - `python mcp_servers/auth_gmail.py` (opens a browser, writes `data/secrets/token.json`).
4. Check the model before anything else:
   - `python scripts/list_models.py` and copy the exact Nano id into `NEBIUS_MODEL` in `.env`.
   - `python scripts/check_tool_calling.py` should print PASS.
5. `uvicorn app.main:app --reload`, then open http://127.0.0.1:8000

Tests (offline, no keys needed): `pytest`

## Endpoints

| | |
|---|---|
| `GET /` | minimal chat UI |
| `POST /chat` `{message, session_id?}` | returns `{answer, session_id, tool_calls, memory_hits}` |
| `GET /health` | model, MCP tools loaded, memory item count |
| `GET /memory/search?q=` | debug: what the vector store recalls for a query |

## Known limits (on purpose, for now)

- No auth on the API. Bind to localhost only.
- Session history is in RAM; only ChromaDB persists.
- Memory is injected as a hint to the prompt. It does not yet short-circuit the LLM call.
- Gmail scope is `gmail.readonly`, so a prompt-injected email can't send or delete anything.
- The first memory query downloads Chroma's default embedding model (~80 MB).
- `mcp` is pinned `<2`: v2 renamed `FastMCP` to `MCPServer` and changed other APIs.

## Backlog (pick per sprint)

1. Run it for real: confirm Nano slug and tool calling work (steps 4 above). Biggest unknown.
2. Model routing: Nano for tool-calling turns, Super/Ultra for "summarize my week" style reasoning.
3. Semantic cache: answer from memory without calling the LLM when the hit is fresh enough.
4. Daily digest as a Nebius Serverless Job that writes into the same ChromaDB.
5. Streaming responses (SSE) so the UI doesn't sit on "…".
6. Deploy as a Nebius Serverless Endpoint (needs a Dockerfile, persistent volume for `data/`, and auth).

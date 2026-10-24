"""FastAPI interface. Run: uvicorn app.main:app --reload"""
import logging
import shlex
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import Agent
from .config import Settings, get_settings
from .llm import LLM
from .mcp_client import MCPBridge
from .memory import Memory

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class ChatOut(BaseModel):
    answer: str
    session_id: str
    tool_calls: list[str]
    memory_hits: int


def create_app(agent: Agent | None = None, settings: Settings | None = None) -> FastAPI:
    """Pass a prebuilt agent in tests; otherwise real deps are built at startup."""
    s = settings or get_settings()
    sessions: dict[str, list[dict]] = {}  # short-term history, RAM only (POC)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bridge = None
        if agent is None:
            memory = Memory(s.chroma_path)
            bridge = MCPBridge(s.mcp_gmail_command, shlex.split(s.mcp_gmail_args))
            try:
                await bridge.start()
            except Exception:
                # Boot anyway so you can debug the LLM + memory path without Gmail set up.
                log.exception("Gmail MCP failed to start; running with no tools")
            app.state.agent = Agent(LLM(s), bridge, memory, s)
        else:
            app.state.agent = agent
        yield
        if bridge:
            await bridge.stop()

    app = FastAPI(title="nebius-personal-ai", lifespan=lifespan)

    @app.get("/health")
    async def health():
        a: Agent = app.state.agent
        return {
            "status": "ok",
            "model": getattr(a.llm, "model", "?"),
            "tools": [t["function"]["name"] for t in a.bridge.tools],
            "memory_items": a.memory.count(),
        }

    @app.post("/chat", response_model=ChatOut)
    async def chat(body: ChatIn):
        a: Agent = app.state.agent
        sid = body.session_id or uuid.uuid4().hex
        history = sessions.setdefault(sid, [])
        try:
            res = await a.run(body.message, history[-2 * s.history_turns :])
        except Exception as e:
            log.exception("agent failed")
            raise HTTPException(502, f"LLM/agent error: {type(e).__name__}: {e}")
        history += [
            {"role": "user", "content": body.message},
            {"role": "assistant", "content": res.answer},
        ]
        return ChatOut(answer=res.answer, session_id=sid, tool_calls=res.tool_calls, memory_hits=res.memory_hits)

    @app.get("/memory/search")
    async def memory_search(q: str, k: int = 5):
        """Debug: see what the vector store would recall for a query."""
        return app.state.agent.memory.search(q, k, max_distance=2.0)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()

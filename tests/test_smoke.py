"""Offline smoke tests: fake LLM + fake embeddings + a REAL stdio MCP round trip."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastapi.testclient import TestClient

from app.agent import Agent
from app.config import Settings
from app.main import create_app
from app.mcp_client import MCPBridge
from app.memory import Memory

FAKE_SERVER = str(Path(__file__).parent / "fake_mcp_server.py")


class HashEmbed(EmbeddingFunction):
    """Deterministic bag-of-words embedding so tests need no model download."""

    def __init__(self): ...

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for doc in input:
            v = [0.0] * 64
            for w in doc.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v)
        return out

    @staticmethod
    def name() -> str:
        return "hash-embed"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbed()


class FakeLLM:
    """Turn 1: call search_emails. Turn 2: answer using the tool output."""

    model = "fake"

    def __init__(self):
        self.calls = 0
        self.seen_system = []

    async def chat(self, messages, tools=None):
        self.calls += 1
        self.seen_system.append(messages[0]["content"])
        if messages[-1]["role"] == "tool":
            return NS(content="<think>hmm</think>Invoice 42 ($120) is due Friday.", tool_calls=None)
        call = NS(id="c1", function=NS(name="search_emails", arguments=json.dumps({"query": "invoice"})))
        return NS(content=None, tool_calls=[call])


@pytest.fixture
def settings(tmp_path):
    return Settings(chroma_path=tmp_path / "chroma", memory_max_distance=0.9)


@pytest.fixture
def memory(settings):
    return Memory(settings.chroma_path, embedding_function=HashEmbed())


async def make_bridge():
    b = MCPBridge(sys.executable, [FAKE_SERVER])
    await b.start()
    return b


@pytest.mark.asyncio
async def test_mcp_bridge_lists_and_calls_tool():
    b = await make_bridge()
    try:
        assert [t["function"]["name"] for t in b.tools] == ["search_emails"]
        out = json.loads(await b.call("search_emails", {"query": "x"}))
        assert out[0]["id"] == "m1"
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_agent_tool_loop_and_memory(settings, memory):
    b = await make_bridge()
    try:
        llm = FakeLLM()
        agent = Agent(llm, b, memory, settings)

        r1 = await agent.run("any invoices due?", [])
        assert r1.tool_calls == ["search_emails"]
        assert r1.answer == "Invoice 42 ($120) is due Friday."  # <think> stripped
        assert memory.count() == 2  # 1 email + 1 turn

        # second ask: memory should now be recalled into the system prompt
        r2 = await agent.run("when is invoice 42 due?", [])
        assert r2.memory_hits >= 1
        assert "Invoice 42" in llm.seen_system[-1]
    finally:
        await b.stop()


def test_snippet_does_not_clobber_full_body(memory):
    memory.ingest_tool_result("get_email", json.dumps({"id": "z", "from": "a", "subject": "s", "date": "d", "body": "FULL BODY"}))
    memory.ingest_tool_result("search_emails", json.dumps([{"id": "z", "from": "a", "subject": "s", "date": "d", "snippet": "snip"}]))
    assert memory.count() == 1
    assert "FULL BODY" in memory.col.get(ids=["email:z"])["documents"][0]


def test_api_chat_and_health(settings, memory):
    class StubAgent:
        llm = NS(model="stub")
        bridge = NS(tools=[])
        memory = None

        async def run(self, message, history):
            from app.agent import AgentResult
            return AgentResult(answer=f"echo:{message}:{len(history)}")

    stub = StubAgent()
    stub.memory = memory
    with TestClient(create_app(agent=stub, settings=settings)) as c:
        assert c.get("/health").json()["model"] == "stub"
        r = c.post("/chat", json={"message": "hi"}).json()
        assert r["answer"] == "echo:hi:0"
        r2 = c.post("/chat", json={"message": "again", "session_id": r["session_id"]}).json()
        assert r2["answer"] == "echo:again:2"  # history carried over
        assert c.post("/chat", json={"message": ""}).status_code == 422
        assert c.get("/").status_code == 200

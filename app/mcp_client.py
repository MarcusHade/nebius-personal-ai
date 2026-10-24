"""MCP client: spawns an MCP server over stdio and exposes its tools to the LLM."""
import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger("mcp_bridge")


class MCPBridge:
    def __init__(self, command: str, args: list[str]):
        self.params = StdioServerParameters(command=command, args=args)
        self.session: ClientSession | None = None
        self.tools: list[dict] = []  # OpenAI function-calling schema
        self._stack = AsyncExitStack()

    async def start(self) -> None:
        read, write = await self._stack.enter_async_context(stdio_client(self.params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        listed = await self.session.list_tools()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in listed.tools
        ]
        log.info("MCP tools: %s", [t["function"]["name"] for t in self.tools])

    async def call(self, name: str, args: dict) -> str:
        if not self.session:
            return "ERROR: MCP server not connected"
        res = await self.session.call_tool(name, args)
        text = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
        return f"ERROR: {text}" if res.isError else text

    async def stop(self) -> None:
        await self._stack.aclose()

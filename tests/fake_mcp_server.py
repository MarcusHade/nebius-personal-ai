"""Tiny stand-in MCP server so we can test the real stdio plumbing without Gmail."""
import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-gmail")


@mcp.tool()
def search_emails(query: str = "in:inbox", max_results: int = 10) -> str:
    """Search emails."""
    return json.dumps([
        {"id": "m1", "from": "billing@acme.com", "subject": "Invoice 42 due Friday",
         "date": "Fri, 23 Oct 2026", "snippet": "Your invoice of $120 is due Friday."},
    ])


if __name__ == "__main__":
    mcp.run()

"""Minimal READ-ONLY Gmail MCP server (stdio).

Own code on purpose: it only ever holds the gmail.readonly scope, and we know
exactly what it does with your inbox. Never print() here - stdout is the MCP wire.
Run `python mcp_servers/auth_gmail.py` once first to create the token.
"""
import base64
import html
import json
import os
import re
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
TOKEN = Path(os.getenv("GMAIL_TOKEN_PATH", ROOT / "data" / "secrets" / "token.json"))
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
BODY_LIMIT = 4000

mcp = FastMCP("gmail-readonly")
_svc = None


def svc():
    global _svc
    if _svc is None:
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
        _svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return _svc


def _headers(msg: dict) -> dict:
    h = {x["name"].lower(): x["value"] for x in msg.get("payload", {}).get("headers", [])}
    return {"from": h.get("from", ""), "subject": h.get("subject", ""), "date": h.get("date", "")}


def _body_text(payload: dict) -> str:
    """Prefer text/plain; fall back to tag-stripped text/html."""
    plain, htm = [], []

    def walk(p):
        data = p.get("body", {}).get("data")
        mt = p.get("mimeType", "")
        if data and mt in ("text/plain", "text/html"):
            txt = base64.urlsafe_b64decode(data + "==").decode("utf-8", "replace")
            (plain if mt == "text/plain" else htm).append(txt)
        for sub in p.get("parts", []) or []:
            walk(sub)

    walk(payload)
    if plain:
        return "\n".join(plain)
    raw = "\n".join(htm)
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


@mcp.tool()
def search_emails(query: str = "in:inbox", max_results: int = 10) -> str:
    """Search Gmail with Gmail search syntax (e.g. 'is:unread newer_than:1d', 'from:alice').
    Returns a JSON list of {id, from, subject, date, snippet}."""
    max_results = max(1, min(max_results, 25))
    ids = svc().users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    out = []
    for m in ids.get("messages", []):
        full = (
            svc().users().messages()
            .get(userId="me", id=m["id"], format="metadata", metadataHeaders=["From", "Subject", "Date"])
            .execute()
        )
        out.append({"id": m["id"], **_headers(full), "snippet": full.get("snippet", "")})
    return json.dumps(out)


@mcp.tool()
def get_email(message_id: str) -> str:
    """Read one email by id. Returns JSON {id, from, subject, date, body} (body truncated)."""
    full = svc().users().messages().get(userId="me", id=message_id, format="full").execute()
    body = re.sub(r"\n\s*\n+", "\n\n", _body_text(full.get("payload", {}))).strip()
    return json.dumps({"id": message_id, **_headers(full), "body": body[:BODY_LIMIT]})


@mcp.tool()
def list_labels() -> str:
    """List Gmail labels (useful for 'label:xyz' searches)."""
    labels = svc().users().labels().list(userId="me").execute().get("labels", [])
    return json.dumps([l["name"] for l in labels])


if __name__ == "__main__":
    mcp.run()  # stdio

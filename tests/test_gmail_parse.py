import base64

from mcp_servers.gmail_server import _body_text


def b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def test_prefers_plain_over_html():
    p = {"mimeType": "multipart/alternative", "parts": [
        {"mimeType": "text/plain", "body": {"data": b64("plain wins")}},
        {"mimeType": "text/html", "body": {"data": b64("<b>html</b>")}},
    ]}
    assert _body_text(p) == "plain wins"


def test_html_fallback_strips_tags_and_scripts():
    p = {"mimeType": "text/html", "body": {"data": b64("<style>x{}</style><p>Hello&nbsp;<b>world</b></p><script>evil()</script>")}}
    out = _body_text(p)
    assert "Hello" in out and "world" in out and "evil" not in out and "<" not in out

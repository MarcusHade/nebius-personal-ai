"""Does the chosen Nemotron model do OpenAI-style tool calling on Token Factory?
This is the #1 risk for the whole project, so test it before debugging anything else.
Usage: python scripts/check_tool_calling.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from app.config import get_settings  # noqa: E402

s = get_settings()
client = OpenAI(api_key=s.nebius_api_key, base_url=s.nebius_base_url)
tools = [{
    "type": "function",
    "function": {
        "name": "search_emails",
        "description": "Search Gmail",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}]
r = client.chat.completions.create(
    model=s.nebius_model,
    messages=[{"role": "user", "content": "Find my unread emails from today."}],
    tools=tools,
    tool_choice="auto",
)
m = r.choices[0].message
print("model:", s.nebius_model)
if m.tool_calls:
    print("PASS tool call ->", m.tool_calls[0].function.name, m.tool_calls[0].function.arguments)
else:
    print("FAIL no tool call. Raw content:\n", m.content)

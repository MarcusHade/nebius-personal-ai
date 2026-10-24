"""Print Token Factory model ids that look like Nemotron, so you can set NEBIUS_MODEL.
Usage: python scripts/list_models.py   (reads .env)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from app.config import get_settings  # noqa: E402

s = get_settings()
client = OpenAI(api_key=s.nebius_api_key, base_url=s.nebius_base_url)
ids = sorted(m.id for m in client.models.list())
nemo = [i for i in ids if "nemotron" in i.lower()]
print(f"{len(ids)} models total. Nemotron ones:")
print("\n".join(nemo) or "(none matched - eyeball the full list below)")
if not nemo:
    print("\n".join(ids))

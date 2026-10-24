"""All settings live here. Override anything via env vars or a .env file."""
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- Nebius Token Factory (OpenAI-compatible) ---
    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    # Smallest Nemotron. Slug is UNVERIFIED -> run `python scripts/list_models.py`
    # and paste the exact id into .env as NEBIUS_MODEL.
    nebius_model: str = "nvidia/nemotron-3-nano-30b-a3b"

    # --- Agent ---
    max_agent_steps: int = 5          # tool-call loop cap
    max_tool_result_chars: int = 6000  # truncate what we feed back to the LLM
    history_turns: int = 6             # short-term turns kept per session (in RAM)

    # --- Memory (ChromaDB) ---
    chroma_path: Path = ROOT / "data" / "chroma"
    memory_top_k: int = 4
    memory_max_distance: float = 0.7   # cosine distance; lower = stricter

    # --- Gmail MCP server (our own, read-only; swap for any stdio MCP server) ---
    mcp_gmail_command: str = sys.executable
    mcp_gmail_args: str = str(ROOT / "mcp_servers" / "gmail_server.py")
    gmail_credentials_path: Path = ROOT / "data" / "secrets" / "credentials.json"
    gmail_token_path: Path = ROOT / "data" / "secrets" / "token.json"


def get_settings() -> Settings:
    return Settings()

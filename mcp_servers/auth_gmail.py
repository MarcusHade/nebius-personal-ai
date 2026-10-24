"""One-time Google OAuth. Opens a browser, writes data/secrets/token.json.

Prereq: Google Cloud project -> enable Gmail API -> OAuth consent screen (add yourself as
a test user) -> Credentials -> OAuth client ID, type "Desktop app" -> download JSON to
data/secrets/credentials.json
"""
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "data" / "secrets"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

if __name__ == "__main__":
    creds_file = SECRETS / "credentials.json"
    if not creds_file.exists():
        raise SystemExit(f"Missing {creds_file}. See docstring / README step 3.")
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
    creds = flow.run_local_server(port=0)
    (SECRETS / "token.json").write_text(creds.to_json())
    print("Saved", SECRETS / "token.json")

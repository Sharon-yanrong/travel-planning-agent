"""Runtime config. Reads a local .env file (if present) then environment vars.
Provider-agnostic: any OpenAI-compatible endpoint works by changing these 3 vars."""
import os


def _load_dotenv(path=None):
    """Tiny .env loader so we don't need an extra dependency.
    Looks for .env next to this file, so it works from any working directory."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

if not API_KEY:
    raise SystemExit(
        "No LLM_API_KEY found. Copy .env.example to .env and put your key in it.\n"
        "(DeepSeek: register at platform.deepseek.com, top up a little, create an API key.)"
    )

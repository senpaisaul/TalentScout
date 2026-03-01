"""
config.py
---------
Centralised configuration loaded from environment variables.
Never commit real API keys — use a .env file locally.
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL:   str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Add it to your .env file: OPENAI_API_KEY=your_key_here"
    )

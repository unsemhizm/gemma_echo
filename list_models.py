import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from google import genai

api_key = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key)

for m in client.models.list():
    print(f"Name: {m.name}, Supported: {m.supported_actions}")

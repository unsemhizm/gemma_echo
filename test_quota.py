import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key)

test_models = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
]

for m in test_models:
    try:
        print(f"Testing {m}...", end=" ")
        resp = client.models.generate_content(model=m, contents="Say hello")
        print(f"SUCCESS: {resp.text.strip()[:20]}")
    except Exception as e:
        print(f"FAILED: {str(e)[:80]}")

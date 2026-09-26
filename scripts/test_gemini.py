from __future__ import annotations

"""Test Gemini API key with the new google.genai SDK (not deprecated google.generativeai)."""
import os
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
env_path = Path("/home/soumalya/Work/reframe_ai/.env")
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("FAIL: GEMINI_API_KEY not found in .env")
    sys.exit(1)

print(f"Key present: {len(api_key)} chars, prefix: {api_key[:15]}...")

from google import genai

client = genai.Client(api_key=api_key)
print("google.genai SDK imported successfully")

# List available models
print("\n--- Available Gemini models (first 10) ---")
for model in client.models.list():
    if "gemini" in model.name.lower():
        print(f"  {model.name}: {', '.join(model.supported_actions or [])}")

# Test a simple prompt
print("\n--- Testing API call ---")
try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Hello from ReframeAI. Please respond with exactly: GEMINI_OK",
    )
    print(f"Response: '{response.text.strip()}'")
    if response.text.strip() == "GEMINI_OK":
        print("\n✅ Gemini API key WORKS — call succeeded via google.genai SDK")
    else:
        print(f"\n⚠️ API responded but with: '{response.text.strip()}'")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

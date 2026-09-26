from __future__ import annotations

"""Test Gemini API key connectivity."""
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

try:
    import google.generativeai as genai
    print(f"google-generativeai version: {genai.__version__}")
except ImportError:
    print("FAIL: google-generativeai not installed")
    sys.exit(1)

# Configure and test
genai.configure(api_key=api_key)

# List available models
print("\n--- Available Gemini models ---")
for m in genai.list_models():
    if "gemini" in m.name.lower():
        print(f"  {m.name}: {', '.join(m.supported_generation_methods[:3])}")

# Test a simple prompt
print("\n--- Testing API call ---")
try:
    model = genai.GenerativeModel("gemini-3.8-flash")
    response = model.generate_content("Hello from ReframeAI. Please respond with exactly: GEMINI_OK")
    print(f"Response: '{response.text.strip()}'")
    if response.text.strip() == "GEMINI_OK":
        print("\n✅ Gemini API key WORKS — call succeeded")
    else:
        print(f"\n⚠️ API responded but with: '{response.text.strip()}'")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

"""
Shared google-genai client used by all agents.
Import `gemini` from here instead of creating multiple clients.
"""

import google.generativeai as genai
from config import GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. "
        "Copy .env.example to .env and add your key from "
        "https://aistudio.google.com/app/apikey"
    )

genai.configure(api_key=GOOGLE_API_KEY)

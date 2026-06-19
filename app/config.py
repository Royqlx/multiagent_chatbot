import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ── API ────────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────────
# Free-tier Gemini models (no billing required)
LLM_MODEL: str = "gemini-2.5-flash"           # generation & question-gen

# ── Paths ──────────────────────────────────────────────────────────────────────
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma"
DATA_PATH: str = "./data"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500    # words per chunk
CHUNK_OVERLAP: int = 50  # word overlap between chunks

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K: int = 5

# ── ChromaDB collection name ───────────────────────────────────────────────────
COLLECTION_NAME: str = "ais_documents"

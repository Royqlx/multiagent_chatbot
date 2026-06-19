"""
generate_questions.py
─────────────────────
Generates 50 synthetic Q&A pairs from ChromaDB chunks using Gemini 2.5 Flash.
Run AFTER ingest.py.

Usage:
    cd app
    python generate_questions.py
"""

import json
import random
import os
import chromadb
from google import genai
from agents.retriever import GeminiEmbeddingFunction
from config import CHROMA_PATH, LLM_MODEL, COLLECTION_NAME

QUESTION_PROMPT = """\
You are generating evaluation questions for an automotive standards chatbot.

Given the following passage from an AIS (Automotive Industry Standard) document, generate {n} question(s).

Difficulty: {difficulty}
Type: {q_type}

Rules:
- Questions must be answerable from the passage alone.
- Include the answer for each question.
- Return ONLY a JSON array, no markdown fences:
  [{{"question": "...", "answer": "...", "difficulty": "{difficulty}", "type": "{q_type}"}}]

Passage:
{passage}"""

CONFIGS = [
    {"difficulty": "easy",   "q_type": "factual",   "n": 20},
    {"difficulty": "medium", "q_type": "reasoning", "n": 20},
    {"difficulty": "hard",   "q_type": "multi-hop", "n": 10},
]


def get_random_chunks(n: int) -> list[str]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION_NAME, embedding_function=GeminiEmbeddingFunction())
    all_ids = col.get()["ids"]
    sample_ids = random.sample(all_ids, min(n, len(all_ids)))
    return col.get(ids=sample_ids)["documents"]


def generate_questions() -> list[dict]:
    model = genai.GenerativeModel(LLM_MODEL)
    all_questions: list[dict] = []

    for cfg in CONFIGS:
        chunks = get_random_chunks(cfg["n"])
        batch: list[dict] = []

        for chunk in chunks:
            prompt = QUESTION_PROMPT.format(
                n=1,
                difficulty=cfg["difficulty"],
                q_type=cfg["q_type"],
                passage=chunk[:1500],
            )
            try:
                response = model.generate_content(prompt)
                text = response.text.strip().replace("```json", "").replace("```", "").strip()
                batch.extend(json.loads(text))
            except Exception as e:
                print(f"  Skipped chunk: {e}")

        trimmed = batch[: cfg["n"]]
        all_questions.extend(trimmed)
        print(f"Generated {len(trimmed)} {cfg['difficulty']} {cfg['q_type']} questions")

    all_questions = all_questions[:50]

    os.makedirs("eval", exist_ok=True)
    with open("eval/questions.json", "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\nSaved {len(all_questions)} questions to eval/questions.json")
    return all_questions


if __name__ == "__main__":
    generate_questions()

"""
ingest.py
─────────
Run ONCE to parse all PDFs in ./data and build the ChromaDB vector store.

Usage:
    cd app
    python ingest.py

Embeddings: Google text-embedding-004 (free, 768-dim)
"""

import os
import fitz  # PyMuPDF
import chromadb
from tqdm import tqdm

import gemini_client  # noqa: F401 — configures genai
from agents.retriever import GeminiEmbeddingFunction
from config import CHROMA_PATH, DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def ingest() -> None:
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print(
            f"No PDFs found in {DATA_PATH}.\n"
            "Download AIS documents from https://www.araiindia.com/downloads/ais-downloads"
        )
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting ingestion...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = GeminiEmbeddingFunction()

    # Reset collection on each run (to avoid duplicates)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    all_chunks, all_ids, all_metadata = [], [], []

    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        text = extract_text_from_pdf(os.path.join(DATA_PATH, pdf_file))
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{pdf_file}_chunk_{i}")
            all_metadata.append({"source": pdf_file, "chunk_index": i})

    # Embed & store in batches of 100
    batch_size = 100
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding & storing"):
        collection.add(
            documents=all_chunks[i: i + batch_size],
            ids=all_ids[i: i + batch_size],
            metadatas=all_metadata[i: i + batch_size],
        )

    print(f"\nDone. {len(all_chunks)} chunks stored in ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    ingest()

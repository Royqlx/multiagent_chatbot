"""
ingest.py
─────────
Run ONCE to parse all PDFs in ./data and build ChromaDB.
"""

import os
import fitz  # PyMuPDF
import chromadb
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Documents, Embeddings

from config import CHROMA_PATH, DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME


# -----------------------------
# Embedding Function (SINGLE SOURCE OF TRUTH)
# -----------------------------
class SentenceEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()


# -----------------------------
# PDF Utils
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap

    return chunks


# -----------------------------
# Ingestion Pipeline
# -----------------------------
def ingest() -> None:
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {DATA_PATH}")
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting ingestion...")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    ef = SentenceEmbeddingFunction()

    # Reset collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    all_chunks, all_ids, all_metadata = [], [], []

    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        text = extract_text_from_pdf(os.path.join(DATA_PATH, pdf_file))
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{pdf_file}_chunk_{i}")
            all_metadata.append({
                "source": pdf_file,
                "chunk_index": i
            })

    batch_size = 100

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding & storing"):
        collection.add(
            documents=all_chunks[i:i + batch_size],
            ids=all_ids[i:i + batch_size],
            metadatas=all_metadata[i:i + batch_size],
        )

    print(f"Done. {len(all_chunks)} chunks stored in ChromaDB.")
if __name__ == "__main__":
    ingest()
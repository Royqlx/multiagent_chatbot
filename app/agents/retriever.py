"""
RetrieverAgent
──────────────
Objective : Given a user query, return the top-K relevant chunks from ChromaDB.
Input     : query string
Output    : list of dicts — {text, source, score}

Embeddings are generated with Google's text-embedding-004 model (free tier).
ChromaDB uses a custom EmbeddingFunction so the same model handles both
ingestion and retrieval.
"""

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
import google.generativeai as genai  # noqa: F401 — ensures client is configured
import gemini_client  # configures genai on import
from config import CHROMA_PATH, EMBEDDING_MODEL, TOP_K, COLLECTION_NAME


class GeminiEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible wrapper around Gemini text-embedding-004."""

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        import google.generativeai as genai
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=input,
            task_type="retrieval_document",
        )
        return result["embedding"] if isinstance(input, str) else result["embedding"]


class RetrieverAgent:
    """
    Wraps a ChromaDB persistent collection and exposes a `retrieve` method.
    The collection must already exist (run ingest.py first).
    """

    def __init__(self):
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(
            COLLECTION_NAME,
            embedding_function=GeminiEmbeddingFunction(),
        )

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": results["distances"][0][i],
            })
        return chunks

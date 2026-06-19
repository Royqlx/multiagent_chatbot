"""
RetrieverAgent
──────────────
Objective : Given a user query, return the top-K relevant chunks from ChromaDB.
Input     : query string
Output    : list of dicts — {text, source, score}
"""

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, TOP_K, COLLECTION_NAME
import os


# -----------------------------
# Gemini Embedding Function (NEW SDK)
# -----------------------------
class SentenceEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()


# -----------------------------
# Retriever Agent
# -----------------------------
class RetrieverAgent:
    """
    Wraps a ChromaDB persistent collection and exposes a `retrieve` method.
    """

    def __init__(self):
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=SentenceEmbeddingFunction(),
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
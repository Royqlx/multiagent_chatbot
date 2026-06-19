"""
OrchestratorAgent
─────────────────
Objective : Coordinate the full RAG pipeline and manage conversation memory.
Input     : user query string
Output    : dict — {query, answer, retrieved_chunks}

Memory:
  Short-term → self.history (list of {role, content})  — cleared per session
  Long-term  → ChromaDB managed by RetrieverAgent      — persists across runs

NOTE: Gemini's chat API uses "model" (not "assistant") as the assistant role.
The history stored here uses "model" to stay compatible with GeneratorAgent.
"""

from agents.retriever import RetrieverAgent
from agents.generator import GeneratorAgent
from config import TOP_K


class OrchestratorAgent:
    def __init__(self):
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
        self.history: list[dict] = []   # short-term conversation memory

    def chat(self, query: str, k: int = TOP_K) -> dict:
        # 1. Retrieve relevant chunks
        chunks = self.retriever.retrieve(query, k=k)

        # 2. Generate answer (passes history so the model has multi-turn context)
        answer = self.generator.generate(query, chunks, history=self.history)

        # 3. Persist this turn in short-term memory
        self.history.append({"role": "user",  "content": query})
        self.history.append({"role": "model", "content": answer})

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunks,
        }

    def reset_memory(self) -> None:
        """Clear short-term memory to start a fresh conversation."""
        self.history = [] 

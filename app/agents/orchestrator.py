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

from app.config import TOP_K
from .contracts import TraceEvent


class OrchestratorAgent:
    def __init__(self, retriever=None, generator=None, planner=None, critic=None, max_revision_attempts: int = 1):
        if retriever is None:
            from .retriever import RetrieverAgent
            retriever = RetrieverAgent()
        if generator is None:
            from .generator import GeneratorAgent
            generator = GeneratorAgent()
        if planner is None:
            from .planner import PlannerAgent
            planner = PlannerAgent()
        if critic is None:
            from .critic import CriticAgent
            critic = CriticAgent()
        self.retriever = retriever
        self.generator = generator
        self.planner = planner
        self.critic = critic
        self.max_revision_attempts = max_revision_attempts
        self.history: list[dict] = []

    @staticmethod
    def _event(agent: str, decision: str, reason: str, attempt: int) -> dict:
        return TraceEvent(agent, decision, reason, attempt).__dict__

    def chat(self, query: str, k: int = TOP_K) -> dict:
        plan = self.planner.plan(query)
        trace = [self._event("planner", plan.intent, "query_classified", 0)]
        if not plan.requires_retrieval:
            return {"query": query, "answer": "I can only answer questions grounded in the AIS documents.", "retrieved_chunks": [], "trace": trace + [self._event("orchestrator", "reject", "out_of_scope", 0)]}

        search_query = plan.search_query
        for attempt in range(self.max_revision_attempts + 1):
            chunks = self.retriever.retrieve(search_query, k=k)
            trace.append(self._event("retriever", "retrieved", f"{len(chunks)}_chunks", attempt))
            answer = self.generator.generate(query, chunks, history=self.history)
            trace.append(self._event("generator", "drafted", "answer_generated", attempt))
            critique = self.critic.review(query, answer, chunks)
            trace.append(self._event("critic", critique.verdict, ",".join(critique.reasons), attempt))
            if critique.verdict == "accept":
                self.history.extend(({"role": "user", "content": query}, {"role": "model", "content": answer}))
                return {"query": query, "answer": answer, "retrieved_chunks": chunks, "trace": trace, "critique": critique.__dict__}
            if critique.verdict == "reject" or attempt == self.max_revision_attempts:
                break
            search_query = (critique.revised_query or plan.search_query).strip() or plan.search_query

        return {"query": query, "answer": "I cannot provide a reliable answer because there is insufficient evidence in the retrieved documents.", "retrieved_chunks": chunks, "trace": trace + [self._event("orchestrator", "reject", "insufficient_evidence", attempt)], "critique": critique.__dict__}

    def reset_memory(self) -> None:
        """Clear short-term memory to start a fresh conversation."""
        self.history = [] 

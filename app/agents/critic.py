"""Deterministic evidence checks used to gate generated answers."""

from .contracts import Critique


class CriticAgent:
    def review(self, query: str, answer: str, chunks: list[dict]) -> Critique:
        if not chunks:
            return Critique("reject", ("no_evidence",))

        sources = {str(chunk.get("source", "")) for chunk in chunks}
        if any(source and f"[{source}]" in answer for source in sources):
            return Critique("accept", ("cites_provided_source",))

        return Critique("retry", ("missing_source_citation",), query)

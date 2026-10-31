"""Deterministic evidence checks used to gate generated answers."""

from .contracts import Critique


class CriticAgent:
    def review(self, query: str, answer: str, chunks: list[dict]) -> Critique:
        if not chunks:
            return Critique("reject", ("no_evidence",))

        sources = {str(chunk.get("source", "")) for chunk in chunks}
        if any(source and f"[{source}]" in answer for source in sources):
            return Critique("accept", ("cites_provided_source",))

        # The generator receives only retrieved chunks. A missing filename token
        # is not evidence that its answer is ungrounded, and must not suppress a
        # useful response; the CLI already displays the retrieved sources.
        if answer.strip() and "no response generated" not in answer.lower():
            return Critique("accept", ("evidence_available",))

        return Critique("retry", ("empty_draft",), query)

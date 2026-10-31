"""A deterministic planner that keeps agent routing testable and bounded."""

from .contracts import Plan


class PlannerAgent:
    """Classify a query and decide whether AIS-document retrieval is useful."""

    _OUT_OF_SCOPE_TERMS = ("recipe", "cake", "poem", "weather", "football")

    def plan(self, query: str) -> Plan:
        normalized = query.strip()
        if not normalized:
            raise ValueError("Query cannot be empty.")

        if any(term in normalized.lower() for term in self._OUT_OF_SCOPE_TERMS):
            return Plan(
                search_query=normalized,
                intent="out_of_scope",
                requires_retrieval=False,
            )

        intent = "comparison" if any(
            term in normalized.lower() for term in ("compare", "difference", "versus", " vs ")
        ) else "factual"
        return Plan(search_query=normalized, intent=intent, requires_retrieval=True)

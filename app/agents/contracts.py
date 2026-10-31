"""Typed contracts passed between the quality-gated RAG agents."""

from dataclasses import dataclass
from typing import Literal


Intent = Literal["factual", "comparison", "follow_up", "out_of_scope"]


@dataclass(frozen=True)
class Plan:
    search_query: str
    intent: Intent
    requires_retrieval: bool


@dataclass(frozen=True)
class TraceEvent:
    agent: str
    decision: str
    reason: str
    attempt: int


@dataclass(frozen=True)
class Critique:
    verdict: Literal["accept", "retry", "reject"]
    reasons: tuple[str, ...]
    revised_query: str | None = None

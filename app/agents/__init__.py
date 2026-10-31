"""Agent implementations for the AIS chatbot.

Imports stay lazy so lightweight agents and tests do not require optional model
or vector-store dependencies at package-import time.
"""

__all__ = ["GeneratorAgent", "OrchestratorAgent", "RetrieverAgent"]


def __getattr__(name: str):
    if name == "GeneratorAgent":
        from .generator import GeneratorAgent

        return GeneratorAgent
    if name == "OrchestratorAgent":
        from .orchestrator import OrchestratorAgent

        return OrchestratorAgent
    if name == "RetrieverAgent":
        from .retriever import RetrieverAgent

        return RetrieverAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

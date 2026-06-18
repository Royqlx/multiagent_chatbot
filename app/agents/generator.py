"""
GeneratorAgent
──────────────
Objective : Generate a grounded answer using retrieved context + conversation history.
Input     : query string, list of chunk dicts, conversation history list
Output    : answer string

Uses Gemini 2.5 Flash (free tier via google-genai).
Conversation history is formatted as alternating user/model turns.
"""

import google.generativeai as genai
import gemini_client  # noqa: F401 — configures genai on import
from config import LLM_MODEL

SYSTEM_PROMPT = (
    "You are an expert assistant for Automotive Industry Standards (AIS) documents "
    "published by ARAI (Automotive Research Association of India).\n\n"
    "Answer questions strictly based on the provided context. "
    "If the answer is not in the context, say: "
    "\"I could not find this information in the provided documents.\" "
    "Do not hallucinate or add information outside the context."
)


class GeneratorAgent:
    """Generates answers using Gemini 2.5 Flash with multi-turn history support."""

    def __init__(self):
        self._model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

    def generate(self, query: str, chunks: list[dict], history: list[dict] | None = None) -> str:
        """
        Parameters
        ----------
        query   : current user question
        chunks  : retrieved chunks from RetrieverAgent
        history : list of {role, content} dicts from previous turns
                  (role must be "user" or "model")
        """
        history = history or []

        # Build context block from retrieved chunks
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )

        # Convert stored history to Gemini's Content format
        gemini_history = []
        for turn in history:
            gemini_history.append({
                "role": turn["role"],          # "user" or "model"
                "parts": [turn["content"]],
            })

        chat = self._model.start_chat(history=gemini_history)

        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        response = chat.send_message(user_message)
        return response.text

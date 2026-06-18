"""
main.py — CLI entry point for the Multi-Agent RAG Chatbot.

Usage:
    cd app
    python ../main.py
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from agents.orchestrator import OrchestratorAgent


def main() -> None:
    print("=" * 60)
    print("  Multi-Agent RAG Chatbot — AIS Documents  (Gemini 2.5 Flash)")
    print("  Commands: 'quit' to exit | 'reset' to clear chat history")
    print("=" * 60)

    bot = OrchestratorAgent()

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye.")
            break
        if query.lower() == "reset":
            bot.reset_memory()
            print("Chat history cleared.")
            continue

        result = bot.chat(query)

        print(f"\nAssistant: {result['answer']}")
        sources = ", ".join(sorted({c["source"] for c in result["retrieved_chunks"]}))
        print(f"\n[Sources: {sources}]")


if __name__ == "__main__":
    main()

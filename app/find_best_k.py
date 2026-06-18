"""
find_best_k.py
──────────────
Finds the optimal K by evaluating Precision@K, Recall@K, and MRR at K = 3, 5, 7, 10.
Run AFTER generate_questions.py.

Usage:
    cd app
    python find_best_k.py
"""

import json
import os
import pandas as pd
from agents.orchestrator import OrchestratorAgent

K_VALUES = [3, 5, 7, 10]
STOPWORDS = {"the", "a", "an", "is", "in", "of", "to", "and", "or"}


def evaluate_at_k(qa_pairs: list[dict], k: int) -> dict:
    bot = OrchestratorAgent()
    precision_scores, recall_scores, mrr_scores = [], [], []

    for qa in qa_pairs:
        bot.reset_memory()
        result = bot.chat(qa["question"], k=k)
        keywords = set(qa.get("answer", "").lower().split()) - STOPWORDS
        chunks = result["retrieved_chunks"]

        relevant_flags = [
            1 if len(keywords & set(c["text"].lower().split())) >= 3 else 0
            for c in chunks
        ]
        retrieved_relevant = sum(relevant_flags)

        precision_scores.append(retrieved_relevant / k if k else 0)
        recall_scores.append(min(retrieved_relevant, 1.0))  # assume 1 relevant doc per query

        rr = next((1 / rank for rank, flag in enumerate(relevant_flags, 1) if flag), 0)
        mrr_scores.append(rr)

    n = len(qa_pairs)
    return {
        "K":            k,
        "Precision@K":  round(sum(precision_scores) / n, 4),
        "Recall@K":     round(sum(recall_scores) / n, 4),
        "MRR":          round(sum(mrr_scores) / n, 4),
    }


def main() -> None:
    with open("eval/questions.json") as f:
        qa_pairs = json.load(f)

    results = []
    for k in K_VALUES:
        print(f"Evaluating K={k}...")
        scores = evaluate_at_k(qa_pairs, k)
        results.append(scores)
        print(f"  Precision@{k}: {scores['Precision@K']}  "
              f"Recall@{k}: {scores['Recall@K']}  MRR: {scores['MRR']}")

    os.makedirs("eval", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("eval/k_selection.csv", index=False)
    print("\nSaved to eval/k_selection.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

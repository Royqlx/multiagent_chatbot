"""Credential-free evaluation helpers for retrieval and agent outcomes."""

from math import log2


def _identifier(chunk: dict) -> tuple[str, int | None]:
    return str(chunk.get("source", "")), chunk.get("chunk_index")


def evaluate_retrieval(cases: list[dict], retrieve, k: int) -> dict[str, float]:
    recalls, reciprocal_ranks, ndcgs = [], [], []
    for case in cases:
        expected = {tuple(item) for item in case["expected_chunks"]}
        retrieved = retrieve(case["question"], k)[:k]
        flags = [_identifier(chunk) in expected for chunk in retrieved]
        recalls.append(bool(any(flags)))
        reciprocal_ranks.append(next((1 / rank for rank, flag in enumerate(flags, 1) if flag), 0.0))
        dcg = sum(1 / log2(rank + 1) for rank, flag in enumerate(flags, 1) if flag)
        ideal_count = min(len(expected), k)
        ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
    count = len(cases)
    return {"recall_at_k": sum(recalls) / count, "mrr": sum(reciprocal_ranks) / count, "ndcg_at_k": sum(ndcgs) / count}


def evaluate_agent_outcomes(cases: list[dict], chat) -> dict[str, float]:
    citation_checks, refusal_checks = [], []
    for case in cases:
        result = chat(case["question"])
        sources = {chunk.get("source", "") for chunk in result.get("retrieved_chunks", [])}
        answer = result["answer"]
        citation_checks.append(not sources or any(f"[{source}]" in answer for source in sources))
        if case.get("expect_refusal"):
            refusal_checks.append("AIS documents" in answer or "insufficient evidence" in answer.lower())
    return {"citation_validity": sum(citation_checks) / len(citation_checks), "refusal_accuracy": sum(refusal_checks) / len(refusal_checks) if refusal_checks else 1.0}

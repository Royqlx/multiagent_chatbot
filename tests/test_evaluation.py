from app.evaluation import evaluate_agent_outcomes, evaluate_retrieval


def test_evaluate_retrieval_reports_perfect_metrics_for_ranked_match():
    cases = [{"question": "scope", "expected_chunks": [["AIS-004.pdf", 0]]}]

    metrics = evaluate_retrieval(cases, lambda _query, _k: [{"source": "AIS-004.pdf", "chunk_index": 0}], 3)

    assert metrics == {"recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0}


def test_evaluate_agent_outcomes_checks_citations_and_refusals():
    cases = [{"question": "recipe", "expect_refusal": True}]
    metrics = evaluate_agent_outcomes(cases, lambda _query: {"answer": "I can only answer questions grounded in the AIS documents.", "retrieved_chunks": []})

    assert metrics == {"citation_validity": 1.0, "refusal_accuracy": 1.0}

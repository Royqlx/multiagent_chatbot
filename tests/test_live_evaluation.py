from app.live_evaluation import build_records, summarize_scores


class FakeBot:
    def reset_memory(self):
        pass

    def chat(self, question):
        return {
            "answer": "AIS-206 applies to category M vehicles [AIS-206.pdf].",
            "retrieved_chunks": [
                {"text": "AIS-206 applies to category M vehicles.", "source": "AIS-206.pdf", "chunk_index": 2}
            ],
            "trace": [{"agent": "critic", "decision": "accept", "reason": "evidence_available", "attempt": 0}],
        }


def test_build_records_runs_each_supported_case_with_fresh_memory():
    cases = [
        {"id": "ais-206", "question": "What is AIS-206?", "reference_answer": "It applies to category M vehicles."},
        {"id": "unsupported", "question": "Write a recipe", "expect_refusal": True},
    ]

    records = build_records(cases, FakeBot())

    assert len(records) == 2
    assert records[0]["contexts"] == ["AIS-206 applies to category M vehicles."]
    assert records[0]["reference"] == "It applies to category M vehicles."
    assert records[0]["source_ids"] == ["AIS-206.pdf#2"]
    assert records[1]["include_in_ragas"] is False


def test_summarize_scores_averages_numeric_ragas_columns():
    summary = summarize_scores([
        {"faithfulness": 0.8, "answer_relevancy": 0.6},
        {"faithfulness": 1.0, "answer_relevancy": 0.8},
    ])

    assert summary == {"faithfulness": 0.9, "answer_relevancy": 0.7}

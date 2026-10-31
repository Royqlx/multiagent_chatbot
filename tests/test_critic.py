from app.agents.critic import CriticAgent


def test_review_rejects_when_no_evidence_was_retrieved():
    critique = CriticAgent().review("What is AIS-004?", "AIS-004 applies.", [])

    assert critique.verdict == "reject"
    assert "no_evidence" in critique.reasons


def test_review_accepts_substantive_answer_when_evidence_is_available_without_filename_citation():
    critique = CriticAgent().review(
        "What is AIS-004?",
        "AIS-004 applies to all vehicles.",
        [{"source": "AIS-004.pdf", "text": "Scope information."}],
    )

    assert critique.verdict == "accept"
    assert "evidence_available" in critique.reasons


def test_review_accepts_answer_that_cites_a_provided_source():
    critique = CriticAgent().review(
        "What is AIS-004?",
        "The scope is stated in [AIS-004.pdf].",
        [{"source": "AIS-004.pdf", "text": "Scope information."}],
    )

    assert critique.verdict == "accept"

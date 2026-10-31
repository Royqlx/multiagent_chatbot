import importlib

import pytest

from app.agents.planner import PlannerAgent


def test_agents_package_imports_from_project_root():
    assert importlib.import_module("app.agents") is not None


def test_plan_rejects_blank_query():
    with pytest.raises(ValueError, match="empty"):
        PlannerAgent().plan("   ")


def test_plan_marks_ais_question_for_retrieval():
    plan = PlannerAgent().plan("What is the scope of AIS-004?")

    assert plan.requires_retrieval is True
    assert plan.search_query == "What is the scope of AIS-004?"
    assert plan.intent == "factual"


def test_plan_marks_clearly_unrelated_query_out_of_scope():
    plan = PlannerAgent().plan("Write me a chocolate cake recipe")

    assert plan.requires_retrieval is False
    assert plan.intent == "out_of_scope"

from app.agents.contracts import Critique, Plan
from app.agents.orchestrator import OrchestratorAgent


class FakePlanner:
    def __init__(self, plan):
        self.plan_result = plan

    def plan(self, query):
        return self.plan_result


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k):
        self.queries.append(query)
        return [{"source": "AIS-004.pdf", "chunk_index": 0, "text": "Scope"}]


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, query, chunks, history=None):
        self.calls += 1
        return "The scope is stated in [AIS-004.pdf]."


class FakeCritic:
    def __init__(self, critiques):
        self.critiques = iter(critiques)

    def review(self, query, answer, chunks):
        return next(self.critiques)


def test_chat_returns_accept_trace():
    bot = OrchestratorAgent(
        planner=FakePlanner(Plan("scope", "factual", True)),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        critic=FakeCritic([Critique("accept", ("grounded",))]),
    )

    result = bot.chat("What is the scope?")

    assert result["answer"] == "The scope is stated in [AIS-004.pdf]."
    assert [event["agent"] for event in result["trace"]] == [
        "planner", "retriever", "generator", "critic"
    ]


def test_chat_retries_once_with_critic_query():
    retriever = FakeRetriever()
    bot = OrchestratorAgent(
        planner=FakePlanner(Plan("scope", "factual", True)),
        retriever=retriever,
        generator=FakeGenerator(),
        critic=FakeCritic([
            Critique("retry", ("missing_evidence",), "revised scope"),
            Critique("accept", ("grounded",)),
        ]),
    )

    result = bot.chat("What is the scope?")

    assert retriever.queries == ["scope", "revised scope"]
    assert result["trace"][-1]["decision"] == "accept"


def test_chat_rejects_after_retry_limit():
    bot = OrchestratorAgent(
        planner=FakePlanner(Plan("scope", "factual", True)),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        critic=FakeCritic([Critique("retry", ("missing_evidence",))] * 2),
    )

    result = bot.chat("What is the scope?")

    assert "insufficient evidence" in result["answer"].lower()
    assert result["trace"][-1]["decision"] == "reject"


def test_chat_rejects_out_of_scope_without_retrieval():
    retriever = FakeRetriever()
    bot = OrchestratorAgent(
        planner=FakePlanner(Plan("cake", "out_of_scope", False)),
        retriever=retriever,
        generator=FakeGenerator(),
        critic=FakeCritic([]),
    )

    result = bot.chat("Write a cake recipe")

    assert retriever.queries == []
    assert "AIS documents" in result["answer"]

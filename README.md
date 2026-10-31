# Quality-Gated Multi-Agent RAG Chatbot for AIS Documents

An evidence-first chatbot for Automotive Industry Standards (AIS) PDFs from ARAI. It indexes local documents in ChromaDB and uses Gemini to answer only from retrieved evidence.

```text
Question → Planner → Retriever → Generator → Critic
                                      ↑          │
                                      └─ one retry┘
```

The critic accepts an answer that cites available sources, requests one improved retrieval attempt, or returns an insufficient-evidence response. Each response includes a structured trace for inspection.

## Prerequisites

- Python 3.11 or 3.12
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- One or more AIS PDF documents

## Quick start (Windows PowerShell)

Run these commands from the project root:

```powershell

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install "sentence-transformers>=5,<6" "transformers>=4.41,<5" "tokenizers>=0.22,<0.23"

Set-Content -Path .env -Value "GOOGLE_API_KEY=your_gemini_api_key"

New-Item -ItemType Directory -Force -Path app\data | Out-Null
# Copy one or more AIS PDF files into app\data

cd app
python ingest.py
cd ..

python main.py
```

The first ingestion downloads the `all-MiniLM-L6-v2` embedding model. ChromaDB is written to `chroma/` at the project root. Re-run ingestion after adding or replacing PDFs; it rebuilds the collection.

> Do not commit `.env`, `app/data/`, or `chroma/`.

## Chat commands

```text
You: What is the scope of AIS-004?
Assistant: ...

[Sources: AIS-004.pdf]
```

- Type an AIS-related question to chat.
- Type `reset` to clear short-term conversation history.
- Type `quit` to exit.

If a request is outside the AIS corpus or the evidence is inadequate, the system refuses transparently instead of fabricating an answer.

## Agent roles

| Agent | Responsibility |
|---|---|
| `PlannerAgent` | Classifies a request and determines whether retrieval is needed. |
| `RetrieverAgent` | Retrieves the top-K relevant document chunks from ChromaDB. |
| `GeneratorAgent` | Produces a source-constrained Gemini answer from retrieved chunks. |
| `CriticAgent` | Checks that an answer cites supplied evidence and accepts, retries, or rejects it. |
| `OrchestratorAgent` | Coordinates the graph, preserves conversation memory, enforces the one-retry limit, and returns the trace. |

## Deterministic evaluation

The project includes credential-free tests for planner decisions, critic outcomes, retry limits, source handling, and retrieval metrics. They do not contact Gemini, ChromaDB, or download models.

```powershell
python -m pytest tests -q
```

`app.evaluation` calculates Recall@K, MRR, nDCG@K, citation validity, and expected-refusal accuracy. The old Ragas-based scripts are exploratory utilities and are not part of the deterministic suite.

## Project layout

```text
app/
  agents/
    planner.py        # request planning
    retriever.py      # ChromaDB search
    generator.py      # Gemini drafting
    critic.py         # evidence/citation gate
    orchestrator.py   # bounded agent graph
  evaluation.py       # pure deterministic metrics
  ingest.py           # PDF ingestion
  data/               # add PDFs here (created locally)
tests/                # credential-free tests and fixtures
main.py               # CLI entry point
```

## Configuration and limitations

Update `app/config.py` to change the model, retrieval `TOP_K`, chunk size, or overlap. The default generation model is `gemini-2.5-flash`; embeddings use local `all-MiniLM-L6-v2`.

- The CLI prints sources; programmatic callers can inspect the full `trace` returned by `OrchestratorAgent.chat()`.
- Source attribution is document-level today. Adding page-level metadata during ingestion is the next useful improvement.
- Ragas/LLM-as-a-judge evaluation remains optional because it needs external model calls and version-compatible dependencies.

# Multi-Agent RAG Chatbot — Gemini 2.5 Flash Edition

A multi-agent Retrieval-Augmented Generation (RAG) chatbot for AIS (Automotive Industry Standards) documents, powered entirely by **free Google Gemini APIs** — no OpenAI account or billing required.

---

## Architecture

```
User Query
    │
    ▼
OrchestratorAgent          ← coordinates pipeline + manages memory
    ├── RetrieverAgent     ← Gemini text-embedding-004 → ChromaDB
    ├── GeneratorAgent     ← Gemini 2.5 Flash (chat with history)
    └── EvaluatorAgent     ← Ragas metrics via LangChain-Google wrapper
```

### Agent responsibilities

| Agent | Input | Output |
|---|---|---|
| **OrchestratorAgent** | user query | answer dict + sources |
| **RetrieverAgent** | query string | top-K chunks from ChromaDB |
| **GeneratorAgent** | query + chunks + history | grounded answer string |
| **EvaluatorAgent** | questions JSON path | Ragas metrics DataFrame + CSV |

---

## Free-tier models used

| Purpose | Model |
|---|---|
| LLM generation | `gemini-2.5-flash` |
| Embeddings | `models/text-embedding-004` (768-dim) |

Get your free API key at **https://aistudio.google.com/app/apikey**

---

## Project structure

```
multiagent_chatbot/
├── main.py                   # CLI entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── data/                     # ← drop your AIS PDFs here
├── chroma_db/                # auto-created by ingest.py
└── app/
    ├── config.py             # all settings in one place
    ├── ingest.py             # PDF → ChromaDB pipeline
    ├── generate_questions.py # synthetic Q&A generation
    ├── find_best_k.py        # Precision/Recall/MRR sweep over K values
    └── agents/
        ├── __init__.py
        ├── orchestrator.py   # pipeline coordinator + short-term memory
        ├── retriever.py      # ChromaDB + Gemini embeddings
        ├── generator.py      # Gemini 2.5 Flash answer generation
        └── evaluator.py      # Ragas evaluation (faithfulness, relevance, precision, recall)
```

---

## Setup

```bash
# 1. Clone / unzip project
cd gemini_chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and paste your GOOGLE_API_KEY

# 5. Add PDFs to the data/ folder
#    Download AIS docs from: https://www.araiindia.com/downloads/ais-downloads

# 6. Ingest documents (run once, or again after adding new PDFs)
cd app
python ingest.py

# 7. Start chatting
cd ..
python main.py
```

---

## Evaluation workflow

```bash
cd app

# Generate 50 synthetic Q&A pairs from your documents
python generate_questions.py     # → eval/questions.json

# Find the optimal K for retrieval
python find_best_k.py            # → eval/k_selection.csv

# Run full Ragas evaluation
python -c "from agents.evaluator import EvaluatorAgent; EvaluatorAgent().run_evaluation()"
# → eval/results.csv
```

---

## Configuration (`app/config.py`)

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model for generation |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini embedding model |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Word overlap between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |

---

## Key differences from the OpenAI version

| | OpenAI version | This version |
|---|---|---|
| LLM | `gpt-4o` / `gpt-4o-mini` | `gemini-2.5-flash` (free) |
| Embeddings | `text-embedding-3-small` | `text-embedding-004` (free) |
| Client | `openai.OpenAI` | `google-generativeai` |
| History role | `"assistant"` | `"model"` (Gemini convention) |
| Ragas LLM | OpenAI default | LangChain `ChatGoogleGenerativeAI` |
| Cost | Paid API | Free tier |

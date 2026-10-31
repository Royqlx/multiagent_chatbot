"""Run optional Ragas scoring against the live quality-gated AIS chatbot."""

import argparse
import csv
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from statistics import mean


RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def build_records(cases: list[dict], bot) -> list[dict]:
    """Run benchmark cases with isolated chat memory and capture evidence."""
    records = []
    for case in cases:
        bot.reset_memory()
        result = bot.chat(case["question"])
        chunks = result.get("retrieved_chunks", [])
        records.append({
            "id": case.get("id", case["question"]),
            "question": case["question"],
            "answer": result["answer"],
            "contexts": [chunk["text"] for chunk in chunks],
            "reference": case.get("reference_answer", ""),
            "source_ids": [f"{chunk.get('source', '')}#{chunk.get('chunk_index', '')}" for chunk in chunks],
            "trace": result.get("trace", []),
            "include_in_ragas": not case.get("expect_refusal", False) and bool(case.get("reference_answer")),
        })
    return records


def summarize_scores(rows: list[dict]) -> dict[str, float]:
    """Return rounded means for metric columns present in scored rows."""
    summary = {}
    for metric in RAGAS_METRICS:
        values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (float, int))]
        if values:
            summary[metric] = round(mean(values), 4)
    return summary


def score_with_ragas(records: list[dict]) -> list[dict]:
    """Score supported records with Ragas; imports stay optional for normal chat."""
    scored = [record for record in records if record["include_in_ragas"]]
    if not scored:
        return []

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ModuleNotFoundError as error:
        # Ragas 0.4.x imports this optional Vertex integration eagerly. The
        # current LangChain community package no longer ships it, and this
        # evaluator uses Gemini instead. Provide the unused type solely so the
        # Ragas import can complete; no Vertex credentials or calls are made.
        if error.name != "langchain_community.chat_models.vertexai":
            raise RuntimeError(
                "Ragas evaluation dependencies are incompatible or missing. "
                "Install the evaluation dependency set before running this command."
            ) from error
        vertex_module = types.ModuleType(error.name)
        vertex_module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[error.name] = vertex_module
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError as error:
        raise RuntimeError(
            "Ragas evaluation dependencies are incompatible or missing. "
            "Install the evaluation dependency set before running this command."
        ) from error

    from app.config import EMBEDDING_MODEL, GOOGLE_API_KEY, LLM_MODEL
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is required for live Ragas evaluation.")

    dataset = Dataset.from_list([
        {"question": item["question"], "answer": item["answer"], "contexts": item["contexts"], "ground_truth": item["reference"]}
        for item in scored
    ])
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    ).to_pandas().to_dict(orient="records")
    return scores


def _load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"] if isinstance(payload, dict) else payload


def _write_report(output_dir: Path, records: list[dict], scored_rows: list[dict]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"ragas-report-{timestamp}.json"
    csv_path = output_dir / f"ragas-scores-{timestamp}.csv"
    json_path.write_text(json.dumps({"records": records, "ragas_scores": scored_rows, "summary": summarize_scores(scored_rows)}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *RAGAS_METRICS])
        writer.writeheader()
        for record, score in zip((item for item in records if item["include_in_ragas"]), scored_rows):
            writer.writerow({"id": record["id"], **{metric: score.get(metric) for metric in RAGAS_METRICS}})
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Ragas evaluation for AIS chatbot benchmarks.")
    parser.add_argument("--benchmark", required=True, type=Path, help="JSON list of reviewed benchmark cases")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/reports"))
    args = parser.parse_args()

    if not args.benchmark.is_file():
        raise SystemExit(f"Benchmark file not found: {args.benchmark}")
    from app.agents.orchestrator import OrchestratorAgent

    records = build_records(_load_cases(args.benchmark), OrchestratorAgent())
    scored_rows = score_with_ragas(records)
    json_path, csv_path = _write_report(args.output_dir, records, scored_rows)
    print(f"Scored {len(scored_rows)} benchmark case(s).")
    print(json.dumps(summarize_scores(scored_rows), indent=2))
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()

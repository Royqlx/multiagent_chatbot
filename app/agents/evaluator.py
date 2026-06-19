"""
EvaluatorAgent
──────────────
Objective : Evaluate chatbot quality using Ragas metrics.
Input     : path to questions JSON  (format: [{question, answer}, ...])
Output    : results DataFrame + saved CSV

Metrics
-------
- Faithfulness       : Is the answer grounded in the retrieved context?
- Answer Relevance   : Does the answer address the question?
- Context Precision  : Of retrieved chunks, what fraction are relevant?
- Context Recall     : Of all relevant info, what fraction was retrieved?

Ragas wraps calls to an LLM internally. We configure it to use Gemini 2.5 Flash
via LangChain's ChatGoogleGenerativeAI wrapper so Ragas works without OpenAI.
"""

import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from agents.orchestrator import OrchestratorAgent
from config import TOP_K, LLM_MODEL, EMBEDDING_MODEL, GOOGLE_API_KEY


def _ragas_llm():
    """LangChain-wrapped Gemini LLM for Ragas internals."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )


def _ragas_embeddings():
    """LangChain-wrapped Gemini embeddings for Ragas internals."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


class EvaluatorAgent:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()

    def run_evaluation(
        self,
        questions_path: str = "eval/questions.json",
        output_path: str = "eval/results.csv",
    ) -> pd.DataFrame:
        with open(questions_path) as f:
            qa_pairs = json.load(f)

        print(f"Running evaluation on {len(qa_pairs)} questions...")

        records = []
        for qa in qa_pairs:
            self.orchestrator.reset_memory()
            result = self.orchestrator.chat(qa["question"], k=TOP_K)
            records.append({
                "question":    qa["question"],
                "answer":      result["answer"],
                "contexts":    [c["text"] for c in result["retrieved_chunks"]],
                "ground_truth": qa.get("answer", ""),
            })

        dataset = Dataset.from_list(records)

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=_ragas_llm(),
            embeddings=_ragas_embeddings(),
        )

        results_df = scores.to_pandas()
        results_df.to_csv(output_path, index=False)

        print(f"\nEvaluation complete. Results saved to {output_path}")
        cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        print(results_df[cols].mean())

        return results_df 

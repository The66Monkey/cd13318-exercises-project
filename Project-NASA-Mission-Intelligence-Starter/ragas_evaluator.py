from typing import Dict, List
from ragas import evaluate
from ragas.metrics import (
    response_relevancy,
    faithfulness,
)
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from statistics import mean


def get_local_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    embed = LangchainEmbeddingsWrapper(get_local_embeddings())

    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })

    result = evaluate(
        data,
        metrics=[response_relevancy, faithfulness],
        embeddings=embed,
    )

    row = result.to_pandas().iloc[0]
    return {
        "response_relevancy": float(row["response_relevancy"]),
        "faithfulness": float(row["faithfulness"]),
    }


def run_batch_evaluation(
    questions_file: str,
    rag_retriever,
    llm_generator
):
    with open(questions_file, "r") as f:
        questions = [q.strip() for q in f.readlines() if q.strip()]

    results = []
    print("\n=== Running Batch Evaluation ===\n")

    for q in questions:
        print(f"Question: {q}")

        contexts = rag_retriever(q)
        if not contexts:
            print("  ⚠️ No contexts retrieved — skipping.")
            print("-" * 40)
            continue

        answer = llm_generator(q, contexts)
        if not answer:
            print("  ⚠️ No answer generated — skipping.")
            print("-" * 40)
            continue

        scores = evaluate_response_quality(q, answer, contexts)
        results.append(scores)

        print("Scores:")
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")
        print("-" * 40)

    if not results:
        print("\n❌ No valid evaluations were produced.")
        return {}

    print("\n=== Aggregate Metrics ===")
    aggregate = {
        metric: mean(r[metric] for r in results)
        for metric in results[0].keys()
    }

    for k, v in aggregate.items():
        print(f"{k}: {v:.4f}")

    return aggregate

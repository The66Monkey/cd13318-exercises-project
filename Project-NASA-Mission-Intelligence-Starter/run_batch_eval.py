from simple_evaluator import evaluate_response
from statistics import mean
from llm_client import generate_response


def llm_generator(question, contexts):
    trimmed = [c[:300] for c in contexts]
    context_str = "\n\n".join(trimmed)

    if len(context_str) > 2000:
        context_str = context_str[:2000]

    answer = generate_response(
        openai_key="not-needed",
        user_message=question,
        context=context_str,
        conversation_history=[],
        model="local-model"
    )

    if not answer or not answer.strip():
        return "No answer generated."

    return answer


def run_batch_evaluation(questions_file, rag_retriever, llm_generator):
    with open(questions_file) as f:
        questions = [q.strip() for q in f if q.strip()]

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

        scores = evaluate_response(q, answer, contexts)
        results.append(scores)

        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")
        print("-" * 40)

    print("\n=== Aggregate Metrics ===")
    agg = {k: mean(r[k] for r in results) for k in results[0]}
    for k, v in agg.items():
        print(f"{k}: {v:.4f}")

    return agg


# ---------------------------------------------------------
# RAG RETRIEVER (correct version)
# ---------------------------------------------------------
if __name__ == "__main__":
    from rag_client import initialize_rag_system, retrieve_documents

    # Load your Chroma collection ONCE
    collection = initialize_rag_system(
        "./chroma_db_openai",
        "nasa_space_missions_text"
    )

    def rag_retriever(question):
        results = retrieve_documents(collection, question, n_results=3)
        if not results or not results.get("documents"):
            return []
        return results["documents"][0]

    run_batch_evaluation("evaluation_dataset.txt", rag_retriever, llm_generator)

from typing import List, Dict
from statistics import mean
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def response_relevancy(question: str, answer: str) -> float:
    q_emb = embedder.embed_query(question)
    a_emb = embedder.embed_query(answer)
    return cosine(q_emb, a_emb)


def faithfulness(answer: str, contexts: List[str]) -> float:
    a_emb = embedder.embed_query(answer)
    ctx_embs = [embedder.embed_query(c) for c in contexts]
    sims = [cosine(a_emb, c) for c in ctx_embs]
    return float(max(sims))  # best‑match context


def evaluate_response(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    return {
        "response_relevancy": response_relevancy(question, answer),
        "faithfulness": faithfulness(answer, contexts),
    }

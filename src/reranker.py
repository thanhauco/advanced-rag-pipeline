from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import DocumentChunk
from .retrieval import RetrievalResult


@dataclass
class RerankedResult:
    chunk: DocumentChunk
    score: float


class CrossEncoderReranker:
    """Lightweight proxy for rerankers such as Cohere Rerank."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def rerank(
        self, query: str, candidates: List[RetrievalResult], top_k: int = 4
    ) -> List[RerankedResult]:
        if not candidates:
            return []

        documents = [result.chunk.text for result in candidates]
        matrix = self.vectorizer.fit_transform(documents + [query])
        query_vec = matrix[-1]
        doc_vecs = matrix[:-1]
        similarities = cosine_similarity(query_vec, doc_vecs)[0]

        reranked: List[RerankedResult] = []
        for result, sim in zip(candidates, similarities):
            blended = 0.5 * result.score + 0.5 * sim
            reranked.append(RerankedResult(chunk=result.chunk, score=blended))
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

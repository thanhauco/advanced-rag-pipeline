from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .indexing import HybridIndexer
from .models import DocumentChunk
from .query_processor import QueryBundle


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float


class HybridRetriever:
    """Combines TF-IDF similarity with lightweight lexical overlap."""

    def __init__(self, indexer: HybridIndexer) -> None:
        self.indexer = indexer
        self.lexical_vectorizer = CountVectorizer(stop_words="english")
        self._fit_lexical()

    def _fit_lexical(self) -> None:
        corpus = [chunk.text for chunk in self.indexer.chunks]
        if corpus:
            self.lexical_matrix = self.lexical_vectorizer.fit_transform(corpus)
        else:
            self.lexical_matrix = None

    def _lexical_score(self, query: str) -> List[float]:
        if self.lexical_matrix is None:
            return [0.0] * len(self.indexer.chunks)
        query_vec = self.lexical_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.lexical_matrix)[0]
        return scores.tolist()

    def retrieve(self, bundle: QueryBundle, top_k: int = 6) -> List[RetrievalResult]:
        tfidf_candidates = self.indexer.batch_search(bundle.rewrites, top_k=top_k * 2)
        lexical_scores = self._lexical_score(bundle.original)
        combined = []
        for chunk, tfidf_score in tfidf_candidates:
            lex_score = lexical_scores[self.indexer.chunks.index(chunk)]
            score = 0.7 * tfidf_score + 0.3 * lex_score
            combined.append(RetrievalResult(chunk=chunk, score=score))
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:top_k]


def aggregate_context(chunks: Sequence[DocumentChunk]) -> str:
    return "\n".join(f"- {chunk.text}" for chunk in chunks)

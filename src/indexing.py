from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Document, DocumentChunk


@dataclass
class SemanticChunker:
    chunk_size: int = 80
    overlap: int = 20

    def chunk(self, document: Document) -> List[DocumentChunk]:
        tokens = document.content.split()
        chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size - self.overlap)
        for idx in range(0, len(tokens), step):
            window = tokens[idx : idx + self.chunk_size]
            if not window:
                continue
            chunk_text = " ".join(window)
            chunk_id = f"{document.id}-chunk-{len(chunks)}"
            metadata = {**document.metadata, "source_title": document.title}
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )
        return chunks


class HybridIndexer:
    """TF-IDF indexer approximating dense + lexical scoring."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunks: List[DocumentChunk] = []
        self.matrix = None

    def build(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        corpus = [chunk.text for chunk in self.chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> List[tuple[DocumentChunk, float]]:
        if not self.chunks or self.matrix is None:
            raise RuntimeError("Index has not been built.")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]

    def batch_search(
        self, queries: Iterable[str], top_k: int = 5
    ) -> List[tuple[DocumentChunk, float]]:
        aggregated: List[tuple[DocumentChunk, float]] = []
        seen = {}
        for query in queries:
            for chunk, score in self.search(query, top_k=top_k):
                if chunk.chunk_id not in seen or score > seen[chunk.chunk_id]:
                    seen[chunk.chunk_id] = score
        for chunk in self.chunks:
            if chunk.chunk_id in seen:
                aggregated.append((chunk, seen[chunk.chunk_id]))
        aggregated.sort(key=lambda pair: pair[1], reverse=True)
        return aggregated[:top_k]

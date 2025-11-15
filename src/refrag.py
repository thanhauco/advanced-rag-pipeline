from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .models import DocumentChunk


@dataclass
class MicroChunk:
    chunk_id: str
    text: str


class RefragCompressor:
    """Splits chunks into fixed-size token windows."""

    def __init__(self, micro_size: int = 16) -> None:
        self.micro_size = micro_size

    def compress(self, chunk: DocumentChunk) -> List[MicroChunk]:
        tokens = chunk.text.split()
        micros: List[MicroChunk] = []
        for idx in range(0, len(tokens), self.micro_size):
            window = tokens[idx : idx + self.micro_size]
            if not window:
                continue
            micro_id = f"{chunk.chunk_id}-micro-{len(micros)}"
            micros.append(MicroChunk(chunk_id=micro_id, text=" ".join(window)))
        return micros

    def compress_documents(self, chunks: Sequence[DocumentChunk]) -> List[MicroChunk]:
        micro_chunks: List[MicroChunk] = []
        for chunk in chunks:
            micro_chunks.extend(self.compress(chunk))
        return micro_chunks


class RefragSelector:
    """Scores micro-chunks using a lightweight heuristic (proxy for RL policy)."""

    def __init__(self, retain_ratio: float = 0.3) -> None:
        self.retain_ratio = retain_ratio

    def select(self, query: str, micros: Sequence[MicroChunk]) -> List[MicroChunk]:
        if not micros:
            return []
        query_tokens = set(query.lower().split())
        scored = []
        for micro in micros:
            overlap = sum(
                1 for token in micro.text.lower().split() if token in query_tokens
            )
            scored.append((micro, overlap))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        budget = max(1, int(len(micros) * self.retain_ratio))
        return [micro for micro, _ in scored[:budget]]


class RefragDecoder:
    def decode(self, selected: Iterable[MicroChunk]) -> str:
        return " ".join(micro.text for micro in selected)

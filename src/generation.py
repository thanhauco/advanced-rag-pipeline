from __future__ import annotations

from typing import Sequence

from .models import DocumentChunk
from .retrieval import aggregate_context


class TemplateGenerator:
    """Simple deterministic generator substituting retrieved context."""

    def generate(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
        refrag_summary: str | None = None,
    ) -> str:
        context = aggregate_context(chunks)
        refrag_section = (
            f"\n\nREFRAG-selected highlights:\n{refrag_summary}"
            if refrag_summary
            else ""
        )
        return (
            f"Question: {query}\n\n"
            "Context extracted from the knowledge base:\n"
            f"{context}\n"
            f"{refrag_section}\n\n"
            "Answer Outline:\n"
            "- Reference the indexing/query/retrieval stages from the context.\n"
            "- Emphasize reranking benefits and REFRAG efficiency gains.\n"
            "- Close with evaluation recommendations."
        )

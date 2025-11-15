from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


def _generate_synonym_variants(query: str, synonyms: Dict[str, List[str]]) -> List[str]:
    variants = set()
    lower = query.lower()
    for token, replacements in synonyms.items():
        if token in lower:
            for repl in replacements:
                variants.add(re.sub(token, repl, lower))
    return [query] + sorted(variants)


@dataclass
class QueryBundle:
    """Container for processed queries."""

    original: str
    rewrites: List[str]
    hypothetical: str
    decomposed: List[str]


@dataclass
class QueryProcessor:
    """Handles expansions such as HyDE and decomposition."""

    synonyms: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "rag": ["retrieval augmented generation", "retrieval grounded generation"],
            "reranking": ["re-ranking", "reorder results"],
            "pipeline": ["workflow", "stack"],
        }
    )

    def expand_synonyms(self, query: str) -> List[str]:
        return _generate_synonym_variants(query, self.synonyms)

    def generate_hypothetical(self, query: str) -> str:
        return (
            f"Hypothetical answer for '{query}': "
            "The pipeline likely needs indexing, retrieval, reranking, generation, and evaluation steps."
        )

    def decompose(self, query: str) -> List[str]:
        """Naively split multi-part questions."""
        for delimiter in (" and ", ";", ",", " & "):
            if delimiter in query.lower():
                return [part.strip() for part in query.split(delimiter) if part.strip()]
        return [query]

    def process(self, query: str) -> QueryBundle:
        rewrites = self.expand_synonyms(query)
        hypothetical = self.generate_hypothetical(query)
        decomposed = self.decompose(query)
        return QueryBundle(
            original=query,
            rewrites=rewrites,
            hypothetical=hypothetical,
            decomposed=decomposed,
        )


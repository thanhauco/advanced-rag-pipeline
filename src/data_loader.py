from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import Document


def load_documents(path: str | Path) -> List[Document]:
    """Load documents from a JSON file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(
            id=item["id"],
            title=item.get("title", item["id"]),
            content=item["content"],
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]


def iter_documents(path: str | Path) -> Iterable[Document]:
    """Yield documents lazily for streaming pipelines."""
    for doc in load_documents(path):
        yield doc


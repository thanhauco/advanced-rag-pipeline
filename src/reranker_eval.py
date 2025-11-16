from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.table import Table

from .pipeline import build_pipeline
from .reranker import CrossEncoderReranker

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class RerankerSetting:
    name: str
    retrieval_weight: float
    rerank_weight: float
    top_k: int


def load_config(path: str | Path) -> tuple[List[str], List[RerankerSetting]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    queries = raw.get("queries", [])
    settings = [
        RerankerSetting(
            name=item["name"],
            retrieval_weight=float(item["retrieval_weight"]),
            rerank_weight=float(item["rerank_weight"]),
            top_k=int(item.get("top_k", 4)),
        )
        for item in raw.get("reranker_settings", [])
    ]
    return queries, settings


@app.command()
def evaluate(
    config_path: str = typer.Option(
        "configs/reranker_eval.yaml", help="Path to reranker evaluation config."
    ),
    data_path: str = typer.Option(
        "data/knowledge_base.json", help="Path to the knowledge base."
    ),
) -> None:
    queries, settings = load_config(config_path)
    retriever, processor = build_pipeline(data_path)
    reranker = CrossEncoderReranker()
    table = Table(title="Reranker Sensitivity Sweep", show_lines=True)
    table.add_column("Query", style="magenta", overflow="fold")
    table.add_column("Setting", style="cyan")
    table.add_column("Retrieval Weight", justify="right")
    table.add_column("Rerank Weight", justify="right")
    table.add_column("Top-k", justify="right")
    table.add_column("Avg Score", justify="right")

    for query in queries:
        bundle = processor.process(query)
        retrieval_results = retriever.retrieve(bundle)
        for setting in settings:
            reranked = reranker.rerank(
                query=query,
                candidates=retrieval_results,
                top_k=setting.top_k,
                retrieval_weight=setting.retrieval_weight,
                rerank_weight=setting.rerank_weight,
            )
            avg_score = (
                sum(item.score for item in reranked) / len(reranked)
                if reranked
                else 0.0
            )
            table.add_row(
                query,
                setting.name,
                f"{setting.retrieval_weight:.2f}",
                f"{setting.rerank_weight:.2f}",
                str(setting.top_k),
                f"{avg_score:.3f}",
            )
    console.print(table)


if __name__ == "__main__":
    app()

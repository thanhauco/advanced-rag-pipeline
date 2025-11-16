from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import typer
from rich.console import Console
from rich.table import Table

from .pipeline import PipelineArtifacts, build_pipeline, run_pipeline
from .refrag import RefragCompressor, RefragDecoder, RefragSelector

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class SelectorConfig:
    name: str
    micro_chunk_size: int
    retain_ratio: float


def load_config(path: str | Path) -> tuple[List[SelectorConfig], List[str]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    selectors = [
        SelectorConfig(
            name=item["name"],
            micro_chunk_size=int(item["micro_chunk_size"]),
            retain_ratio=float(item["retain_ratio"]),
        )
        for item in raw.get("selectors", [])
    ]
    return selectors, raw.get("queries", [])


def measure_compression(artifacts: PipelineArtifacts) -> int:
    return sum(len(chunk.text.split()) for chunk in artifacts.chunks)


def run_selector(
    selector_cfg: SelectorConfig,
    query: str,
    artifacts: PipelineArtifacts,
) -> Dict[str, float]:
    compressor = RefragCompressor(micro_size=selector_cfg.micro_chunk_size)
    selector = RefragSelector(retain_ratio=selector_cfg.retain_ratio)
    decoder = RefragDecoder()
    micros = compressor.compress_documents(artifacts.chunks)
    selected = selector.select(query, micros)
    summary = decoder.decode(selected)
    total_tokens = sum(len(micro.text.split()) for micro in micros)
    selected_tokens = sum(len(micro.text.split()) for micro in selected)
    compression_ratio = selected_tokens / total_tokens if total_tokens else 0.0

    return {
        "selector": selector_cfg.name,
        "micro_size": selector_cfg.micro_chunk_size,
        "retain_ratio": selector_cfg.retain_ratio,
        "summary_tokens": len(summary.split()),
        "compression_ratio": compression_ratio,
    }


@app.command()
def tune(
    config_path: str = typer.Option(
        "configs/refrag_config.yaml", help="Path to selector config YAML."
    ),
    data_path: str = typer.Option(
        "data/knowledge_base.json", help="Path to knowledge base."
    ),
) -> None:
    selectors, queries = load_config(config_path)
    retriever, processor = build_pipeline(data_path)
    table = Table(title="REFRAG Selector Sweep", show_lines=True)
    table.add_column("Selector", style="cyan")
    table.add_column("Query", style="magenta", overflow="fold")
    table.add_column("Micro Size", justify="right")
    table.add_column("Retain Ratio", justify="right")
    table.add_column("Summary Tokens", justify="right")
    table.add_column("Compression Ratio", justify="right")

    for query in queries:
        artifacts = run_pipeline(
            query=query,
            data_path=data_path,
            retriever=retriever,
            processor=processor,
        )
        for selector_cfg in selectors:
            metrics = run_selector(selector_cfg, query, artifacts)
            table.add_row(
                metrics["selector"],
                query,
                str(metrics["micro_size"]),
                f"{metrics['retain_ratio']:.2f}",
                str(int(metrics["summary_tokens"])),
                f"{metrics['compression_ratio']:.2f}",
            )
    console.print(table)


if __name__ == "__main__":
    app()

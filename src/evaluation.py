from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import typer
from rich.console import Console
from rich.table import Table

from .pipeline import PipelineArtifacts, build_pipeline, run_pipeline

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class EvalSample:
    question: str
    expected_keywords: List[str]
    notes: str = ""


@dataclass
class EvalResult:
    question: str
    coverage: float
    matched_keywords: List[str]
    notes: str


def load_samples(path: str | Path) -> List[EvalSample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalSample(
            question=item["question"],
            expected_keywords=item.get("expected_keywords", []),
            notes=item.get("notes", ""),
        )
        for item in data
    ]


def score_keywords(text: str, keywords: Sequence[str]) -> tuple[float, List[str]]:
    lowered = text.lower()
    matches = [kw for kw in keywords if kw.lower() in lowered]
    coverage = len(matches) / len(keywords) if keywords else 1.0
    return coverage, matches


def evaluate_sample(
    sample: EvalSample,
    data_path: str,
    retriever=None,
    processor=None,
) -> EvalResult:
    artifacts: PipelineArtifacts = run_pipeline(
        query=sample.question,
        data_path=data_path,
        retriever=retriever,
        processor=processor,
    )
    context_text = "\n".join(chunk.text for chunk in artifacts.chunks)
    combined_text = (
        f"{context_text}\n{artifacts.answer_outline}\n{artifacts.refrag_summary}"
    )
    coverage, matches = score_keywords(combined_text, sample.expected_keywords)
    return EvalResult(
        question=sample.question,
        coverage=coverage,
        matched_keywords=matches,
        notes=sample.notes,
    )


@app.command()
def run(
    questions_path: str = typer.Option(
        "data/eval_questions.json", help="Path to evaluation questions JSON."
    ),
    data_path: str = typer.Option(
        "data/knowledge_base.json", help="Path to the knowledge base JSON."
    ),
) -> None:
    samples = load_samples(questions_path)
    retriever, processor = build_pipeline(data_path)
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Question", style="cyan", overflow="fold", justify="left")
    table.add_column("Coverage", style="green", justify="center")
    table.add_column("Matched Keywords", style="magenta", overflow="fold")
    table.add_column("Notes", style="yellow", overflow="fold")

    for sample in samples:
        result = evaluate_sample(
            sample, data_path=data_path, retriever=retriever, processor=processor
        )
        coverage_pct = f"{result.coverage * 100:.0f}%"
        table.add_row(
            sample.question,
            coverage_pct,
            ", ".join(result.matched_keywords) or "-",
            result.notes or "-",
        )
    console.print(table)


if __name__ == "__main__":
    app()

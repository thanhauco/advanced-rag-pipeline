from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from .data_loader import load_documents
from .generation import TemplateGenerator
from .indexing import HybridIndexer, SemanticChunker
from .models import DocumentChunk
from .query_processor import QueryProcessor
from .refrag import RefragCompressor, RefragDecoder, RefragSelector
from .retrieval import HybridRetriever
from .reranker import CrossEncoderReranker

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class PipelineArtifacts:
    chunks: List[DocumentChunk]
    refrag_summary: str
    answer_outline: str


def build_pipeline(data_path: str) -> Tuple[HybridRetriever, QueryProcessor]:
    documents = load_documents(data_path)
    chunker = SemanticChunker()
    chunks = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    indexer = HybridIndexer()
    indexer.build(chunks)
    retriever = HybridRetriever(indexer=indexer)
    return retriever, QueryProcessor()


def run_pipeline(
    query: str,
    data_path: str = "data/knowledge_base.json",
    retriever: Optional[HybridRetriever] = None,
    processor: Optional[QueryProcessor] = None,
) -> PipelineArtifacts:
    if retriever is None or processor is None:
        retriever, processor = build_pipeline(data_path)
    bundle = processor.process(query)
    retrieval_results = retriever.retrieve(bundle, top_k=6)
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, retrieval_results, top_k=4)
    top_chunks = [result.chunk for result in reranked]

    # REFRAG-inspired compression
    compressor = RefragCompressor()
    selector = RefragSelector()
    decoder = RefragDecoder()
    micros = compressor.compress_documents(top_chunks)
    selected = selector.select(query, micros)
    refrag_summary = decoder.decode(selected)

    generator = TemplateGenerator()
    outline = generator.generate(
        query=query,
        chunks=top_chunks,
        refrag_summary=refrag_summary,
    )
    return PipelineArtifacts(
        chunks=top_chunks,
        refrag_summary=refrag_summary,
        answer_outline=outline,
    )


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to run through the RAG pipeline."),
    data_path: str = typer.Option(
        "data/knowledge_base.json", help="Path to the knowledge base JSON."
    ),
) -> None:
    retriever, processor = build_pipeline(data_path)
    artifacts = run_pipeline(
        query=query,
        data_path=data_path,
        retriever=retriever,
        processor=processor,
    )
    table = Table(title="Advanced RAG Pipeline Output")
    table.add_column("REFRAG Summary", style="cyan", overflow="fold")
    table.add_column("Answer Outline", style="green", overflow="fold")
    table.add_row(artifacts.refrag_summary, artifacts.answer_outline)
    console.print(table)


if __name__ == "__main__":
    app()

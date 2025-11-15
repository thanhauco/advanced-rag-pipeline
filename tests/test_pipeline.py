from pathlib import Path

from src.data_loader import load_documents
from src.indexing import SemanticChunker
from src.pipeline import build_pipeline, run_pipeline

DATA_PATH = Path("data/knowledge_base.json")


def test_load_documents_has_entries():
    documents = load_documents(DATA_PATH)
    assert documents, "Expected at least one document in the knowledge base."
    assert documents[0].content


def test_semantic_chunker_respects_window():
    documents = load_documents(DATA_PATH)
    doc = documents[0]
    chunker = SemanticChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk(doc)
    assert chunks, "Chunker should produce at least one chunk."
    assert all(len(chunk.text.split()) <= 20 for chunk in chunks)


def test_run_pipeline_produces_refrag_summary():
    retriever, processor = build_pipeline(str(DATA_PATH))
    artifacts = run_pipeline(
        query="How does reranking improve the RAG pipeline?",
        data_path=str(DATA_PATH),
        retriever=retriever,
        processor=processor,
    )
    assert artifacts.chunks, "Pipeline should retrieve chunks."
    assert len(artifacts.refrag_summary.split()) > 0
    assert "rerank" in artifacts.answer_outline.lower()

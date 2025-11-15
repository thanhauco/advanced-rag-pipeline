# Advanced RAG Pipeline Playground

Learn Retrieval-Augmented Generation (RAG) concepts with runnable code that mirrors the advanced pipeline described in the provided Grok article. The repository contains structured notes, a small knowledge base, and Python modules that implement core stages such as indexing, query processing, retrieval, reranking, and REFRAG-inspired compression ideas.

## What's Inside

- `data/knowledge_base.json` – curated facts from the article, ready for indexing.
- `data/eval_questions.json` – sample queries + keywords for pipeline evaluation.
- `src/` – Python package with the following modules:
  - `models.py` – data structures (documents + chunks).
  - `data_loader.py` – utilities to load the JSON knowledge base.
  - `query_processor.py` – demonstrates synonym expansion, Hypothetical Document Embeddings (HyDE), and multi-query decomposition.
  - `indexing.py` – semantic chunker and TF-IDF hybrid indexer.
  - `retrieval.py` – hybrid retriever (dense-like + lexical) plus context aggregation.
  - `reranker.py` – lightweight cross-encoder–style reranker.
  - `refrag.py` – REFRAG-inspired compress/sense/expand components.
  - `generation.py` – simple template generator to inspect retrieved context.
  - `pipeline.py` – Typer CLI that wires the stages together (`python -m src.pipeline ask "question"`).
  - `evaluation.py` – CLI to score keyword coverage over sample questions.
- `docs/master_plan.md` – step-by-step learning roadmap covering indexing through REFRAG enhancements.
- `docs/diagrams.md` – ASCII diagrams for the full pipeline, reranking, and REFRAG.
- `docs/tutorial.md` – hands-on walkthrough for running the CLI + evaluation.
- `tests/` – pytest suite covering data loading, chunking, and pipeline execution.
- `requirements.txt` – Python dependencies (FAISS, sentence-transformers, etc.) for experimentation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline ask "How does reranking improve the RAG pipeline?"
python -m src.evaluation run  # optional keyword-coverage eval
pytest  # run unit tests
```

After installing dependencies you can run the CLI above, open the `src/` modules in a notebook or REPL, extend them into a complete RAG service, or integrate with frameworks such as LangChain, LlamaIndex, or Haystack.

## Learning Objectives

1. **Indexing & Preparation** – explore semantic chunking, hybrid dense+lexical indexing, metadata enrichment, and knowledge-graph hooks. Expect ~20-50% precision gains.
2. **Query Processing** – apply query rewriting, HyDE, and multi-query decomposition to boost recall for ambiguous or long-tail questions.
3. **Retrieval** – implement hybrid search, multi-stage (coarse-to-fine) retrieval, or graph traversal for large knowledge bases.
4. **Post-Retrieval Processing** – rerank, compress, or fuse results to save 30-40% context tokens.
5. **Generation** – prompt adaptively, run self-consistency, or use agentic loops to critique and iterate.
6. **Evaluation & Iteration** – measure faithfulness, relevance, and precision/recall using frameworks like RAGAS; add A/B tests with human or LLM judges.
7. **REFRAG Enhancements** – experiment with compress-sense-expand decoding to achieve 2-4× fewer tokens and 30× faster inference on long contexts.
8. **Reranking Practices** – integrate cross-encoders (e.g., Cohere Rerank, BERT) to reorder candidates before generation or REFRAG selection.

## Next Steps

See `docs/master_plan.md` for a detailed build roadmap and suggested experiments. Extend `src/` with retrieval pipelines (FAISS, BM25), rerankers, REFRAG selectors, and evaluation scripts tied to your projects.

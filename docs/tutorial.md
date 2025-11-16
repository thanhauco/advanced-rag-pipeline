# Advanced RAG Tutorial

Use this walkthrough to explore each component introduced in the project. Every step maps to techniques from the Grok article.

## 1. Load and Inspect the Knowledge Base

```bash
cat data/knowledge_base.json | jq '.[0]'
```

Observe how metadata encodes stages (indexing, retrieval, etc.)—extend it with your own documents as you learn.

## 2. Run the Pipeline CLI

```bash
python -m src.pipeline ask "What makes REFRAG efficient?"
```

Output highlights:

- **REFRAG Summary** – compressed micro-chunks selected by the heuristic selector (stand-in for RL policy).
- **Answer Outline** – template showing how to frame an LLM prompt using retrieved context.

Experiment by editing `src/query_processor.py` (e.g., add synonyms) and rerun the CLI to see retrieval changes.

## 3. Examine Components

Open these files to understand each stage:

- `src/indexing.py` – semantic chunker + TF-IDF hybrid index.
- `src/retrieval.py` – multi-query hybrid retrieval with lexical blending.
- `src/reranker.py` – cross-encoder–style reranker.
- `src/refrag.py` – compress → sense → expand prototype.

Add comments or counters to observe scoring behaviour.

## 4. Run Evaluation

```bash
python -m src.evaluation run --questions data/eval_questions.json
```

This computes keyword coverage for each sample question, acting as a lightweight proxy for context precision. Modify `data/eval_questions.json` with your own keywords to track progress.

## 5. Extend Further

- Swap the TF-IDF indexer for FAISS + BM25 to align with the hybrid search recommendations.
- Replace the heuristic reranker with Cohere Rerank or a BERT cross-encoder.
- Plug REFRAG modules into an LLM (e.g., Llama-3) to test token savings.
- Add evaluation metrics from RAGAS or LLM-as-a-judge to replace keyword coverage.
- Open `notebooks/rag_playground.ipynb` to script multi-step experiments (build pipeline, run queries, evaluate keyword coverage) without touching the CLI.
- Run `python -m src.refrag_tuning tune --config configs/refrag_config.yaml` to compare micro-chunk sizes and retain ratios; update the YAML with your own sweeps.
- Run `python -m src.reranker_eval evaluate --config configs/reranker_eval.yaml --output reports/reranker_eval.json` to study how different retrieval/rerank weightings influence average relevance scores and log the results.

Iteratively evolve the pipeline following `docs/master_plan.md`, logging each experiment to build intuition about advanced RAG behaviour.

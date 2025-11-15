# ASCII Diagrams – Advanced RAG, REFRAG, and Reranking

## 1. Advanced RAG Pipeline

```
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────┐    ┌─────────────────────┐    ┌───────────────┐    ┌─────────────────────┐
│Indexing & Prep   │    │Query Processing    │    │Retrieval         │    │Post-Retrieval       │    │Generation      │    │Evaluation & Iteration│
│- chunk + metadata│    │- rewrites + HyDE   │    │- hybrid search   │    │- rerank/compress    │    │- adaptive prompts│  │- metrics + A/B tests │
└─────────┬────────┘    └──────────┬─────────┘    └────────┬────────┘    └──────────┬──────────┘    └──────┬─────────┘    └──────────┬───────────┘
          │                         │                      │                        │                    │                       │
          └─────────────┬───────────┴───────────┬──────────┴────────────┬───────────┴────────────┬───────┴────────────┬──────────────┘
                        │                       │                      │                        │                     │
                        ▼                       ▼                      ▼                        ▼                     ▼
                   Knowledge Base        Processed Query          Candidate Docs         Refined Context        Deploy + Monitor
```

## 2. Query Processing Internals

```
┌─────────┐   rewrites   ┌──────────────┐
│ Original├─────────────►│Synonym Expander│
│ Query   │              └──────┬───────┘
└────┬────┘   hypothetical            │
     │          answer                ▼
     │       ┌────────────┐      ┌─────────────┐
     └──────►│HyDE Builder│      │Decomposer   │
             └─────┬──────┘      └────┬────────┘
                   │                 sub-queries
                   ▼                      ▼
            HyDE Embedding        Multi-query Bundle
```

## 3. Retrieval + Reranking Flow

```
┌────────────┐           ┌───────────────┐           ┌──────────────┐
│Embedding DB│◄──────────┤Hybrid Retriever├──────────►│Top-N Candidates│
└──────┬─────┘           └───────┬───────┘           └─────┬─────────┘
       │                         │                           │
       │                         ▼                           ▼
       │                 Lexical Scores            Cross-Encoder Reranker
       │                         │                           │
       ▼                         └──────────────┬────────────┘
Dense Embeds                                 Final Top-k
```

## 4. REFRAG Compress–Sense–Expand

```
┌────────────┐      ┌────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│Retrieval   │────►│Micro-Chunk  │────►│Compression   │────►│Sense/Select  │────►│Expand+Decode │
│(vector DB) │      │(16 tokens) │      │(tiny encoder)│      │(RL policy)  │      │(LLM decoder) │
└────────────┘      └─────┬──────┘      └─────┬───────┘      └─────┬───────┘      └─────┬───────┘
                           │                   │                   │                   │
                           ▼                   ▼                   ▼                   ▼
                     Micro-chunks       Embedding seq.       Sparse mask        Selected context
                                                                              (few expanded chunks)
```

## 5. Evaluation & Feedback Loop

```
┌──────────────┐    outputs    ┌─────────────┐    metrics    ┌─────────────┐
│LLM Generation├──────────────►│Evaluators   ├──────────────►│Dashboard    │
└──────┬───────┘               │(RAGAS, judges)              │(W&B, custom)│
       │                       └──────┬──────┘               └──────┬──────┘
       │                              │                           │
       └────────────feedback──────────┴───────────────optimize─────┘
```

Each diagram maps directly to the Grok article’s processes, giving you a visual reference while building the code modules.

## 6. End-to-End Advanced RAG Flow

```
+-------------------+       +-------------------+       +-------------------+
|  User Query       |       | Query Processing  |       |   Retrieval       |
|  (Input)          | ----> | - Rewrite/Expand  | ----> | - Hybrid Search   |
|                   |       | - HyDE Embed      |       | - Top-k Fetch     |
+-------------------+       +-------------------+       +-------------------+
          |                           |                           |
          v                           v                           v
+-------------------+       +-------------------+       +-------------------+
| Indexing/Prep     | <---- | (Parallel: Index  | <---- | Knowledge Base    |
| (Offline)         |       |  Maintenance)     |       | (Docs/Chunks)     |
| - Chunking        |       |                   |       |                   |
| - Embeddings      |       +-------------------+       +-------------------+
+-------------------+                 ^                           ^
          |                           |                           |
          v                           |                           |
+-------------------+       +-------------------+       +-------------------+
| Post-Retrieval    | <---- |   Reranking       | <---- | Candidate Chunks  |
| - Compress/Fuse   |       | - Cross-Encoder   |       | (Top-100)         |
| - Filter          |       | - Reorder Top-k   |       +-------------------+
+-------------------+       +-------------------+                 |
          |                           ^                           |
          v                           |                           |
+-------------------+                 |                           |
|   Generation      | <----------------+                           |
| - Prompt LLM      |                                           |
| - Adaptive Prompt |                                           |
+-------------------+                                           |
          |                                                     |
          v                                                     |
+-------------------+                                         |
|  Response Output  | <---------------------------------------+
|  (Grounded Answer)|
+-------------------+
          |
          v
+-------------------+
| Evaluation/Iter   |
| - Metrics (RAGAS) |
| - Tune Loop       |
+-------------------+
```

## 7. REFRAG Compress–Sense–Expand Loop

```
+-------------------+       +-------------------+       +-------------------+
|  User Query       | ----> |   Retrieval       | ----> | Retrieved         |
|  (Input)          |       | (Standard RAG)    |       | Passages          |
|                   |       | - Vector Fetch    |       | (Top-k Docs)      |
+-------------------+       +-------------------+       +-------------------+
          |                           |                           |
          v                           v                           v
+-------------------+       +-------------------+       +-------------------+
| (Offline: Index)  | <---- |   Compress        | ----> | Micro-Chunks      |
| - Embed Store     |       | - Split (16-tok)  |       | Embeddings        |
+-------------------+       | - Lightweight Enc |       | (Seq per Passage) |
                            +-------------------+       +-------------------+
                                      |                           ^
                                      v                           |
                            +-------------------+                 |
                            |     Sense/Select  | <----------------+
                            | - RL Policy Score |
                            | - Sparse Mask     |  (Top 20-30%)
                            | (Query + Embs)    |
                            +-------------------+
                                      |
                                      v
                            +-------------------+       +-------------------+
                            |   Expand & Decode | ----> |   LLM Generation  |
                            | - Re-expand Tokens|       | - Inject Context  |
                            | - Attention on    |       | - Output Response |
                            |   Selected Chunks |       +-------------------+
                            +-------------------+                 |
                                      |                           |
                                      v                           v
                            +-------------------+       +-------------------+
                            | Feedback Loop     | <---- | Eval Metrics      |
                            | - RLHF Fine-tune  |       | (Perplexity/ROUGE)|
                            +-------------------+       +-------------------+
```

## 8. Reranking Focus View

```
[Initial Retrieval Output]                  [Reranking Step]                  [Final Top-k for Generation]

+---------------------------+               +---------------------------+     +---------------------------+
| Candidate Chunks (Top-100)|               | Cross-Encoder Model       |     | Refined Top-10 Chunks     |
|                           |               | - Query-Passage Pairs     |     | (High-Relevance Only)     |
|  Chunk 1: Score 0.85      | ---->         | - Semantic + Lexical      | --> |  Chunk 5: Score 0.98      |
|  Chunk 47: Score 0.72     |               |   Scoring                 |     |  Chunk 1: Score 0.95      |
|  Chunk 23: Score 0.91     |               | - Reorder by Relevance    |     |  Chunk 23: Score 0.92     |
|  ... (Noisy/Irrelevant)   |               +---------------------------+     |  ... (Filtered Noise)     |
|  Chunk 99: Score 0.45     |                         |                     |  Chunk 72: Score 0.88     |
+---------------------------+                         v                     +---------------------------+
                                   (Filters ~80% Noise)                           |
                                                                                  v
                                                                               [Prompt to LLM]
```

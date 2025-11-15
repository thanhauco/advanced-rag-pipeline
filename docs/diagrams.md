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


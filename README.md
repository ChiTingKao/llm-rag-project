## System Architecture

This project implements a full RAG pipeline:

1. Query Encoding (E5)
2. Vector Retrieval (FAISS)
3. Cross-Encoder Reranking
4. LLM Generation (Ollama)

## Features

- Sentence-aware chunking
- Hybrid retrieval-ready design
- Reranking (bge-reranker)
- Evaluation pipeline

## Future Work

- Hybrid Search (BM25 + Vector)
- Query Rewriting
- Multi-hop QA

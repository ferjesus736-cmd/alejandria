<div align="center">

# 📚 Alejandria

**Production-grade Distributed RAG Engine · Multi-Tenant · Hybrid Search · Async Ingestion**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-DC143C?style=flat-square)](https://qdrant.tech/)
[![Celery](https://img.shields.io/badge/Celery-Async-37814A?style=flat-square&logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

> Alejandria is a modular, multi-tenant Retrieval-Augmented Generation (RAG) backend built for production workloads.
> It combines hybrid vector search (dense + sparse + RRF fusion) with cross-encoder reranking and provider-agnostic LLM generation — all behind a clean async REST API.

</div>

---

## Table of Contents

- [Why Alejandria](#why-alejandria)
- [Architecture Overview](#architecture-overview)
- [Key Design Decisions](#key-design-decisions)
- [Feature Matrix](#feature-matrix)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [LLM Provider Configuration](#llm-provider-configuration)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Production Deployment](#production-deployment)
- [Roadmap](#roadmap)

---

## Why Alejandria

Most RAG prototypes are single-user toy demos. Alejandria was designed from day one as a **multi-tenant service**: each user gets isolated vector collections in Qdrant, async document ingestion via Celery workers, and a retrieval layer that fuses dense and sparse signals rather than relying on cosine similarity alone.

The result is a backend you can drop in front of any LLM provider (NVIDIA, Ollama, Hugging Face local) and expose to multiple clients without cross-contamination of knowledge bases.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                          FastAPI (app.py)                         │
│   POST /upload        POST /chat        POST /chat/stream         │
└────────────┬─────────────────┬───────────────────┬───────────────┘
             │                 │                   │
     ┌───────▼──────┐  ┌───────▼──────┐   ┌────────▼──────────┐
     │  Celery/Local │  │  Retriever   │   │  StreamingResponse │
     │   Worker      │  │  (Hybrid)    │   │  (token generator) │
     └───────┬───────┘  └───────┬──────┘   └────────────────────┘
             │                  │
    ┌────────▼────────┐  ┌──────▼──────────┐
    │ Ingestion Layer │  │ Qdrant VectorDB  │
    │  reader →        │  │ (per-user col.) │
    │  chunker →       │  └──────┬──────────┘
    │  embedder →      │         │
    │  vectorstore     │  ┌──────▼──────────┐
    └─────────────────┘  │  CrossEncoder    │
                          │  Reranker        │
                          └──────┬──────────┘
                                 │
                          ┌──────▼──────────┐
                          │  LLM Provider   │
                          │  (Ollama /       │
                          │   NVIDIA /       │
                          │   HuggingFace)   │
                          └─────────────────┘
```

---

## Key Design Decisions

**Multi-tenancy via header isolation** — Every request carries an `x-user-id` header. Qdrant collections are namespaced per user; no shared state ever leaks between tenants.

**Hybrid search with RRF fusion** — Dense embeddings (semantic) are combined with sparse BM25-style vectors. Reciprocal Rank Fusion (RRF) merges both ranked lists before reranking, capturing both semantic similarity and lexical overlap.

**Pluggable worker execution** — The ingestion worker auto-detects whether Celery is available. In development it falls back to a daemon thread (`_LocalTask`), making the service dependency-free for quick local iteration. In production it binds to Redis transparently — zero code changes required.

**Provider-agnostic generation** — `_build_provider()` resolves the LLM at startup via env vars. Swapping from a local SmolLM2 to NVIDIA `meta/llama-3.1-8b-instruct` is a single env var change, with no code touched.

**MD5-keyed retrieval cache** — Results are cached in memory keyed by `(user_id, question, top_k)`. This eliminates redundant Qdrant round-trips on repeated questions within the same process lifetime.

**Hard chunk safety gate** — Documents exceeding 5,000 chunks are rejected at the worker level before any vector is written, preventing runaway storage costs from malicious or accidental uploads.

---

## Feature Matrix

| Capability | Status |
|---|---|
| Multi-tenant isolation (per-user Qdrant collections) | ✅ |
| Async document ingestion (Celery + Redis) | ✅ |
| Local fallback worker (no Redis required) | ✅ |
| Hybrid dense + sparse vector search | ✅ |
| Reciprocal Rank Fusion (RRF) | ✅ |
| Cross-encoder reranking | ✅ |
| Streaming chat (`text/plain` SSE-compatible) | ✅ |
| Ollama provider | ✅ |
| NVIDIA NIM provider | ✅ |
| Hugging Face local model provider | ✅ |
| Dockerized deployment (API + Worker + Redis) | ✅ |
| In-memory retrieval cache | ✅ |
| Chunk safety limit (max 5000 chunks/doc) | ✅ |
| Structured source attribution in `/chat` response | ✅ |

---

## Project Structure

```
alejandria/
├── app.py                  # FastAPI entrypoint — routes, provider init, cache
├── worker.py               # Celery task + local thread fallback
│
├── ingestion/
│   └── reader.py           # Document reading (txt, pdf, etc.)
│
├── chunking/
│   └── chunker.py          # Text chunking strategy
│
├── embeddings/
│   └── embedder.py         # Dense + sparse embedding generation
│
├── vectorstore/
│   └── qdrant_store.py     # Qdrant collection init, vector insertion
│
├── retrieval/
│   └── retriever.py        # Hybrid search + RRF + CrossEncoder reranking
│
├── generation/
│   └── generator.py        # generate_answer / stream_answer
│
├── providers/
│   ├── huggingface_provider.py
│   ├── nvidia_provider.py
│   └── ollama_provider.py
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/ferjesus736-cmd/alejandria.git
cd alejandria

# Configure your provider (see Environment Variables below)
cp .env.example .env   # edit as needed

docker compose up --build
```

Services started:

| Service | URL |
|---|---|
| API | `http://localhost:8000` |
| Qdrant | `http://localhost:6333` |
| Redis | `redis://localhost:6379` |

### Option B — Local development (no Redis required)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export CELERY_EXECUTION_MODE=local
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3

uvicorn app:app --reload
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `nvidia`, `huggingface` |
| `OLLAMA_MODEL` | `llama3` | Model name served by local Ollama |
| `NVIDIA_API_KEY` | — | NVIDIA NIM API key |
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | NVIDIA model identifier |
| `HF_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` | Hugging Face model ID or path |
| `CELERY_EXECUTION_MODE` | `celery` | `celery` (production) or `local` (dev) |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery message broker |
| `CELERY_RESULT_BACKEND` | same as broker | Celery result store |

---

## API Reference

### `POST /upload`

Upload a document for async ingestion. The file is chunked, embedded, and stored in the user's isolated Qdrant collection.

**Headers**

| Header | Required | Description |
|---|---|---|
| `x-user-id` | ✅ | Tenant identifier |

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | Document to ingest (txt, pdf, etc.) |

**Response**

```json
{
  "status": "queued",
  "task_id": "a3f7c...",
  "doc": "report.pdf",
  "user_id": "acme-corp"
}
```

**Example**

```bash
curl -X POST http://localhost:8000/upload \
  -H "x-user-id: acme-corp" \
  -F "file=@report.pdf"
```

---

### `POST /chat`

Ask a question against the user's knowledge base. Returns the answer and ranked source attribution.

**Headers** — `Content-Type: application/json`, `x-user-id: <tenant>`

**Body**

```json
{
  "question": "What were the main findings of the Q3 audit?",
  "top_k": 5
}
```

**Response**

```json
{
  "answer": "The Q3 audit identified three critical gaps in ...",
  "sources": [
    { "doc_id": "audit_q3.pdf", "chunk": 14, "score": 0.921, "rrf_score": 0.0312 },
    { "doc_id": "audit_q3.pdf", "chunk": 17, "score": 0.884, "rrf_score": 0.0289 }
  ]
}
```

---

### `POST /chat/stream`

Same as `/chat` but streams the LLM response token by token (`text/plain`).

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "x-user-id: acme-corp" \
  -d '{"question": "Summarize the risk section", "top_k": 5}'
```

---

## LLM Provider Configuration

### Ollama (local inference)

```bash
# Install Ollama and pull a model
ollama pull llama3

export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3
```

### NVIDIA NIM

```bash
export LLM_PROVIDER=nvidia
export NVIDIA_API_KEY=nvapi-...
export NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

### Hugging Face (local)

```bash
export LLM_PROVIDER=huggingface
export HF_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
```

> Any model available on the Hugging Face Hub can be used; larger models require appropriate GPU resources.

---

## Ingestion Pipeline

When a file is uploaded, the worker executes this pipeline:

```
read_document(tmp_path)
    ↓
chunk_text(text)               # Splits into semantic chunks
    ↓
create_embeddings(chunks)      # Dense + sparse vectors
    ↓
init_collection(user_id)       # Creates Qdrant collection if absent
    ↓
insert_vectors(chunks, vectors, doc_id, user_id)
    ↓
Cleanup temp file
```

Each stage logs timing metrics (`embed_seconds`, `store_seconds`, `total_seconds`) for operational visibility. Documents exceeding **5,000 chunks** are rejected before any write occurs.

---

## Retrieval Pipeline

```
retrieve(question, top_k, user_id)
    ↓
Dense query vector + Sparse query vector
    ↓
Qdrant hybrid search (per-user collection)
    ↓
Reciprocal Rank Fusion (RRF) of both ranked lists
    ↓
CrossEncoder reranking of top candidates
    ↓
build_context(results)         # Assembles prompt context string
    ↓
generate_answer / stream_answer
```

`top_k` is capped at **8** server-side to prevent context window overflow regardless of client-supplied values.

---

## Production Deployment

The included `docker-compose.yml` provisions three services:

```yaml
services:
  api:     # FastAPI + Uvicorn
  worker:  # Celery worker consuming the ingestion queue
  redis:   # Message broker + result backend
```

A production checklist before going live:

- [ ] Mount a persistent volume for Qdrant data (`/qdrant/storage`)
- [ ] Set `NVIDIA_API_KEY` or point `OLLAMA_MODEL` to a production-grade model
- [ ] Place an Nginx reverse proxy in front with TLS termination
- [ ] Add a `SECRET_HEADER` middleware to validate `x-user-id` against an auth layer
- [ ] Configure Celery concurrency (`--concurrency`) based on available CPU/GPU
- [ ] Monitor task queue depth via Flower (`celery flower`)

---

## Roadmap

- [ ] `/task/{id}` status endpoint for ingestion progress polling
- [ ] JWT-based tenant authentication middleware
- [ ] Persistent retrieval cache (Redis-backed) for horizontal scaling
- [ ] Metadata filtering on retrieval (date ranges, document tags)
- [ ] OpenTelemetry tracing integration
- [ ] Evaluation harness (RAGAS metrics) for retrieval quality benchmarking
- [ ] Kubernetes Helm chart

---

<div align="center">

Built with precision by **[ferjesus736-cmd](https://github.com/ferjesus736-cmd)** · Santiago, Chile

*"Not just a RAG demo — a service that scales."*

</div>

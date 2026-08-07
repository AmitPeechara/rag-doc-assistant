# RAG Documentation Assistant

A retrieval-augmented generation (RAG) system that ingests API/technical documentation
(`.md`, `.txt`, `.pdf`) and answers natural-language questions grounded in that content,
with streaming responses and source attribution.

Built as Project 3 in a self-directed AI engineering learning path, following Project 1
(structured extraction with Gemini's `response_schema`). This project focuses on the
retrieval half of LLM applications: chunking, embeddings, vector search, and multi-document
isolation.

## Why this exists

Most RAG tutorials stop at "chunk it, embed it, retrieve it, done." This project treats
that as the starting point, not the finish line - the interesting engineering problems
showed up once multiple documents entered the picture: **how do you stop a system from
silently blending answers across unrelated documents when the user doesn't specify which
one they mean?**

## Architecture

```
File upload (.md/.txt/.pdf)
        |
        v
  extractors.py  --  text extraction, page-aware for PDFs
        |
        v
  chunker.py     --  token-based sliding window (tiktoken, 500 tokens / 100 overlap)
        |
        v
  chroma_client.py -- Gemini embeddings (RETRIEVAL_DOCUMENT) -> Chroma (persisted, metadata-tagged)


User question
        |
        v
  chroma_client.py -- Gemini embeddings (RETRIEVAL_QUERY) -> Chroma similarity search
        |             (optionally filtered by doc_id for single-document scope)
        v
  generator.py    -- retrieved chunks + question -> prompt -> Gemini streaming generation
        |
        v
  Streamlit UI    -- streamed answer + expandable source citations
```

## Key design decisions

**Multi-document isolation via metadata filtering, not separate collections.**
All chunks live in a single Chroma collection, tagged with `doc_id`/`doc_name` metadata.
Queries scoped to one document pass `where={"doc_id": ...}` as a hard filter at the
vector-search level - not a post-hoc re-ranking step. When no document is specified and
retrieved chunks span multiple documents, the system flags this explicitly rather than
silently merging answers across unrelated sources (enforced two ways: a system-prompt
instruction to the LLM, and a structural check on `docs_referenced` in the API response
that the UI surfaces independently of whether the LLM remembers to mention it).

**Asymmetric embeddings for documents vs. queries.**
Gemini's embedding model distinguishes `RETRIEVAL_DOCUMENT` (for stored chunks) from
`RETRIEVAL_QUERY` (for the user's question) task types. Using the wrong one doesn't error -
it silently degrades retrieval quality. `chroma_client.py` enforces this via two separate
functions (`embed_documents` / `embed_query`) rather than a shared function with a flag,
so the correct task_type can't be mixed up at the call site.

**Token-based chunking (tiktoken), not character or word count.**
Chunk size is measured in tokens via `tiktoken`'s `cl100k_base` encoding - an approximation
of Gemini's actual tokenizer, but consistent sizing matters more here than exact parity.
Chunking operates on a flattened token stream across a whole document (not per-page), so
chunks can span page boundaries without losing continuity; each chunk is tagged with the
page it starts on for citation purposes.

**Filename-based document IDs with explicit re-ingestion handling.**
`doc_id` is a sanitized filename rather than a generated UUID, favoring human-readable
debugging over strict uniqueness guarantees. Re-uploading a file with the same name
deletes and replaces its existing chunks (logged, not silent) rather than creating
duplicates or silently no-op'ing - this was a deliberate fix after testing showed Chroma's
default `add()` behavior silently skips duplicate IDs rather than erroring or updating.

**Streaming with metadata sent ahead of the answer.**
`/query/stream` returns retrieval results (sources, referenced documents) as a JSON line
before the answer begins streaming - the UI can show which sources were used while the
answer is still generating, rather than only after the full response completes.

**Containerized as two independent services, not one image.**
The API and UI ship as separate Docker images (`api/Dockerfile`, `ui/Dockerfile`), each
carrying only the dependencies it needs. They communicate over HTTP, with the UI's target
URL controlled by an `API_BASE_URL` environment variable rather than hardcoded - the same
image works unmodified whether run directly on a laptop, in Docker, or later in Kubernetes,
with only the environment changing.

## Tech stack

- **LLM / embeddings:** Google Gemini (`gemini-3.6-flash` for generation, `gemini-embedding-001`
  for embeddings)
- **Vector store:** ChromaDB (persistent, local)
- **Backend:** FastAPI
- **Tokenization:** tiktoken (`cl100k_base`, used as a chunking-consistency proxy)
- **PDF extraction:** pypdf
- **UI:** Streamlit
- **Containerization:** Docker (separate images for API and UI, connected via a user-defined
  Docker network)

## API

| Endpoint | Method | Description |
|---|---|---|
| `/ingest` | POST | Upload a file, extract, chunk, embed, store |
| `/documents` | GET | List all ingested documents |
| `/documents/{doc_id}` | DELETE | Remove a document's chunks |
| `/query` | POST | Ask a question, get a complete (non-streaming) answer |
| `/query/stream` | POST | Ask a question, get a streamed answer with sources sent first |

## Setup (local, no Docker)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

Run the backend and UI in separate terminals:
```bash
# Terminal 1
cd api && python main.py

# Terminal 2
cd ui && streamlit run app.py
```

## Running with Docker

Each service has its own Dockerfile (`api/Dockerfile`, `ui/Dockerfile`), built from the
repo root as the build context. A `.env` file with `GEMINI_API_KEY=your_key_here` is
required in the repo root (not committed - already covered by `.gitignore`).

**1. Build both images:**
```bash
docker build -t rag-api:v1 -f api/Dockerfile .
docker build -t rag-ui:v1 -f ui/Dockerfile .
```

**2. Create a shared network** (so the containers can resolve each other by name instead
of `localhost`, which only refers to a container itself):
```bash
docker network create rag-net
```

**3. Run the API**, publishing port 8000 to the host and loading the Gemini key from `.env`:
```bash
docker run -d --name rag-api --network rag-net -p 8000:8000 --env-file .env rag-api:v1
```

**4. Run the UI**, publishing port 8501 and pointing it at the API by container name:
```bash
docker run -d --name rag-ui --network rag-net -p 8501:8501 -e API_BASE_URL=http://rag-api:8000 rag-ui:v1
```

**5. Open the app:** `http://localhost:8501`

The API is independently reachable at `http://localhost:8000/docs` (FastAPI's interactive
docs) for standalone testing.

To stop and remove both containers:
```bash
docker stop rag-api rag-ui
docker rm rag-api rag-ui
```

## Known limitations (v1)

- Single-user, local-only - no authentication or multi-tenant isolation
- No OCR - scanned/image-based PDFs are rejected with an explicit error rather than
  silently producing empty results
- Retrieval is single-stage (top-k similarity search) - no re-ranking layer
- Chunking is fixed-size with overlap, not semantic/structure-aware - a reasonable
  default for structured API docs, but not optimized for long-form prose

## What I'd do differently at scale

- Swap Chroma for a service-backed vector store (e.g., pgvector, Qdrant) once
  multi-user isolation and concurrent access matter
- Add a re-ranking stage (cross-encoder) between retrieval and generation
- Move from filename-based `doc_id` to a generated ID with filename as a searchable
  metadata field, once true uniqueness (not just human readability) becomes a requirement
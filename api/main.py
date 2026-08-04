"""
main.py
FastAPI app. Step 3 scope only: /ingest, /documents, /delete.
/query is intentionally NOT here yet - per agreed build order, we verify ingestion
end-to-end first, then build query+retrieval as its own step (step 4), then add
streaming on top of a working non-streaming query (step 5).
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

import sys
sys.path.append(str(Path(__file__).parent.parent))  # allow importing sibling modules

from ingestion.extractors import extract_text, ExtractionError
from ingestion.chunker import chunk_text
from vectorstore.chroma_client import add_document_chunks, list_documents, delete_document, sanitize_doc_id, query_chunks
from rag.generator import generate_answer, generate_answer_stream
from schemas import IngestResponse, DocumentSummary, DeleteResponse, QueryRequest, QueryResponse, SourceChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Doc Assistant - Ingestion API")

ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Accepts a file upload, extracts text, chunks it, embeds + stores in Chroma.
    Re-ingestion of the same filename auto-replaces existing chunks (logged, not silent
    in logs - see chroma_client.add_document_chunks).
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # UploadFile is a stream - write to a temp file so extractors.py (which expects a path) can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        segments = extract_text(tmp_path)
    except ExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # always clean up temp file

    chunks = chunk_text(segments)
    if not chunks:
        raise HTTPException(status_code=422, detail=f"No chunkable content extracted from '{file.filename}'.")

    doc_id = sanitize_doc_id(file.filename)
    chunk_count = add_document_chunks(doc_id=doc_id, doc_name=file.filename, chunks=chunks)

    return IngestResponse(
        doc_id=doc_id,
        doc_name=file.filename,
        chunk_count=chunk_count,
        status="success",
    )


@app.get("/documents", response_model=list[DocumentSummary])
async def get_documents():
    """Lists all ingested documents - powers the UI doc selector (FR3)."""
    docs = list_documents()
    return [DocumentSummary(**d) for d in docs]


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_doc(doc_id: str):
    """Deletes all chunks for a given doc_id (FR5)."""
    deleted_count = delete_document(doc_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for doc_id='{doc_id}'.")
    return DeleteResponse(doc_id=doc_id, chunks_deleted=deleted_count)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Non-streaming query endpoint (step 4). Retrieves relevant chunks, optionally
    scoped to a single doc_id, then generates an answer grounded in those chunks.
    Streaming version comes in step 5, built on top of this once verified correct.
    """
    retrieval = query_chunks(
        question=request.question,
        doc_id=request.doc_id,
        n_results=request.n_results,
    )

    retrieved_chunks = [
        {
            "text": doc,
            "doc_name": meta["doc_name"],
            "page_number": meta["page_number"],
        }
        for doc, meta in zip(retrieval["documents"], retrieval["metadatas"])
    ]

    answer = generate_answer(request.question, retrieved_chunks)

    sources = [
        SourceChunk(
            doc_name=meta["doc_name"],
            chunk_index=meta["chunk_index"],
            page_number=meta["page_number"],
            text_preview=doc[:200],
        )
        for doc, meta in zip(retrieval["documents"], retrieval["metadatas"])
    ]

    docs_referenced = list({s.doc_name for s in sources})

    return QueryResponse(answer=answer, sources=sources, docs_referenced=docs_referenced)


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming query endpoint (step 5). Retrieval happens first (fast, not streamed),
    then sources are sent as a single JSON line, then the answer streams token-by-token.

    Response format (plain chunked text, NOT SSE - simplest for Streamlit's st.write_stream
    to consume without extra parsing):
      Line 1: JSON blob - {"sources": [...], "docs_referenced": [...]}
      Line 2+: raw answer text, streamed in pieces as they arrive from Gemini
    """
    retrieval = query_chunks(
        question=request.question,
        doc_id=request.doc_id,
        n_results=request.n_results,
    )

    retrieved_chunks = [
        {
            "text": doc,
            "doc_name": meta["doc_name"],
            "page_number": meta["page_number"],
        }
        for doc, meta in zip(retrieval["documents"], retrieval["metadatas"])
    ]

    sources = [
        {
            "doc_name": meta["doc_name"],
            "chunk_index": meta["chunk_index"],
            "page_number": meta["page_number"],
            "text_preview": doc[:200],
        }
        for doc, meta in zip(retrieval["documents"], retrieval["metadatas"])
    ]
    docs_referenced = list({s["doc_name"] for s in sources})

    def event_generator():
        # First line: sources metadata as JSON, so the UI has it before the answer starts streaming
        header = {"sources": sources, "docs_referenced": docs_referenced}
        yield f"{json.dumps(header)}\n"

        # Remaining pieces: the actual streamed answer text
        for piece in generate_answer_stream(request.question, retrieved_chunks):
            yield piece

    return StreamingResponse(event_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
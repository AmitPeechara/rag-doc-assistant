"""
chroma_client.py
Responsibility: wraps Chroma (persistence, add, query, delete) and Gemini embedding calls.
This is the ONLY module that should talk to Chroma directly - ingestion API and query API
both go through here, so isolation/filtering logic lives in one place (NFR3).

Key design decisions baked in:
- task_type is asymmetric: RETRIEVAL_DOCUMENT for ingested chunks, RETRIEVAL_QUERY for
  user questions. Mixing these up silently degrades retrieval - no error, just bad results.
- doc_id = sanitized filename (not UUID) - re-ingestion of the same filename triggers
  delete-then-add (silent to user, but logged).
- Chroma persists to disk via PersistentClient - survives app restarts (NFR7).
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from google import genai
from google.genai.types import EmbedContentConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"
CHROMA_PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "doc_chunks"
load_dotenv()
_gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # reads GEMINI_API_KEY from env
_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def sanitize_doc_id(filename: str) -> str:
    """filename -> safe doc_id. Strips extension, replaces spaces/special chars."""
    stem = Path(filename).stem
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in stem)
    return safe.lower()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts for STORAGE. Uses RETRIEVAL_DOCUMENT task_type."""
    response = _gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in response.embeddings]


def embed_query(text: str) -> list[float]:
    """Embed a single user question for SEARCH. Uses RETRIEVAL_QUERY task_type.
    NOTE: intentionally a separate function from embed_documents, not just a flag,
    so it's impossible to accidentally call this with the wrong task_type at a callsite."""
    response = _gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def add_document_chunks(doc_id: str, doc_name: str, chunks: list[dict]) -> int:
    """
    Adds a document's chunks to Chroma. If doc_id already exists, deletes existing
    chunks first (silent auto-replace, but logged - per agreed re-ingestion behavior).

    chunks: list of {"text": ..., "chunk_index": ..., "page_number": ...} from chunker.py
    Returns: number of chunks added.
    """
    existing = _collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        logger.info(f"Re-ingesting doc_id='{doc_id}': deleting {len(existing['ids'])} existing chunks.")
        _collection.delete(ids=existing["ids"])

    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)

    ids = [f"{doc_id}_chunk{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunk_index": c["chunk_index"],
            "page_number": c["page_number"] if c["page_number"] is not None else -1,
            # Chroma metadata doesn't accept None - using -1 as sentinel for "no page concept"
        }
        for c in chunks
    ]

    _collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info(f"Added {len(chunks)} chunks for doc_id='{doc_id}' ({doc_name}).")
    return len(chunks)


def query_chunks(question: str, doc_id: str | None = None, n_results: int = 5) -> dict:
    """
    Retrieves top-n chunks relevant to the question.
    If doc_id is provided, results are filtered to that doc only (isolation - NFR3).
    If doc_id is None, searches across all docs.

    Returns dict with parallel lists: documents, metadatas, distances.
    """
    query_vector = embed_query(question)

    where_filter = {"doc_id": doc_id} if doc_id else None

    results = _collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where=where_filter,
    )

    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def list_documents() -> list[dict]:
    """Returns summary of all ingested docs: doc_id, doc_name, chunk_count. Powers FR3/UI dropdown."""
    all_chunks = _collection.get()
    if not all_chunks["ids"]:
        return []

    doc_summary = {}
    for meta in all_chunks["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in doc_summary:
            doc_summary[doc_id] = {"doc_id": doc_id, "doc_name": meta["doc_name"], "chunk_count": 0}
        doc_summary[doc_id]["chunk_count"] += 1

    return list(doc_summary.values())


def delete_document(doc_id: str) -> int:
    """Deletes all chunks for a given doc_id. Returns number of chunks deleted. Powers FR5."""
    existing = _collection.get(where={"doc_id": doc_id})
    if not existing["ids"]:
        return 0
    _collection.delete(ids=existing["ids"])
    logger.info(f"Deleted {len(existing['ids'])} chunks for doc_id='{doc_id}'.")
    return len(existing["ids"])


if __name__ == "__main__":
    # Standalone smoke test - run: python chroma_client.py
    # Tests the REAL Gemini embedding call + Chroma add/query/delete end-to-end.
    # Requires GEMINI_API_KEY set in environment (.env loaded via python-dotenv, or exported).

    print("=== Testing add_document_chunks (real Gemini embeddings) ===")
    fake_chunks_doc1 = [
        {"text": "Razorpay base URL is https://api.razorpay.com/v1", "chunk_index": 0, "page_number": 1},
        {"text": "Razorpay auth uses Basic Auth with API key and secret", "chunk_index": 1, "page_number": 1},
    ]
    fake_chunks_doc2 = [
        {"text": "Stripe base URL is https://api.stripe.com/v1", "chunk_index": 0, "page_number": 1},
    ]

    added1 = add_document_chunks("razorpay_test", "razorpay_test.md", fake_chunks_doc1)
    added2 = add_document_chunks("stripe_test", "stripe_test.md", fake_chunks_doc2)
    print(f"Added {added1} chunks for razorpay_test, {added2} chunks for stripe_test.")

    print("\n=== Testing list_documents ===")
    docs = list_documents()
    for d in docs:
        print(f"  {d}")

    print("\n=== Testing query_chunks WITHOUT doc_id filter ===")
    results = query_chunks("What is the base URL?", n_results=3)
    for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
        print(f"  [{meta['doc_name']}] dist={dist:.4f} | {doc}")

    print("\n=== Testing query_chunks WITH doc_id filter (razorpay_test only) ===")
    results = query_chunks("What is the base URL?", doc_id="razorpay_test", n_results=3)
    for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
        print(f"  [{meta['doc_name']}] dist={dist:.4f} | {doc}")
    print("If ONLY razorpay_test appears above, isolation (NFR3) is confirmed with REAL embeddings.")

    print("\n=== Testing re-ingestion (should delete + replace, logged) ===")
    added1_again = add_document_chunks("razorpay_test", "razorpay_test.md", fake_chunks_doc1)
    print(f"Re-added {added1_again} chunks (check log line above for delete confirmation).")

    print("\n=== Cleaning up test data ===")
    deleted1 = delete_document("razorpay_test")
    deleted2 = delete_document("stripe_test")
    print(f"Deleted {deleted1} chunks from razorpay_test, {deleted2} from stripe_test.")
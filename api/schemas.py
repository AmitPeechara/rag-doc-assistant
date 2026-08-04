"""
schemas.py
Pydantic models for request/response validation across the API.
"""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    doc_id: str
    doc_name: str
    chunk_count: int
    status: str


class DocumentSummary(BaseModel):
    doc_id: str
    doc_name: str
    chunk_count: int


class QueryRequest(BaseModel):
    question: str
    doc_id: str | None = None  # None = search across all docs
    n_results: int = 5


class SourceChunk(BaseModel):
    doc_name: str
    chunk_index: int
    page_number: int
    text_preview: str


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    docs_referenced: list[str]  # distinct doc_names that showed up in retrieval
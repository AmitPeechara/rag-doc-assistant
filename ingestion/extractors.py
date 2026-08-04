"""
extractors.py
Responsibility: file -> raw text, structured by page (where applicable).
No chunking, no embedding logic here - keep this module single-purpose.

Return shape (for all file types):
    List[dict] where each dict is: {"text": str, "page_number": int | None}

For .md/.txt: single-element list, page_number=None (no pagination concept).
For .pdf: one element per page, page_number=1-indexed page number.
"""

from pathlib import Path
from pypdf import PdfReader


MIN_EXTRACTED_CHARS_THRESHOLD = 50  # below this (for multi-page pdf), flag as likely scanned/image-based


class ExtractionError(Exception):
    """Raised when text extraction fails or produces unusable output."""
    pass


def extract_text(file_path: str) -> list[dict]:
    """
    Main entry point. Dispatches based on file extension.
    Returns list of {"text": ..., "page_number": ...} segments.
    Raises ExtractionError on unreadable/unsupported/empty content.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".md", ".txt"):
        return _extract_plain_text(path)
    elif suffix == ".pdf":
        return _extract_pdf(path)
    else:
        raise ExtractionError(f"Unsupported file type: {suffix}. Supported: .md, .txt, .pdf")


def _extract_plain_text(path: Path) -> list[dict]:
    """Read .md/.txt with encoding fallback."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # fallback for files with non-UTF-8 encoding rather than crashing
        text = path.read_text(encoding="latin-1", errors="replace")

    if not text.strip():
        raise ExtractionError(f"File '{path.name}' is empty or contains no readable text.")

    return [{"text": text, "page_number": None}]


def _extract_pdf(path: Path) -> list[dict]:
    """Read .pdf, extract text per page, detect likely scanned/image-based PDFs."""
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ExtractionError(f"Could not open PDF '{path.name}': {e}")

    if len(reader.pages) == 0:
        raise ExtractionError(f"PDF '{path.name}' has no pages.")

    segments = []
    total_chars = 0

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        total_chars += len(page_text.strip())
        # still include empty pages in output for now - filtered/handled at chunking stage
        segments.append({"text": page_text, "page_number": i + 1})

    # Heuristic: if a multi-page PDF yields almost no text, it's likely scanned/image-based
    if len(reader.pages) > 1 and total_chars < MIN_EXTRACTED_CHARS_THRESHOLD:
        raise ExtractionError(
            f"PDF '{path.name}' appears to be scanned/image-based (extracted only "
            f"{total_chars} chars across {len(reader.pages)} pages). OCR is not supported in v1."
        )

    return segments


if __name__ == "__main__":
    # Standalone smoke test - run directly: python extractors.py <file_path>
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extractors.py <file_path>")
        sys.exit(1)

    try:
        result = extract_text(sys.argv[1])
        print(f"Extracted {len(result)} segment(s) from {sys.argv[1]}")
        for seg in result[:3]:  # preview first 3 segments only
            page_label = f"page {seg['page_number']}" if seg["page_number"] else "single segment"
            preview = seg["text"][:150].replace("\n", " ")
            print(f"  [{page_label}] {preview}...")
    except ExtractionError as e:
        print(f"Extraction failed: {e}")
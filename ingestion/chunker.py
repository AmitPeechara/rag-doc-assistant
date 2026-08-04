"""
chunker.py
Responsibility: extracted text segments -> list of overlapping chunks.
Uses tiktoken for token counting (free, open-source, offline - approximation
of Gemini's tokenizer, but consistent chunk sizing matters more than exact match).

Return shape: List[dict], each: {"text": str, "chunk_index": int, "page_number": int | None}
"""

import tiktoken

# cl100k_base is tiktoken's general-purpose encoding (used by GPT-4 family).
# We're using it purely as a consistent token-counting proxy, not because
# it matches Gemini's actual tokenizer - close enough for chunk sizing.
ENCODING = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def chunk_text(segments: list[dict], chunk_size: int = CHUNK_SIZE_TOKENS,
               overlap: int = CHUNK_OVERLAP_TOKENS) -> list[dict]:
    """
    Takes extracted segments (from extractors.py) and produces overlapping chunks.

    Design decisions baked in here:
    - Chunking works on TOKEN counts (via tiktoken), not raw character/word counts.
    - A chunk is tagged with the page_number of wherever it STARTS (if it spans
      multiple pages/segments, we don't track a range - kept simple for v1).
    - Short documents (total tokens < chunk_size) still produce exactly 1 chunk,
      no overlap logic needed - handled naturally by the windowing loop below.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size, or chunks will never advance.")

    # Step 1: flatten all segments into a single token stream, but remember
    # which page each token range came from, so we can tag chunks correctly.
    all_tokens = []       # flat list of token ids across the whole document
    token_page_map = []   # parallel list: page_number for each token in all_tokens

    for seg in segments:
        seg_tokens = ENCODING.encode(seg["text"])
        all_tokens.extend(seg_tokens)
        token_page_map.extend([seg["page_number"]] * len(seg_tokens))

    if len(all_tokens) == 0:
        return []  # nothing to chunk (e.g., all segments were empty pages)

    # Step 2: slide a window of chunk_size tokens, advancing by (chunk_size - overlap) each step
    chunks = []
    chunk_index = 0
    start = 0
    step = chunk_size - overlap

    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        chunk_token_ids = all_tokens[start:end]
        chunk_text_str = ENCODING.decode(chunk_token_ids)

        # tag with the page of the FIRST token in this chunk
        chunk_page = token_page_map[start]

        chunks.append({
            "text": chunk_text_str,
            "chunk_index": chunk_index,
            "page_number": chunk_page,
        })

        chunk_index += 1

        # stop condition: if this chunk already reached the end, don't loop again
        if end == len(all_tokens):
            break

        start += step

    return chunks


if __name__ == "__main__":
    # Standalone smoke test - run directly: python chunker.py
    # Deliberately tests on PLAIN TEXT first, no PDF/extractor dependency,
    # to isolate chunking-overlap correctness per the agreed test order.

    sample_text = " ".join([f"word{i}" for i in range(1200)])  # ~1200 tokens roughly
    fake_segments = [{"text": sample_text, "page_number": None}]

    result = chunk_text(fake_segments)

    print(f"Total tokens in sample: {count_tokens(sample_text)}")
    print(f"Number of chunks produced: {len(result)}")
    for c in result:
        tok_count = count_tokens(c["text"])
        print(f"  chunk_index={c['chunk_index']} | tokens={tok_count} | page={c['page_number']} | "
              f"preview: {c['text'][:50]}...")

    # Sanity check: verify overlap actually happened between consecutive chunks
    if len(result) >= 2:
        chunk0_words = result[0]["text"].split()
        chunk1_words = result[1]["text"].split()
        overlap_found = any(w in chunk1_words[:CHUNK_OVERLAP_TOKENS + 10] for w in chunk0_words[-CHUNK_OVERLAP_TOKENS:])
        print(f"\nOverlap sanity check between chunk 0 and chunk 1: {'PASS' if overlap_found else 'CHECK MANUALLY'}")

    # Edge case test: short document (fewer tokens than chunk_size)
    short_segments = [{"text": "This is a short test document with very few words.", "page_number": 1}]
    short_result = chunk_text(short_segments)
    print(f"\nShort-doc edge case: {len(short_result)} chunk(s) produced (expected: 1)")
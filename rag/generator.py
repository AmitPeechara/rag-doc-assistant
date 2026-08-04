"""
generator.py
Responsibility: retrieved chunks + question -> prompt construction -> Gemini generation.
Non-streaming version (step 4). Streaming version comes in step 5, built on top of this
once retrieval+generation correctness is verified independently of streaming complexity.
"""
import os
from google import genai
from google.genai.types import GenerateContentConfig

from dotenv import load_dotenv

GENERATION_MODEL = "gemini-3.5-flash-lite"
GENERATION_TEMPERATURE = 0.1  # low - this is fact-retrieval, not creative generation
load_dotenv()
_gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # reads GEMINI_API_KEY from env
SYSTEM_INSTRUCTION = """You are a documentation assistant. Answer the user's question using ONLY
the provided context below. Do not use outside knowledge.

If the context does not contain enough information to answer the question, say so explicitly -
do not guess or fabricate an answer.

If the provided context comes from MULTIPLE DIFFERENT documents (check the source labels), and
the user's question does not specify which document they mean, explicitly point this out in your
answer and ask the user to clarify which document they're interested in, rather than silently
blending information from different sources together.

Always mention which document (and page, if available) your answer is based on."""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    retrieved_chunks: list of {"text": ..., "doc_name": ..., "page_number": ...}
    Question is placed AFTER the context, to avoid the 'lost in the middle' effect
    burying the actual instruction/question under a pile of context.
    """
    if not retrieved_chunks:
        context_block = "(No relevant context was found in the document store.)"
    else:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            page_label = f", page {chunk['page_number']}" if chunk.get("page_number", -1) != -1 else ""
            context_parts.append(
                f"--- Source {i+1}: {chunk['doc_name']}{page_label} ---\n{chunk['text']}"
            )
        context_block = "\n\n".join(context_parts)

    return f"""Context:
{context_block}

Question: {question}"""


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Non-streaming: returns the full answer as a string once generation completes."""
    prompt = build_prompt(question, retrieved_chunks)

    response = _gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=GENERATION_TEMPERATURE,
        ),
    )
    return response.text


def generate_answer_stream(question: str, retrieved_chunks: list[dict]):
    """
    Streaming version - yields text pieces as they arrive from Gemini, instead of
    waiting for the full response. Same prompt/config as generate_answer, just a
    different call method. This is a plain generator (not async) - FastAPI's
    StreamingResponse can consume a sync generator directly.
    """
    prompt = build_prompt(question, retrieved_chunks)

    stream = _gemini_client.models.generate_content_stream(
        model=GENERATION_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=GENERATION_TEMPERATURE,
        ),
    )
    for chunk in stream:
        if chunk.text:  # some chunks may carry no text (e.g., metadata-only chunks)
            yield chunk.text


if __name__ == "__main__":
    # Standalone smoke test - run: python generator.py
    # Tests prompt construction + real Gemini generation call, WITHOUT touching Chroma/retrieval.
    # This isolates generation correctness from retrieval correctness (same pattern as before).

    print("=== Test 1: Single-doc context, clear answer expected ===")
    fake_chunks_single_doc = [
        {"text": "Razorpay base URL is https://api.razorpay.com/v1", "doc_name": "razorpay_sample.md", "page_number": -1},
        {"text": "Razorpay auth uses Basic Auth with API key and secret", "doc_name": "razorpay_sample.md", "page_number": -1},
    ]
    answer = generate_answer("What is the base URL and auth method?", fake_chunks_single_doc)
    print(answer)

    print("\n=== Test 2: Multi-doc context, ambiguous question - should flag ambiguity ===")
    fake_chunks_multi_doc = [
        {"text": "Razorpay base URL is https://api.razorpay.com/v1", "doc_name": "razorpay_sample.md", "page_number": -1},
        {"text": "Stripe base URL is https://api.stripe.com/v1", "doc_name": "stripe_sample.md", "page_number": -1},
    ]
    answer = generate_answer("Give me the endpoint details", fake_chunks_multi_doc)
    print(answer)

    print("\n=== Test 3: Empty context - should say it doesn't know, not fabricate ===")
    answer = generate_answer("What is the rate limit for the XYZ API?", [])
    print(answer)

    print("\n=== Test 4: Streaming version - should print incrementally ===")
    for piece in generate_answer_stream("What is the base URL and auth method?", fake_chunks_single_doc):
        print(piece, end="", flush=True)
    print()  # newline after stream completes
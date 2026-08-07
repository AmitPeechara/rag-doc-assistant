# """
# Standalone Chroma smoke test.
# Run this FIRST, before wiring Chroma into the actual project.
# Goal: confirm persistence, collection creation, add, and metadata-filtered query all work.
# """

# import chromadb

# # PersistentClient writes to disk at the given path (NFR7 - persistence across restarts)
# client = chromadb.PersistentClient(path="./chroma_db")

# # get_or_create so re-running this script doesn't blow up on a second run
# collection = client.get_or_create_collection(name="doc_chunks")

# # --- Simulate two "documents" with fake embeddings (real embeddings come from Gemini later) ---
# # Using tiny fake vectors just to prove the mechanics work - NOT real semantic embeddings
# fake_embedding_dim = 8

# collection.add(
#     ids=["doc1_chunk0", "doc1_chunk1"],
#     embeddings=[[0.1]*fake_embedding_dim, [0.2]*fake_embedding_dim],
#     documents=["Razorpay base URL is https://api.razorpay.com/v1", "Razorpay auth uses Basic Auth with API key/secret"],
#     metadatas=[
#         {"doc_id": "doc1", "doc_name": "razorpay_sample.md", "chunk_index": 0},
#         {"doc_id": "doc1", "doc_name": "razorpay_sample.md", "chunk_index": 1},
#     ],
# )

# collection.add(
#     ids=["doc2_chunk0"],
#     embeddings=[[0.9]*fake_embedding_dim],
#     documents=["Stripe base URL is https://api.stripe.com/v1"],
#     metadatas=[
#         {"doc_id": "doc2", "doc_name": "stripe_sample.md", "chunk_index": 0},
#     ],
# )

# print(f"Total chunks in collection: {collection.count()}")

# # --- Test 1: query WITHOUT doc_id filter (searches across all docs) ---
# print("\n--- Query without doc_id filter ---")
# results = collection.query(
#     query_embeddings=[[0.15]*fake_embedding_dim],
#     n_results=3,
# )
# for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
#     print(f"  [{meta['doc_name']}] {doc}")

# # --- Test 2: query WITH doc_id filter (this is the isolation test - NFR3) ---
# print("\n--- Query WITH doc_id filter (doc1 only) ---")
# results = collection.query(
#     query_embeddings=[[0.15]*fake_embedding_dim],
#     n_results=3,
#     where={"doc_id": "doc1"},  # metadata filtering - this is the key mechanism for isolation
# )
# for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
#     print(f"  [{meta['doc_name']}] {doc}")

# print("\nIf Test 2 only shows razorpay_sample.md chunks, metadata filtering/isolation works correctly.")
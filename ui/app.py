"""
ui/app.py
Streamlit UI for the RAG doc assistant (step 6).
Talks to the FastAPI backend (must be running separately on localhost:8000).

Handles the JSON-header-then-raw-text streaming format from /query/stream:
first line is JSON (sources + docs_referenced), everything after streams as
the raw answer text.
"""

import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Doc Assistant", layout="wide")
st.title("RAG Documentation Assistant")

# ---------- Sidebar: upload + doc selector (FR1, FR3) ----------
with st.sidebar:
    st.header("Documents")

    uploaded_file = st.file_uploader("Upload a doc", type=["md", "txt", "pdf"])
    if uploaded_file is not None:
        if st.button("Ingest document"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Ingested '{data['doc_name']}' - {data['chunk_count']} chunks.")
                    else:
                        # NFR4 - surface real errors, not silent failures
                        st.error(f"Ingestion failed: {resp.json().get('detail', resp.text)}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach API: {e}")

    st.divider()

    # Fetch current doc list for the selector (FR3)
    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=10)
        docs = docs_resp.json() if docs_resp.status_code == 200 else []
    except requests.exceptions.RequestException:
        docs = []
        st.warning("Could not reach API to list documents. Is the backend running?")

    doc_options = {"Search all documents": None}
    for d in docs:
        doc_options[f"{d['doc_name']} ({d['chunk_count']} chunks)"] = d["doc_id"]

    selected_label = st.selectbox("Scope question to:", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_label]

    if docs:
        st.caption(f"{len(docs)} document(s) ingested.")
    else:
        st.caption("No documents ingested yet.")

# ---------- Main panel: chat/query (FR2, streaming) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "sources": [...]}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} chunks used)"):
                for s in msg["sources"]:
                    page_label = f", page {s['page_number']}" if s["page_number"] != -1 else ""
                    st.markdown(f"**{s['doc_name']}{page_label}** (chunk {s['chunk_index']})")
                    st.caption(s["text_preview"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            resp = requests.post(
                f"{API_BASE}/query/stream",
                json={"question": question, "doc_id": selected_doc_id, "n_results": 5},
                stream=True,
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")
            resp = None

        if resp is not None and resp.status_code == 200:
            # Parse the stream: first line is JSON (sources), rest is raw answer text.
            byte_iterator = resp.iter_lines(decode_unicode=True)

            # First line: sources header
            header_line = next(byte_iterator, None)
            sources = []
            docs_referenced = []
            if header_line:
                try:
                    header = json.loads(header_line)
                    sources = header.get("sources", [])
                    docs_referenced = header.get("docs_referenced", [])
                except json.JSONDecodeError:
                    # if parsing fails, treat the "header" as part of the answer instead
                    # of silently dropping content
                    pass

            # Warn early if the question was ambiguous across multiple docs (structured
            # signal, not relying solely on the LLM's prose self-report from generator.py)
            if selected_doc_id is None and len(docs_referenced) > 1:
                st.info(f"Note: retrieved context spans multiple documents: {', '.join(docs_referenced)}")

            def text_stream():
                # iter_lines strips newlines, which would mangle spacing across chunks -
                # re-add a space between lines to avoid words getting jammed together
                for line in byte_iterator:
                    yield line + " " if line else " "

            full_answer = st.write_stream(text_stream())

            if sources:
                with st.expander(f"Sources ({len(sources)} chunks used)"):
                    for s in sources:
                        page_label = f", page {s['page_number']}" if s["page_number"] != -1 else ""
                        st.markdown(f"**{s['doc_name']}{page_label}** (chunk {s['chunk_index']})")
                        st.caption(s["text_preview"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })
        elif resp is not None:
            error_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            st.error(f"Query failed: {error_detail}")
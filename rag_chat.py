"""
rag_chat.py — RAG-based Deep Chat for Assyrian-AI Data Analyst v2
Drop this file into the root of the Assyrian-AI-Data-Analyst repo.

Uses FAISS (local, in-memory) + sentence-transformers (free, no API key needed for embeddings)
to retrieve the most relevant rows before calling the LLM.
This gives grounded, cited answers over the actual data — not just a summary.
"""

import re
import json
import numpy as np
import pandas as pd
import requests
import os

# ── Lazy imports so the rest of the app still works if FAISS isn't installed ──

def _import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def _import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        return None


# ─────────────────────────────────────────────
# CHUNKING
# Convert DataFrame rows into text chunks for embedding
# ─────────────────────────────────────────────

def df_to_chunks(df: pd.DataFrame, chunk_size: int = 10) -> list[dict]:
    """
    Convert a DataFrame into text chunks.
    Each chunk covers `chunk_size` rows, serialized as a readable text block.
    Returns a list of dicts: {text, start_row, end_row}
    """
    chunks = []
    cols = df.columns.tolist()
    total = len(df)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        slice_df = df.iloc[start:end]

        lines = [f"Rows {start + 1}–{end} of {total}:"]
        for _, row in slice_df.iterrows():
            row_parts = [f"{col}={repr(row[col])}" for col in cols]
            lines.append("  " + ", ".join(row_parts))

        chunks.append({
            "text": "\n".join(lines),
            "start_row": start,
            "end_row": end - 1,
        })

    return chunks


# ─────────────────────────────────────────────
# INDEX BUILDING
# ─────────────────────────────────────────────

def build_faiss_index(chunks: list[dict], model) -> tuple:
    """
    Embed all chunks using the SentenceTransformer model,
    build a FAISS flat L2 index, and return (index, embeddings_array).
    """
    faiss = _import_faiss()
    if faiss is None:
        raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(
    question: str,
    index,
    chunks: list[dict],
    model,
    top_k: int = 5
) -> list[dict]:
    """
    Embed the question, search FAISS for the nearest chunks,
    return the top_k most relevant chunks.
    """
    q_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)

    return results


# ─────────────────────────────────────────────
# LLM CALL (reuses OpenRouter, same as rest of portfolio)
# ─────────────────────────────────────────────

def call_openrouter_rag(question: str, context_chunks: list[dict], model_name: str = "nvidia/nemotron-3-super-120b-a12b:free") -> str:
    """
    Send retrieved context + question to OpenRouter.
    Returns the model's answer as a string.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return "⚠️ No OpenRouter API key found. Paste your key in the sidebar."

    context_text = "\n\n---\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""You are a data analyst assistant. Answer the user's question using ONLY the data rows provided below as context. Be specific — reference actual values from the data. If the answer isn't in the provided rows, say so clearly.

Retrieved data context:
{context_text}

User question: {question}

Answer:"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    raw = data["choices"][0]["message"]["content"]

    # Robust extraction — handles reasoning model output
    # If content has JSON-like structure embedded, extract just the answer text
    return raw.strip()


# ─────────────────────────────────────────────
# FULL RAG PIPELINE
# ─────────────────────────────────────────────

def rag_answer(
    question: str,
    index,
    chunks: list[dict],
    model,
    top_k: int = 5,
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
) -> dict:
    """
    End-to-end RAG: retrieve relevant chunks, call LLM, return answer + sources.
    Returns: {answer: str, sources: list[dict]}
    """
    relevant = retrieve_relevant_chunks(question, index, chunks, model, top_k=top_k)
    answer = call_openrouter_rag(question, relevant, model_name=openrouter_model)

    return {
        "answer": answer,
        "sources": relevant,
    }

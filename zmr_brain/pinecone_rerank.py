"""
Pinecone Inference API — rerank documents by relevance to a query.

Uses ``Pinecone(api_key=...).inference.rerank`` (models such as ``pinecone-rerank-v0``,
``bge-reranker-v2-m3``, ``cohere-rerank-3.5``). Same API key as vector indexes.

Docs: https://docs.pinecone.io/reference/api/inference/rerank
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from zmr_brain.retrieval import RetrievedChunk


def rerank_chunks_pinecone(
    query: str,
    chunks: List["RetrievedChunk"],
    *,
    top_n: int,
    model: Optional[str] = None,
) -> List["RetrievedChunk"]:
    """
    Re-order ``RetrievedChunk`` rows using Pinecone's rerank model.

    ``chunks`` must preserve the same order as the ``documents`` list sent to the API
    (ranked results use the ``index`` field pointing back into that list).
    """
    if not chunks:
        return []
    n = min(int(top_n), len(chunks))
    if n <= 0:
        return []

    from zmr_brain.clients import get_pinecone_client

    m = (model or os.getenv("PINECONE_RERANK_MODEL", "pinecone-rerank-v0")).strip()
   # max_chars = int(os.getenv("PINECONE_RERANK_MAX_CHARS", "4000"))
    max_chars = int(os.getenv("PINECONE_RERANK_MAX_CHARS", "2000"))


    documents: List[dict] = []
    for c in chunks:
        t = (c.text or "").strip()
        if not t:
            t = " "
        if len(t) > max_chars:
            t = t[:max_chars]
        documents.append({"id": c.vector_id, "text": t})

    pc = get_pinecone_client()
    try:
        result = pc.inference.rerank(
            model=m,
            query=query.strip(),
            documents=documents,
            rank_fields=["text"],
            top_n=n,
            return_documents=False,
            parameters={"truncate": "END"},
        )
    except Exception:
        result = pc.inference.rerank(
            model=m,
            query=query.strip(),
            documents=documents,
            rank_fields=["text"],
            top_n=n,
            return_documents=False,
        )

    data = getattr(result, "data", None)
    if not data:
        return chunks[:n]

    ordered: List["RetrievedChunk"] = []
    seq = list(chunks)
    for rank, rd in enumerate(data, start=1):
        try:
            idx = int(rd.index)
            base = seq[idx]
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        rs = getattr(rd, "score", None)
        score = float(rs) if rs is not None else None
        ordered.append(
            replace(
                base,
                rank=rank,
                score=score,
                pinecone_rerank_score=score,
            )
        )
    return ordered if ordered else chunks[:n]

"""
ZMR Brain query API — 3-tier RBAC with metadata filtering.

Run from repository root:
  pip install -r requirements.txt
  python -m uvicorn zmr_brain.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import os
import queue
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from zmr_brain.answer import answer_with_claude
from zmr_brain.constants import (
    DEFAULT_RERANK_POOL,
    DEFAULT_RRF_K,
    DEFAULT_USER_ROLE,
    PINECONE_INDEX,
    ACCESS_TIERS,
    namespaces_for_email,
    pinecone_access_filter,
)
from zmr_brain.meta_queries import chatbot_meta_reply
from zmr_brain.query_graph import run_query_graph, stream_query_graph
from zmr_brain.query_reformulate import reformulate_query_for_retrieval
from zmr_brain.query_routing import OUT_OF_SCOPE_REPLY, classify_query
from zmr_brain.retrieval import RetrievedChunk, retrieve_for_query

app = FastAPI(title="ZMR Brain", version="0.2.0")

# Docker entrypoint creates this after API + Streamlit pass readiness (see docker/entrypoint.sh).
_STRICT_ST_HEALTH_FILE = "/tmp/zmr-streamlit-health-strict"

_cors_raw = (os.getenv("ZMR_CORS_ORIGINS") or "").strip()
if _cors_raw:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if _cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    user_email: str = Field(
        default="",
        description="User email (zmrcapital.com). Determines access tier for RBAC filtering.",
    )
    user_role: str = Field(
        default=DEFAULT_USER_ROLE,
        description="Legacy role field. Kept for backward compatibility.",
    )
    top_k: int = Field(default=8, ge=1, le=50)
    embed_model: Optional[str] = Field(
        default=None,
        description="Voyage model for query embedding (e.g. voyage-finance-2, voyage-3-large).",
    )
    filter_file_sha256: Optional[str] = Field(
        default=None,
        description="Optional Pinecone metadata filter: file_sha256 exact match.",
    )
    generate_answer: bool = Field(
        default=False,
        description="If true, call Claude with retrieved passages (requires ANTHROPIC_API_KEY).",
    )
    hybrid_rrf: bool = Field(
        default=True,
        description="Merge dense (Pinecone) + lexical (BM25) ranks with Reciprocal Rank Fusion.",
    )
    rrf_k: int = Field(
        default=DEFAULT_RRF_K, ge=1, le=200,
        description="RRF damping constant (typical 60).",
    )
    candidate_pool: Optional[int] = Field(
        default=None, ge=5, le=200,
    )
    pinecone_rerank: bool = Field(default=True)
    rerank_pool: int = Field(default=DEFAULT_RERANK_POOL, ge=5, le=200)
    pinecone_rerank_model: Optional[str] = Field(default=None)
    lexical_mode: str = Field(
        default="bm25",
        description="Keyword leg: bm25, fts, or both.",
    )
    skip_query_reformulation: bool = Field(
        default=False,
        description="If true, skip the Claude Haiku query-rewrite step before retrieval (lower latency).",
    )


class ChunkOut(BaseModel):
    rank: int
    score: Optional[float] = None
    vector_id: str
    doc_name: Optional[str] = None
    source_path: Optional[str] = None
    text: Optional[str] = None
    rrf_score: Optional[float] = None
    semantic_score: Optional[float] = None
    pinecone_rerank_score: Optional[float] = None
    sheet_name: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    gcs_uri: Optional[str] = None
    pinecone_metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    pinecone_index: str
    user_email: str
    user_role: str
    query: str
    chunks: List[ChunkOut]
    answer: Optional[str] = None


class QueryGraphResponse(QueryResponse):
    error: Optional[str] = None
    graph_trace: List[str] = Field(default_factory=list)
    meta_intro: bool = False
    refuse_out_of_scope: bool = False
    retrieval_query: Optional[str] = None


def _serialize_graph_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe view of accumulated graph state for NDJSON streaming."""
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if k == "chunks" and v is not None:
            chunks = v
            if chunks and isinstance(chunks[0], RetrievedChunk):
                out[k] = [_chunk_to_out(c).model_dump() for c in chunks]
            else:
                out[k] = chunks
        else:
            out[k] = v
    return out


def _chunk_to_out(c: RetrievedChunk) -> ChunkOut:
    return ChunkOut(
        rank=c.rank,
        score=c.score,
        vector_id=c.vector_id,
        doc_name=c.doc_name,
        source_path=c.source_path,
        text=c.text,
        rrf_score=c.rrf_score,
        semantic_score=c.semantic_score,
        pinecone_rerank_score=c.pinecone_rerank_score,
        sheet_name=c.sheet_name,
        chunk_index=c.chunk_index,
        total_chunks=c.total_chunks,
        gcs_uri=c.gcs_uri,
        pinecone_metadata=dict(c.pinecone_metadata or {}),
    )


@app.get("/health")
def health():
    body: Dict[str, Any] = {"status": "ok", "service": "zmr-brain", "index": PINECONE_INDEX}
    strict_st = os.path.isfile(_STRICT_ST_HEALTH_FILE) or os.getenv(
        "ZMR_CHECK_STREAMLIT_HEALTH", ""
    ).strip().lower() in ("1", "true", "yes")
    if strict_st:
        st_port = (os.getenv("ST_PORT") or "8501").strip() or "8501"
        url = f"http://127.0.0.1:{st_port}/_stcore/health"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status != 200:
                    return JSONResponse(
                        status_code=503,
                        content={**body, "streamlit": "unhealthy", "http_status": resp.status},
                    )
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return JSONResponse(
                status_code=503,
                content={**body, "streamlit": "unreachable", "error": str(e)},
            )
        body["streamlit"] = "ok"
    return body


@app.get("/v1/access-tiers")
def list_access_tiers() -> Dict[str, Any]:
    return {"tiers": list(ACCESS_TIERS), "index": PINECONE_INDEX}


@app.post("/v1/query", response_model=QueryResponse)
def post_query(body: QueryRequest) -> QueryResponse:
    user_email = (body.user_email or "").strip().lower()
    user_ns = namespaces_for_email(user_email)

    flt: Optional[Dict[str, Any]] = None
    if body.filter_file_sha256:
        flt = {"file_sha256": {"$eq": body.filter_file_sha256.strip()}}

    kind = classify_query(body.query)
    chunks: List[RetrievedChunk] = []
    answer: Optional[str] = None

    if kind == "intro":
        answer = chatbot_meta_reply(body.query) if body.generate_answer else None
    elif kind == "refuse":
        answer = OUT_OF_SCOPE_REPLY if body.generate_answer else None
    else:
        retrieval_q = reformulate_query_for_retrieval(body.query)
        try:
            chunks = retrieve_for_query(
                retrieval_q,
                PINECONE_INDEX,
                top_k=body.top_k,
                embed_model=body.embed_model,
                metadata_filter=flt,
                namespaces=user_ns,
                include_empty_text=False,
                hybrid_rrf=body.hybrid_rrf,
                rrf_k=body.rrf_k,
                candidate_pool=body.candidate_pool,
                pinecone_rerank=body.pinecone_rerank,
                rerank_pool=body.rerank_pool,
                pinecone_rerank_model=body.pinecone_rerank_model,
                lexical_mode=body.lexical_mode,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}") from e

        if body.generate_answer:
            try:
                answer = answer_with_claude(body.query, chunks)
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Answer generation failed: {e}"
                ) from e

    idx_report = ",".join(user_ns) if user_ns else PINECONE_INDEX
    return QueryResponse(
        pinecone_index=idx_report,
        user_email=user_email,
        user_role=body.user_role.strip().lower(),
        query=body.query,
        chunks=[_chunk_to_out(c) for c in chunks],
        answer=answer,
    )


@app.post("/v1/query/graph", response_model=QueryGraphResponse)
def post_query_graph(body: QueryRequest) -> QueryGraphResponse:
    user_email = (body.user_email or "").strip().lower()
    flt: Optional[Dict[str, Any]] = None
    if body.filter_file_sha256:
        flt = {"file_sha256": {"$eq": body.filter_file_sha256.strip()}}

    try:
        final = run_query_graph(
            body.query,
            body.user_role,
            user_email=user_email,
            top_k=body.top_k,
            embed_model=body.embed_model,
            metadata_filter=flt,
            skip_query_reformulation=body.skip_query_reformulation,
            generate_answer=body.generate_answer,
            hybrid_rrf=body.hybrid_rrf,
            rrf_k=body.rrf_k,
            candidate_pool=body.candidate_pool,
            pinecone_rerank=body.pinecone_rerank,
            rerank_pool=body.rerank_pool,
            pinecone_rerank_model=body.pinecone_rerank_model,
            lexical_mode=body.lexical_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph run failed: {e}") from e

    err = final.get("error")
    chunks = final.get("chunks") or []
    idx = final.get("pinecone_index") or PINECONE_INDEX

    if err and not chunks:
        bad_input = "query is empty" in (err or "")
        raise HTTPException(status_code=400 if bad_input else 503, detail=err)

    return QueryGraphResponse(
        pinecone_index=idx,
        user_email=user_email,
        user_role=body.user_role.strip().lower(),
        query=body.query,
        chunks=[_chunk_to_out(c) for c in chunks],
        answer=final.get("answer"),
        error=err,
        graph_trace=list(final.get("graph_trace") or []),
        meta_intro=bool(final.get("meta_intro")),
        refuse_out_of_scope=bool(final.get("refuse_out_of_scope")),
        retrieval_query=final.get("retrieval_query"),
    )


def _graph_stream_events(
    body: QueryRequest, user_email: str, flt: Optional[Dict[str, Any]]
) -> Iterator[str]:
    """
    NDJSON lines: {"node":..., "state":{...}} per graph step, {"heartbeat":true} if idle
    (keeps Railway / proxies from closing long requests that have no bytes for ~30s).
    """
    q: queue.Queue[Tuple[str, Any]] = queue.Queue(maxsize=64)

    def producer() -> None:
        try:
            for node_name, state in stream_query_graph(
                body.query,
                body.user_role,
                user_email=user_email,
                top_k=body.top_k,
                embed_model=body.embed_model,
                metadata_filter=flt,
                skip_query_reformulation=body.skip_query_reformulation,
                generate_answer=body.generate_answer,
                hybrid_rrf=body.hybrid_rrf,
                rrf_k=body.rrf_k,
                candidate_pool=body.candidate_pool,
                pinecone_rerank=body.pinecone_rerank,
                rerank_pool=body.rerank_pool,
                pinecone_rerank_model=body.pinecone_rerank_model,
                lexical_mode=body.lexical_mode,
            ):
                q.put(("step", (node_name, state)))
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            q.put(("done", None))

    hb_sec = float(os.getenv("ZMR_GRAPH_STREAM_HEARTBEAT_SEC", "12"))
    th = threading.Thread(target=producer, daemon=True)
    th.start()
    while True:
        try:
            kind, payload = q.get(timeout=hb_sec)
        except queue.Empty:
            yield json.dumps({"heartbeat": True}) + "\n"
            continue
        if kind == "done":
            break
        if kind == "error":
            yield json.dumps({"error": payload}) + "\n"
            break
        if kind == "step":
            node_name, state = payload
            ser = _serialize_graph_state(dict(state))
            yield json.dumps({"node": node_name, "state": ser}) + "\n"


@app.post("/v1/query/graph/stream")
def post_query_graph_stream(body: QueryRequest) -> StreamingResponse:
    user_email = (body.user_email or "").strip().lower()
    flt: Optional[Dict[str, Any]] = None
    if body.filter_file_sha256:
        flt = {"file_sha256": {"$eq": body.filter_file_sha256.strip()}}

    return StreamingResponse(
        _graph_stream_events(body, user_email, flt),
        media_type="application/x-ndjson",
    )

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
    ACCESS_TIERS,
    PINECONE_INDEX,
    PINECONE_INDEX_BY_TIER,
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


def _invoke_query_graph(body: QueryRequest) -> Dict[str, Any]:
    user_email = (body.user_email or "").strip().lower()
    flt: Optional[Dict[str, Any]] = None
    if body.filter_file_sha256:
        flt = {"file_sha256": {"$eq": body.filter_file_sha256.strip()}}
    return run_query_graph(
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


def _query_graph_response_from_final(
    body: QueryRequest, final: Dict[str, Any]
) -> QueryGraphResponse:
    user_email = (body.user_email or "").strip().lower()
    err = final.get("error")
    chunks = final.get("chunks") or []
    idx = final.get("pinecone_index") or PINECONE_INDEX
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


def _serialize_graph_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe copy of graph state for NDJSON (chunks → ChunkOut dicts)."""
    out: Dict[str, Any] = dict(state)
    raw = out.get("chunks") or []
    serial: List[Dict[str, Any]] = []
    for c in raw:
        if isinstance(c, RetrievedChunk):
            serial.append(_chunk_to_out(c).model_dump())
        elif isinstance(c, dict):
            serial.append(dict(c))
        else:
            serial.append(_chunk_to_out(c).model_dump())
    out["chunks"] = serial
    return out


def _stream_query_graph_kwargs(body: QueryRequest) -> Dict[str, Any]:
    user_email = (body.user_email or "").strip().lower()
    flt: Optional[Dict[str, Any]] = None
    if body.filter_file_sha256:
        flt = {"file_sha256": {"$eq": body.filter_file_sha256.strip()}}
    return {
        "user_email": user_email,
        "top_k": body.top_k,
        "embed_model": body.embed_model,
        "metadata_filter": flt,
        "skip_query_reformulation": body.skip_query_reformulation,
        "generate_answer": body.generate_answer,
        "hybrid_rrf": body.hybrid_rrf,
        "rrf_k": body.rrf_k,
        "candidate_pool": body.candidate_pool,
        "pinecone_rerank": body.pinecone_rerank,
        "rerank_pool": body.rerank_pool,
        "pinecone_rerank_model": body.pinecone_rerank_model,
        "lexical_mode": body.lexical_mode,
    }


def _graph_stream_events(body: QueryRequest) -> Iterator[bytes]:
    hb_sec = max(1.0, float(os.getenv("ZMR_GRAPH_STREAM_HEARTBEAT_SEC", "12")))
    hb_line = (json.dumps({"heartbeat": True}) + "\n").encode("utf-8")
    q: queue.Queue[Tuple[Any, ...]] = queue.Queue()

    def producer() -> None:
        try:
            kw = _stream_query_graph_kwargs(body)
            for node_name, st in stream_query_graph(body.query, body.user_role, **kw):
                q.put(("step", node_name, st))
        except Exception as e:
            q.put(("exc", e))
            return
        q.put(("end",))

    threading.Thread(target=producer, daemon=True).start()

    while True:
        try:
            item = q.get(timeout=hb_sec)
        except queue.Empty:
            yield hb_line
            continue
        kind = item[0]
        if kind == "step":
            node_name = item[1]
            state = item[2]
            payload = {
                "node": node_name,
                "state": _serialize_graph_state(dict(state)),
            }
            yield (json.dumps(payload, default=str) + "\n").encode("utf-8")
        elif kind == "exc":
            err = item[1]
            yield (json.dumps({"error": str(err)}) + "\n").encode("utf-8")
            return
        elif kind == "end":
            break


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


@app.get("/v1/retrieval-status")
def retrieval_status() -> Dict[str, Any]:
    """Lightweight wiring check: Postgres row counts / pinecone_index mix vs configured Pinecone indexes."""
    out: Dict[str, Any] = {
        "postgres_ok": False,
        "postgres_error": None,
        "chunk_tables": {},
        "pinecone_indexes_configured": dict(PINECONE_INDEX_BY_TIER),
        "pinecone_per_index": {},
        "pinecone_indexes_queried_for_full_tier_user": namespaces_for_email(
            "nobody@zmrcapital.com"
        ),
        "pinecone_indexes_queried_for_executive_example": namespaces_for_email(
            "zamir@zmrcapital.com"
        ),
        "hints": [
            "Set ZMR_PINECONE_INDEX_* env vars if Postgres `pinecone_index` values do not match "
            "the names under pinecone_indexes_configured.",
            "If Pinecone shows vectors but chunk tables are empty (or vice versa), run ingest "
            "against this DATABASE_URL or fix DB vs Pinecone project mismatch.",
            "Rows with empty chunk_text and no readable GCS body are dropped before the LLM — "
            "re-ingest or fix GCS credentials if hits exist but answers say no passages.",
        ],
    }

    try:
        from zmr_brain.retrieval import pg_connect, pg_release

        conn, cursor_factory = pg_connect()
        try:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                out["postgres_ok"] = True
                for table in ("chunks_v2", "chunks"):
                    try:
                        cur.execute(f"SELECT COUNT(*)::bigint AS n FROM {table}")
                        row = cur.fetchone()
                        n = int(row["n"]) if row else 0
                        by_idx: Dict[str, int] = {}
                        try:
                            cur.execute(
                                f"SELECT pinecone_index, COUNT(*)::bigint AS n "
                                f"FROM {table} GROUP BY 1 ORDER BY 2 DESC"
                            )
                            for r in cur.fetchall():
                                key = r["pinecone_index"] or ""
                                by_idx[str(key)] = int(r["n"])
                        except Exception:
                            pass
                        out["chunk_tables"][table] = {"total_rows": n, "by_pinecone_index": by_idx}
                    except Exception as e:
                        out["chunk_tables"][table] = {"error": str(e)}
                # Why passages can be empty even when GCS credentials work: DB may point at other buckets.
                try:
                    cur.execute(
                        """
                        SELECT substring(chunk_gcs_uri FROM '^gs://([^/]+)') AS bucket,
                               COUNT(*)::bigint AS n
                        FROM chunks_v2
                        WHERE chunk_gcs_uri ~ '^gs://[^/]+/'
                        GROUP BY 1
                        ORDER BY 2 DESC
                        LIMIT 25
                        """
                    )
                    gs_buckets = {
                        str(r["bucket"]): int(r["n"])
                        for r in cur.fetchall()
                        if r and r.get("bucket")
                    }
                    cur.execute(
                        """
                        SELECT
                          COUNT(*) FILTER (WHERE chunk_gcs_uri ~ '^gs://[^/]+/'
                            AND (chunk_text IS NULL OR btrim(chunk_text::text) = '')
                          )::bigint AS empty_inline_with_gs,
                          COUNT(*) FILTER (WHERE chunk_gcs_uri IS NOT NULL
                            AND btrim(chunk_gcs_uri::text) <> ''
                            AND chunk_gcs_uri !~ '^gs://[^/]+/'
                            AND chunk_gcs_uri !~ '^local:'
                          )::bigint AS non_gs_non_local_uri
                        FROM chunks_v2
                        """
                    )
                    row_w = cur.fetchone() or {}
                    env_bucket = (os.getenv("GCS_ARTIFACTS_BUCKET") or "").strip()
                    top = max(gs_buckets, key=gs_buckets.get) if gs_buckets else None
                    out["chunk_body_wiring"] = {
                        "chunks_v2_gs_bucket_row_counts": gs_buckets,
                        "chunks_v2_empty_inline_text_with_gs_uri": int(
                            row_w.get("empty_inline_with_gs") or 0
                        ),
                        "chunks_v2_non_gs_non_local_uri_rows": int(
                            row_w.get("non_gs_non_local_uri") or 0
                        ),
                        "GCS_ARTIFACTS_BUCKET_env": env_bucket or None,
                        "top_gs_bucket_matches_env": (
                            top == env_bucket if (top and env_bucket) else None
                        ),
                    }
                except Exception as e:
                    out["chunk_body_wiring"] = {"error": str(e)}
        finally:
            pg_release(conn)
    except Exception as e:
        out["postgres_error"] = str(e)

    from zmr_brain.clients import get_pinecone_index

    for name in sorted(set(PINECONE_INDEX_BY_TIER.values())):
        try:
            idx = get_pinecone_index(name)
            stats = idx.describe_index_stats()
            vec_total: Optional[int] = None
            if hasattr(stats, "total_vector_count"):
                vec_total = int(getattr(stats, "total_vector_count") or 0)
            elif isinstance(stats, dict):
                vec_total = int(stats.get("total_vector_count") or 0)
                if not vec_total and stats.get("namespaces"):
                    ns = stats["namespaces"]
                    vec_total = sum(
                        int(v.get("vector_count", 0) or 0)
                        for v in ns.values()
                        if isinstance(v, dict)
                    )
            out["pinecone_per_index"][name] = {
                "ok": True,
                "total_vector_count": vec_total,
            }
        except Exception as e:
            out["pinecone_per_index"][name] = {"ok": False, "error": str(e)}

    try:
        from zmr_brain.gcs_client import gcs_bucket_probe, gcs_credentials_mode

        gcs = gcs_credentials_mode()
        gcs["bucket_probe"] = gcs_bucket_probe()
        out["gcs"] = gcs
    except Exception as e:
        out["gcs"] = {"error": str(e)}

    sha = (os.getenv("RAILWAY_GIT_COMMIT_SHA") or "").strip()
    out["deploy"] = {"railway_git_commit_sha": sha or None}
    out["api_env"] = {
        "voyage_api_key_set": bool((os.getenv("VOYAGE_API_KEY") or "").strip()),
        "pinecone_api_key_set": bool((os.getenv("PINECONE_API_KEY") or "").strip()),
        "anthropic_api_key_set": bool((os.getenv("ANTHROPIC_API_KEY") or "").strip()),
    }

    cw = out.get("chunk_body_wiring")
    if isinstance(cw, dict) and cw.get("top_gs_bucket_matches_env") is False:
        out["hints"].append(
            "chunk_body_wiring.top_gs_bucket_matches_env is false: most `chunk_gcs_uri` rows live in a "
            "different bucket than GCS_ARTIFACTS_BUCKET — fix the env var or re-ingest so URIs match a "
            "bucket this service account can read."
        )

    return out


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
    try:
        final = _invoke_query_graph(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph run failed: {e}") from e

    err = final.get("error")
    chunks = final.get("chunks") or []

    if err and not chunks:
        bad_input = "query is empty" in (err or "")
        raise HTTPException(status_code=400 if bad_input else 503, detail=err)

    return _query_graph_response_from_final(body, final)


@app.post("/v1/query/graph/stream")
def post_query_graph_stream(body: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        _graph_stream_events(body),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )

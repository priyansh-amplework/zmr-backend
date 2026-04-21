"""Embed query → Pinecone → Postgres chunk text (Engineer 2 retrieval path)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.errors
import psycopg2.extras
import psycopg2.pool

from zmr_brain.constants import (
    DEFAULT_HYBRID_CANDIDATE_POOL,
    DEFAULT_RERANK_POOL,
    DEFAULT_RRF_K,
    DEFAULT_VOYAGE_QUERY_MODEL,
    PINECONE_INDEX,
    TIER_NAMESPACE,
)

import sys
import threading as _threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from db_url import ensure_ssl_for_managed  # noqa: E402

_pg_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pg_pool_lock = _threading.Lock()


def _get_pg_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    with _pg_pool_lock:
        if _pg_pool is None:
            url = os.getenv("DATABASE_URL", "").strip()
            if not url:
                raise RuntimeError("DATABASE_URL is not set")
            url = ensure_ssl_for_managed(url)
            _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=8, dsn=url,
            )
    return _pg_pool


def pg_connect() -> Tuple[Any, Any]:
    """Borrow a connection from the pool. Caller MUST call ``pg_release(conn)``."""
    conn = _get_pg_pool().getconn()
    return conn, psycopg2.extras.RealDictCursor


def pg_release(conn: Any) -> None:
    """Return a connection to the pool (instead of closing it)."""
    try:
        _get_pg_pool().putconn(conn)
    except Exception:
        pass


def fetch_chunks_by_vector_ids(
    conn: Any,
    cursor_factory: Any,
    ids: List[str],
    pinecone_index: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Map pinecone_vector_id → row. If pinecone_index is set, disambiguate multi-index upserts."""
    if not ids:
        return {}
    index_clause = " AND pinecone_index = %s" if pinecone_index else ""
    params: Tuple[Any, ...] = (ids,) if not pinecone_index else (ids, pinecone_index)

    index_clause_c = index_clause.replace("pinecone_index", "c.pinecone_index")
    sql_with_gcs = f"""
            SELECT c.pinecone_vector_id, c.chunk_text, c.chunk_gcs_uri, c.chunk_gcs_generation,
                   c.metadata, c.document_id, c.chunk_index, c.total_chunks, c.pinecone_index,
                   d.name AS join_doc_name, d.source_path AS join_source_path
            FROM chunks c
            INNER JOIN documents d ON d.id = c.document_id
            WHERE c.pinecone_vector_id = ANY(%s){index_clause_c}
            """
    sql_basic = f"""
            SELECT c.pinecone_vector_id, c.chunk_text, c.metadata, c.document_id, c.chunk_index, c.total_chunks, c.pinecone_index,
                   d.name AS join_doc_name, d.source_path AS join_source_path
            FROM chunks c
            INNER JOIN documents d ON d.id = c.document_id
            WHERE c.pinecone_vector_id = ANY(%s){index_clause_c}
            """
    sql_v2_with_gcs = f"""
            SELECT c.pinecone_vector_id, c.chunk_text, c.chunk_gcs_uri, NULL::bigint AS chunk_gcs_generation,
                   c.metadata, c.document_id, c.chunk_index, c.total_chunks, c.pinecone_index,
                   d.name AS join_doc_name, d.source_path AS join_source_path
            FROM chunks_v2 c
            INNER JOIN documents_v2 d ON d.id = c.document_id
            WHERE c.pinecone_vector_id = ANY(%s){index_clause_c}
            """
    out: Dict[str, Dict[str, Any]] = {}
    with conn.cursor(cursor_factory=cursor_factory) as cur:
        try:
            cur.execute(sql_with_gcs, params)
        except psycopg2.errors.UndefinedColumn:
            conn.rollback()
            cur.execute(sql_basic, params)
        for r in cur.fetchall():
            out[r["pinecone_vector_id"]] = dict(r)
    try:
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            cur.execute(sql_v2_with_gcs, params)
            for r in cur.fetchall():
                out[r["pinecone_vector_id"]] = dict(r)
    except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn):
        conn.rollback()
    return out


def embed_query(text: str, model: str) -> List[float]:
    from zmr_brain.clients import get_voyage_client

    client = get_voyage_client()
    return client.embed([text], model=model, input_type="query").embeddings[0]


def query_pinecone(
    index_name: str,
    query_vector: List[float],
    *,
    top_k: int = 8,
    metadata_filter: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from zmr_brain.clients import get_pinecone_index

    idx = get_pinecone_index(index_name)
    kwargs: Dict[str, Any] = dict(
        vector=query_vector,
        top_k=int(top_k),
        include_metadata=True,
    )
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    if namespace:
        kwargs["namespace"] = namespace
    res = idx.query(**kwargs)
    matches = getattr(res, "matches", None) or res.get("matches", [])
    normalized: List[Dict[str, Any]] = []
    for m in matches:
        mid = m.id if hasattr(m, "id") else m.get("id")
        score = m.score if hasattr(m, "score") else m.get("score")
        md = m.metadata if hasattr(m, "metadata") else m.get("metadata") or {}
        normalized.append(
            {"id": mid, "score": score, "metadata": dict(md) if md else {}}
        )
    return normalized


def query_pinecone_multi_namespace(
    index_name: str,
    query_vector: List[float],
    namespaces: List[str],
    *,
    top_k: int = 8,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Query multiple namespaces in **one** index; merge by score (descending).

    Prefer :func:`query_pinecone_multi_index` when using separate indexes per tier.
    """
    if not namespaces:
        return []
    if len(namespaces) == 1:
        ns = namespaces[0]
        hits = query_pinecone(
            index_name, query_vector,
            top_k=top_k, metadata_filter=metadata_filter, namespace=ns,
        )
        for h in hits:
            h["namespace"] = ns
        return hits

    def _one(ns: str) -> List[Dict[str, Any]]:
        hits = query_pinecone(
            index_name, query_vector,
            top_k=top_k, metadata_filter=metadata_filter, namespace=ns,
        )
        for h in hits:
            h["namespace"] = ns
        return hits

    all_matches: List[Dict[str, Any]] = []
    workers = min(8, len(namespaces))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, ns) for ns in namespaces]
        for fut in as_completed(futures):
            all_matches.extend(fut.result())
    all_matches.sort(key=lambda m: -(m.get("score") or 0.0))
    return all_matches[:top_k]


def query_pinecone_multi_index(
    index_names: List[str],
    query_vector: List[float],
    *,
    top_k: int = 8,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Query multiple Pinecone **indexes** (default namespace each); merge by score."""
    if not index_names:
        return []
    if len(index_names) == 1:
        name = index_names[0]
        hits = query_pinecone(
            name, query_vector,
            top_k=top_k, metadata_filter=metadata_filter, namespace=None,
        )
        for h in hits:
            h["pinecone_index"] = name
        return hits

    def _one(name: str) -> List[Dict[str, Any]]:
        hits = query_pinecone(
            name, query_vector,
            top_k=top_k, metadata_filter=metadata_filter, namespace=None,
        )
        for h in hits:
            h["pinecone_index"] = name
        return hits

    all_matches: List[Dict[str, Any]] = []
    workers = min(8, len(index_names))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, name): name for name in index_names}
        for fut in as_completed(futs):
            all_matches.extend(fut.result())
    all_matches.sort(key=lambda m: -(m.get("score") or 0.0))
    return all_matches[:top_k]


@dataclass
class RetrievedChunk:
    rank: int
    score: Optional[float]
    vector_id: str
    doc_name: Optional[str]
    source_path: Optional[str]
    sheet_name: Optional[str]
    chunk_index: Optional[int]
    total_chunks: Optional[int]
    text: Optional[str]
    gcs_uri: Optional[str]
    pinecone_metadata: Dict[str, Any]
    rrf_score: Optional[float] = None
    semantic_score: Optional[float] = None
    pinecone_rerank_score: Optional[float] = None


def reciprocal_rank_fusion(
    ranked_id_lists: List[List[str]],
    *,
    k: int = DEFAULT_RRF_K,
) -> List[Tuple[str, float]]:
    """
    Merge ordered hit lists with Reciprocal Rank Fusion (RRF).
    score(d) = sum_i 1 / (k + rank_i(d)); missing from a list → no term.
    """
    scores: Dict[str, float] = {}
    for ids in ranked_id_lists:
        for rank, vid in enumerate(ids, start=1):
            if not vid:
                continue
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def query_lexical_ranked_ids(
    conn: Any,
    cursor_factory: Any,
    query: str,
    pinecone_index: str,
    limit: int,
) -> List[str]:
    """
    Lexical retrieval via PostgreSQL FTS on ``chunks.chunk_tsv`` (not literal BM25).
    Requires migration ``007_chunk_lexical_fts`` and populated ``chunk_lexical`` / ``chunk_text``.
    """
    q = (query or "").strip()
    if not q or limit <= 0:
        return []

    sql_web = """
    WITH t AS (SELECT websearch_to_tsquery('english', %s) AS tsq)
    SELECT pinecone_vector_id FROM (
        SELECT c.pinecone_vector_id, ts_rank_cd(c.chunk_tsv, t.tsq) AS rk
        FROM chunks c, t
        WHERE c.pinecone_index = %s
          AND c.chunk_tsv @@ t.tsq
        UNION ALL
        SELECT c.pinecone_vector_id, ts_rank_cd(c.chunk_tsv, t.tsq) AS rk
        FROM chunks_v2 c, t
        WHERE c.pinecone_index = %s
          AND c.chunk_tsv @@ t.tsq
    ) u
    ORDER BY rk DESC
    LIMIT %s
    """
    sql_plain = """
    WITH t AS (SELECT plainto_tsquery('english', %s) AS tsq)
    SELECT pinecone_vector_id FROM (
        SELECT c.pinecone_vector_id, ts_rank_cd(c.chunk_tsv, t.tsq) AS rk
        FROM chunks c, t
        WHERE c.pinecone_index = %s
          AND c.chunk_tsv @@ t.tsq
        UNION ALL
        SELECT c.pinecone_vector_id, ts_rank_cd(c.chunk_tsv, t.tsq) AS rk
        FROM chunks_v2 c, t
        WHERE c.pinecone_index = %s
          AND c.chunk_tsv @@ t.tsq
    ) u
    ORDER BY rk DESC
    LIMIT %s
    """
    sql_web_legacy = """
    WITH t AS (SELECT websearch_to_tsquery('english', %s) AS tsq)
    SELECT c.pinecone_vector_id
    FROM chunks c, t
    WHERE c.pinecone_index = %s
      AND c.chunk_tsv @@ t.tsq
    ORDER BY ts_rank_cd(c.chunk_tsv, t.tsq) DESC
    LIMIT %s
    """
    sql_plain_legacy = """
    WITH t AS (SELECT plainto_tsquery('english', %s) AS tsq)
    SELECT c.pinecone_vector_id
    FROM chunks c, t
    WHERE c.pinecone_index = %s
      AND c.chunk_tsv @@ t.tsq
    ORDER BY ts_rank_cd(c.chunk_tsv, t.tsq) DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=cursor_factory) as cur:
        try:
            cur.execute(sql_web, (q, pinecone_index, pinecone_index, limit))
        except psycopg2.errors.UndefinedTable:
            conn.rollback()
            try:
                cur.execute(sql_web_legacy, (q, pinecone_index, limit))
            except (psycopg2.errors.UndefinedColumn, psycopg2.errors.UndefinedTable):
                conn.rollback()
                return []
            except Exception:
                conn.rollback()
                try:
                    cur.execute(sql_plain_legacy, (q, pinecone_index, limit))
                except Exception:
                    conn.rollback()
                    return []
        except (psycopg2.errors.UndefinedColumn, psycopg2.errors.UndefinedTable):
            conn.rollback()
            return []
        except Exception:
            conn.rollback()
            try:
                cur.execute(sql_plain, (q, pinecone_index, pinecone_index, limit))
            except psycopg2.errors.UndefinedTable:
                conn.rollback()
                try:
                    cur.execute(sql_plain_legacy, (q, pinecone_index, limit))
                except Exception:
                    conn.rollback()
                    return []
            except Exception:
                conn.rollback()
                return []
        rows = cur.fetchall()
    return [r["pinecone_vector_id"] for r in rows]


def query_lexical_ranked_ids_multi(
    conn: Any,
    cursor_factory: Any,
    query: str,
    pinecone_indexes: List[str],
    limit: int,
) -> List[str]:
    """FTS over chunks in any of the given ``pinecone_index`` values."""
    if len(pinecone_indexes) == 1:
        return query_lexical_ranked_ids(
            conn, cursor_factory, query, pinecone_indexes[0], limit
        )
    q = (query or "").strip()
    if not q or limit <= 0:
        return []

    sql_web = """
    WITH t AS (SELECT websearch_to_tsquery('english', %s) AS tsq)
    SELECT pinecone_vector_id FROM (
        SELECT c.pinecone_vector_id, ts_rank_cd(c.chunk_tsv, t.tsq) AS rk
        FROM chunks c, t
        WHERE c.pinecone_index = ANY(%s)
          AND c.chunk_tsv @@ t.tsq
        UNION ALL
        SELECT c.pinecone_vector_id, ts_rank_cd(c.chunk_tsv, t.tsq) AS rk
        FROM chunks_v2 c, t
        WHERE c.pinecone_index = ANY(%s)
          AND c.chunk_tsv @@ t.tsq
    ) u
    ORDER BY rk DESC
    LIMIT %s
    """
    sql_plain = """
    WITH t AS (SELECT plainto_tsquery('english', %s) AS tsq)
    SELECT pinecone_vector_id FROM (
        SELECT c.pinecone_vector_id, ts_rank_cd(c.chunk_tsv, t.tsq) AS rk
        FROM chunks c, t
        WHERE c.pinecone_index = ANY(%s)
          AND c.chunk_tsv @@ t.tsq
        UNION ALL
        SELECT c.pinecone_vector_id, ts_rank_cd(c.chunk_tsv, t.tsq) AS rk
        FROM chunks_v2 c, t
        WHERE c.pinecone_index = ANY(%s)
          AND c.chunk_tsv @@ t.tsq
    ) u
    ORDER BY rk DESC
    LIMIT %s
    """
    sql_web_legacy = """
    WITH t AS (SELECT websearch_to_tsquery('english', %s) AS tsq)
    SELECT c.pinecone_vector_id
    FROM chunks c, t
    WHERE c.pinecone_index = ANY(%s)
      AND c.chunk_tsv @@ t.tsq
    ORDER BY ts_rank_cd(c.chunk_tsv, t.tsq) DESC
    LIMIT %s
    """
    sql_plain_legacy = """
    WITH t AS (SELECT plainto_tsquery('english', %s) AS tsq)
    SELECT c.pinecone_vector_id
    FROM chunks c, t
    WHERE c.pinecone_index = ANY(%s)
      AND c.chunk_tsv @@ t.tsq
    ORDER BY ts_rank_cd(c.chunk_tsv, t.tsq) DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=cursor_factory) as cur:
        try:
            cur.execute(sql_web, (q, pinecone_indexes, pinecone_indexes, limit))
        except psycopg2.errors.UndefinedTable:
            conn.rollback()
            try:
                cur.execute(sql_web_legacy, (q, pinecone_indexes, limit))
            except (psycopg2.errors.UndefinedColumn, psycopg2.errors.UndefinedTable):
                conn.rollback()
                return []
            except Exception:
                conn.rollback()
                try:
                    cur.execute(sql_plain_legacy, (q, pinecone_indexes, limit))
                except Exception:
                    conn.rollback()
                    return []
        except (psycopg2.errors.UndefinedColumn, psycopg2.errors.UndefinedTable):
            conn.rollback()
            return []
        except Exception:
            conn.rollback()
            try:
                cur.execute(sql_plain, (q, pinecone_indexes, pinecone_indexes, limit))
            except psycopg2.errors.UndefinedTable:
                conn.rollback()
                try:
                    cur.execute(sql_plain_legacy, (q, pinecone_indexes, limit))
                except Exception:
                    conn.rollback()
                    return []
            except Exception:
                conn.rollback()
                return []
        rows = cur.fetchall()
    return [r["pinecone_vector_id"] for r in rows]


def _legacy_namespace_mode(namespaces: List[str]) -> bool:
    """True if ``namespaces`` are logical tier names inside one index (pre–3-index model)."""
    if not namespaces:
        return False
    tier_vals = set(TIER_NAMESPACE.values())
    return set(namespaces).issubset(tier_vals)


def _effective_metadata(
    row: Dict[str, Any], pinecone_md: Dict[str, Any]
) -> Dict[str, Any]:
    out = dict(pinecone_md) if pinecone_md else {}
    if not out.get("doc_name") and row.get("join_doc_name"):
        out["doc_name"] = row["join_doc_name"]
    if not out.get("source_path") and row.get("join_source_path"):
        out["source_path"] = row["join_source_path"]
    return out


def retrieve_for_query(
    query: str,
    pinecone_index: str,
    *,
    top_k: int = 8,
    embed_model: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    namespaces: Optional[List[str]] = None,
    include_empty_text: bool = False,
    hybrid_rrf: bool = True,
    rrf_k: int = DEFAULT_RRF_K,
    candidate_pool: Optional[int] = None,
    pinecone_rerank: bool = True,
    rerank_pool: Optional[int] = None,
    pinecone_rerank_model: Optional[str] = None,
    lexical_mode: str = "bm25",
) -> List[RetrievedChunk]:
    """
    Hybrid (default): dense vectors (Pinecone) + lexical ranks (Postgres FTS on ``chunk_tsv``),
    fused with RRF, then optional **Pinecone Inference rerank** (``pc.inference.rerank``).

    Wall-clock: multi-index / multi-namespace Pinecone calls run concurrently; with hybrid RRF,
    the Pinecone leg overlaps the lexical leg (Postgres). For ``lexical_mode='both'``, BM25 and
    FTS queries run concurrently. Set ``ZMR_RETRIEVAL_PARALLEL=0`` to force the previous
    sequential hybrid timing (debug only).

    Pure semantic: set ``hybrid_rrf=False``. Disable cross-encoder reorder with ``pinecone_rerank=False``.

    Lexical leg (keyword): **BM25** (``rank_bm25`` over ``chunk_lexical``/``chunk_text``) by default;
    set ``lexical_mode='fts'`` for Postgres FTS only, or ``'both'`` to RRF semantic + BM25 + FTS.

    Postgres FTS is **not** BM25 (it uses ``ts_rank_cd``). BM25 is computed in Python over a corpus
    capped by ``ZMR_BM25_MAX_CORPUS`` (see ``zmr_brain/bm25_lexical.py``).

    By default, empty-text chunks are dropped (``include_empty_text=False``) so the LLM and UI do not
    waste slots on metadata-only/empty files. Set ``include_empty_text=True`` only for debugging.

    Lexical rows need migration 007 + ingest text in ``chunk_lexical`` (or non-empty ``chunk_text``).

    ``namespaces``: either (1) **Pinecone index names** for multi-index RBAC (e.g. ``zmr-brain-full``),
    or (2) legacy logical tier names ``full`` / ``executive_only`` / ``restricted_accounting`` for
    the single-index ``zmr-brain-dev`` setup.
    """
    pool = (
        int(candidate_pool)
        if candidate_pool is not None
        else max(DEFAULT_HYBRID_CANDIDATE_POOL, top_k * 3)
    )
    pool = max(pool, top_k)
    rpool = (
        int(rerank_pool)
        if rerank_pool is not None
        else max(DEFAULT_RERANK_POOL, top_k * 3)
    )
    rpool = max(rpool, top_k)

    if query_vector is not None:
        qvec = query_vector
    else:
        if embed_model:
            model = embed_model.strip()
        else:
            from zmr_brain.embed_models import select_voyage_embed_model_for_query
            model, _ = select_voyage_embed_model_for_query(query)
        qvec = embed_query(query, model)

    if namespaces and _legacy_namespace_mode(namespaces):
        idx_list = [pinecone_index]
    else:
        idx_list = (
            list(namespaces)
            if namespaces
            else ([pinecone_index] if pinecone_index else [])
        )

    multi_index_semantic = bool(namespaces and not _legacy_namespace_mode(namespaces))

    def _pinecone_matches() -> List[Dict[str, Any]]:
        if namespaces and _legacy_namespace_mode(namespaces):
            return query_pinecone_multi_namespace(
                pinecone_index, qvec, namespaces,
                top_k=pool, metadata_filter=metadata_filter,
            )
        if namespaces:
            return query_pinecone_multi_index(
                namespaces, qvec,
                top_k=pool, metadata_filter=metadata_filter,
            )
        return query_pinecone(
            pinecone_index, qvec, top_k=pool, metadata_filter=metadata_filter,
        )

    if not hybrid_rrf:
        matches = _pinecone_matches()
        out = _retrieve_semantic_ordered(
            matches,
            pinecone_index,
            include_empty_text=include_empty_text,
            no_index_filter=multi_index_semantic,
        )
        if pinecone_rerank and out:
            from zmr_brain.pinecone_rerank import rerank_chunks_pinecone

            try:
                return rerank_chunks_pinecone(
                    query,
                    out,
                    top_n=top_k,
                    model=pinecone_rerank_model,
                )
            except Exception:
                return out[:top_k]
        return out[:top_k]

    mode = (lexical_mode or os.getenv("ZMR_LEXICAL_MODE", "bm25")).strip().lower()
    if mode not in ("bm25", "fts", "both"):
        mode = "bm25"

    def _lexical_rrf_leg() -> Tuple[str, List[str], Optional[List[str]]]:
        """
        Return (kind, primary_ids, secondary_ids).
        kind ``both`` → BM25 + FTS lists; ``fts`` / ``bm25`` → single list, secondary None.
        """
        from zmr_brain.bm25_lexical import (
            query_bm25_ranked_ids,
            query_bm25_ranked_ids_multi,
        )

        lex_index_key = (
            idx_list[0]
            if len(idx_list) == 1
            else pinecone_index
        )

        if mode == "both":

            def _bm25_ids() -> List[str]:
                c, cf = pg_connect()
                try:
                    if len(idx_list) > 1:
                        return query_bm25_ranked_ids_multi(
                            c, cf, query, idx_list, pool
                        )
                    return query_bm25_ranked_ids(
                        c, cf, query, lex_index_key, pool
                    )
                finally:
                    pg_release(c)

            def _fts_ids() -> List[str]:
                c, cf = pg_connect()
                try:
                    if len(idx_list) > 1:
                        return query_lexical_ranked_ids_multi(
                            c, cf, query, idx_list, pool
                        )
                    return query_lexical_ranked_ids(
                        c, cf, query, lex_index_key, pool
                    )
                finally:
                    pg_release(c)

            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_b = ex.submit(_bm25_ids)
                fut_f = ex.submit(_fts_ids)
                bm25_ids = fut_b.result()
                fts_ids = fut_f.result()
            return ("both", bm25_ids, fts_ids)

        conn, cursor_factory = pg_connect()
        try:
            if mode == "fts":
                if len(idx_list) > 1:
                    lex_ids = query_lexical_ranked_ids_multi(
                        conn, cursor_factory, query, idx_list, pool
                    )
                else:
                    lex_ids = query_lexical_ranked_ids(
                        conn, cursor_factory, query, lex_index_key, pool
                    )
                return ("fts", lex_ids, None)
            if len(idx_list) > 1:
                lex_ids = query_bm25_ranked_ids_multi(
                    conn, cursor_factory, query, idx_list, pool
                )
                if not lex_ids:
                    lex_ids = query_lexical_ranked_ids_multi(
                        conn, cursor_factory, query, idx_list, pool
                    )
            else:
                lex_ids = query_bm25_ranked_ids(
                    conn, cursor_factory, query, lex_index_key, pool
                )
                if not lex_ids:
                    lex_ids = query_lexical_ranked_ids(
                        conn, cursor_factory, query, lex_index_key, pool
                    )
            return ("bm25", lex_ids, None)
        finally:
            pg_release(conn)

    use_parallel = os.getenv("ZMR_RETRIEVAL_PARALLEL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if use_parallel:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_p = ex.submit(_pinecone_matches)
            fut_l = ex.submit(_lexical_rrf_leg)
            matches = fut_p.result()
            leg = fut_l.result()
    else:
        matches = _pinecone_matches()
        leg = _lexical_rrf_leg()

    sem_ids = [m["id"] for m in matches if m.get("id")]
    pinecone_md = {m["id"]: (m.get("metadata") or {}) for m in matches if m.get("id")}
    sem_score = {m["id"]: m.get("score") for m in matches if m.get("id")}

    kind, primary_ids, secondary_ids = leg
    if kind == "both" and secondary_ids is not None:
        fused = reciprocal_rank_fusion(
            [sem_ids, primary_ids, secondary_ids], k=rrf_k
        )[:rpool]
    else:
        fused = reciprocal_rank_fusion([sem_ids, primary_ids], k=rrf_k)[:rpool]

    if not fused:
        return []

    conn, cursor_factory = pg_connect()
    try:
        chunk_map = fetch_chunks_by_vector_ids(
            conn, cursor_factory, [vid for vid, _ in fused],
            None if multi_index_semantic else pinecone_index,
        )
    finally:
        pg_release(conn)

    rows_ordered = [chunk_map.get(vid) or {} for vid, _ in fused]
    texts = _body_texts_for_rows(rows_ordered)

    out: List[RetrievedChunk] = []
    for rank, ((vid, rrf_s), row, text) in enumerate(
        zip(fused, rows_ordered, texts), start=1
    ):
        md = pinecone_md.get(vid, {})
        em = _effective_metadata(row, md)
        if not include_empty_text and not (text and str(text).strip()):
            continue
        out.append(
            RetrievedChunk(
                rank=rank,
                score=rrf_s,
                vector_id=vid,
                doc_name=em.get("doc_name"),
                source_path=em.get("source_path"),
                sheet_name=em.get("sheet_name"),
                chunk_index=row.get("chunk_index"),
                total_chunks=row.get("total_chunks"),
                text=text,
                gcs_uri=row.get("chunk_gcs_uri"),
                pinecone_metadata=em,
                rrf_score=rrf_s,
                semantic_score=sem_score.get(vid),
            )
        )

    if pinecone_rerank and out:
        from zmr_brain.pinecone_rerank import rerank_chunks_pinecone

        try:
            return rerank_chunks_pinecone(
                query,
                out,
                top_n=top_k,
                model=pinecone_rerank_model,
            )
        except Exception:
            return out[:top_k]
    return out[:top_k]


def _retrieve_semantic_ordered(
    matches: List[Dict[str, Any]],
    pinecone_index: str,
    *,
    include_empty_text: bool = True,
    no_index_filter: bool = False,
) -> List[RetrievedChunk]:
    vector_ids = [m["id"] for m in matches if m.get("id")]
    conn, cursor_factory = pg_connect()
    try:
        chunk_map = fetch_chunks_by_vector_ids(
            conn, cursor_factory, vector_ids,
            None if no_index_filter else pinecone_index,
        )
    finally:
        pg_release(conn)

    rows_ordered = [chunk_map.get(vid) or {} for vid in vector_ids]
    texts = _body_texts_for_rows(rows_ordered)

    out: List[RetrievedChunk] = []
    for rank, (m, row, text) in enumerate(zip(matches, rows_ordered, texts), start=1):
        vid = m["id"]
        md = m.get("metadata") or {}
        em = _effective_metadata(row, md)
        if not include_empty_text and not (text and str(text).strip()):
            continue
        out.append(
            RetrievedChunk(
                rank=rank,
                score=m.get("score"),
                vector_id=vid,
                doc_name=em.get("doc_name"),
                source_path=em.get("source_path"),
                sheet_name=em.get("sheet_name"),
                chunk_index=row.get("chunk_index"),
                total_chunks=row.get("total_chunks"),
                text=text,
                gcs_uri=row.get("chunk_gcs_uri"),
                pinecone_metadata=em,
            )
        )
    return out


def _chunk_body_text(row: Dict[str, Any]) -> Optional[str]:
    """Prefer inline chunk_text; else load from chunk_gcs_uri (``gs://`` or ``local:`` repo-relative path)."""
    raw = row.get("chunk_text")
    if raw is not None and str(raw).strip():
        return str(raw)
    uri = row.get("chunk_gcs_uri")
    if not uri or not str(uri).strip():
        return None
    from zmr_brain.chunk_store_local import load_chunk_body_from_uri

    try:
        text = load_chunk_body_from_uri(str(uri).strip())
    except Exception:
        return None
    return text


def _body_texts_for_rows(rows: List[Dict[str, Any]]) -> List[Optional[str]]:
    """
    Resolve ``chunk_text`` or ``chunk_gcs_uri`` for each row. GCS/local reads used to run
    **sequentially** and dominated retrieve time for v2 ingestion; load in parallel when
    there are multiple bodies to fetch.

    Disable with ``ZMR_CHUNK_LOAD_PARALLEL=0``. Cap workers with ``ZMR_CHUNK_LOAD_PARALLEL_MAX``
    (default 12).
    """
    if not rows:
        return []
    if len(rows) == 1:
        return [_chunk_body_text(rows[0])]

    parallel = os.getenv("ZMR_CHUNK_LOAD_PARALLEL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if not parallel:
        return [_chunk_body_text(r) for r in rows]

    try:
        cap = int(os.getenv("ZMR_CHUNK_LOAD_PARALLEL_MAX", "12"))
    except ValueError:
        cap = 12
    max_workers = max(1, min(len(rows), cap))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_chunk_body_text, rows))


def chunk_has_usable_text(c: RetrievedChunk) -> bool:
    """True if chunk body text is non-empty (inline or loaded from GCS/local)."""
    return bool((c.text or "").strip())


def chunks_with_body_text_for_llm(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Keep **every** retrieved chunk that has non-empty body text (same order as retrieval output).
    Renumber ``rank`` to 1..n for the prompt. Metadata-only hits (no text) are dropped so the LLM
    receives only real passages—the full usable subset of the final ``retrieve_for_query`` list.
    """
    with_text = [c for c in chunks if chunk_has_usable_text(c)]
    return [replace(c, rank=i) for i, c in enumerate(with_text, start=1)]


_MAX_CHUNK_CHARS_FOR_LLM = int(os.getenv("ZMR_MAX_CHUNK_CHARS_LLM", "1500"))


def chunks_to_context_blocks(chunks: List[RetrievedChunk]) -> str:
    """Format chunks for the LLM. Prefer :func:`chunks_with_body_text_for_llm` first.

    Each chunk body is truncated to ``ZMR_MAX_CHUNK_CHARS_LLM`` characters
    (default 1500, ~375 tokens) to keep prompt size manageable.
    """
    cap = _MAX_CHUNK_CHARS_FOR_LLM
    parts: List[str] = []
    for c in chunks:
        head = f"[{c.rank}] doc={c.doc_name or '?'} path={c.source_path or '?'}"
        body = (c.text or "").strip() or "(no chunk text — metadata-only ingestion)"
        if len(body) > cap:
            body = body[:cap] + " …"
        parts.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(parts)

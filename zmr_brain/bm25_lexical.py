"""
In-memory BM25 (Okapi) over chunk text loaded from Postgres.

PostgreSQL FTS uses ``ts_rank_cd``, which is **not** BM25. True BM25 needs either an
external search engine (Elasticsearch, etc.), an extension (e.g. ParadeDB), or a library
like ``rank_bm25`` over a tokenized corpus — we do the latter for chunks in one or more
Pinecone indexes (see ``zmr_brain.constants.PINECONE_INDEX_BY_TIER``).

Requires non-empty ``chunk_lexical`` or ``chunk_text`` per row (see migration 007 + ingest).
"""

from __future__ import annotations

import os
import re
import time as _time
import threading as _threading
from typing import Any, Dict, List, Tuple

import psycopg2.errors

from zmr_brain.constants import DEFAULT_BM25_MAX_CORPUS

_BM25_CACHE_TTL = float(os.getenv("ZMR_BM25_CACHE_TTL", "300"))

_bm25_cache: Dict[str, Tuple[Any, List[str], float]] = {}
_bm25_cache_lock = _threading.Lock()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _load_corpus(
    conn: Any,
    cursor_factory: Any,
    pinecone_index: str,
) -> Tuple[Any, List[str]]:
    """Load corpus from Postgres, tokenize, and build BM25Okapi."""
    from rank_bm25 import BM25Okapi

    max_corpus = int(os.getenv("ZMR_BM25_MAX_CORPUS", str(DEFAULT_BM25_MAX_CORPUS)))

    sql = """
    SELECT u.vid, u.body FROM (
        SELECT c.pinecone_vector_id AS vid,
               coalesce(c.chunk_lexical, c.chunk_text, '') AS body
        FROM chunks c
        WHERE c.pinecone_index = %s
          AND length(trim(coalesce(c.chunk_lexical, c.chunk_text, ''))) > 0
        UNION ALL
        SELECT c.pinecone_vector_id AS vid,
               coalesce(c.chunk_lexical, c.chunk_text, '') AS body
        FROM chunks_v2 c
        WHERE c.pinecone_index = %s
          AND length(trim(coalesce(c.chunk_lexical, c.chunk_text, ''))) > 0
    ) u
    LIMIT %s
    """
    with conn.cursor(cursor_factory=cursor_factory) as cur:
        try:
            cur.execute(sql, (pinecone_index, pinecone_index, max_corpus))
        except psycopg2.errors.UndefinedTable:
            conn.rollback()
            try:
                cur.execute(
                    """
                    SELECT c.pinecone_vector_id AS vid,
                           coalesce(c.chunk_lexical, c.chunk_text, '') AS body
                    FROM chunks c
                    WHERE c.pinecone_index = %s
                      AND length(trim(coalesce(c.chunk_lexical, c.chunk_text, ''))) > 0
                    LIMIT %s
                    """,
                    (pinecone_index, max_corpus),
                )
            except (
                psycopg2.errors.UndefinedColumn,
                psycopg2.errors.UndefinedTable,
                Exception,
            ):
                conn.rollback()
                return None, []
        except (
            psycopg2.errors.UndefinedColumn,
            psycopg2.errors.UndefinedTable,
            Exception,
        ):
            conn.rollback()
            return None, []
        rows = cur.fetchall()

    if not rows:
        return None, []

    pairs: List[Tuple[str, List[str]]] = []
    for r in rows:
        vid = r["vid"]
        toks = _tokenize(str(r["body"]))
        if toks:
            pairs.append((vid, toks))

    if not pairs:
        return None, []

    corpus_tokens = [p[1] for p in pairs]
    corpus_ids = [p[0] for p in pairs]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, corpus_ids


def _cache_key_for_indexes(pinecone_indexes: List[str]) -> str:
    return ",".join(sorted(pinecone_indexes)) + "|chunks_v2"


def query_bm25_ranked_ids_multi(
    conn: Any,
    cursor_factory: Any,
    query: str,
    pinecone_indexes: List[str],
    limit: int,
) -> List[str]:
    """BM25 over chunks whose ``pinecone_index`` is in ``pinecone_indexes``."""
    if len(pinecone_indexes) == 1:
        return query_bm25_ranked_ids(
            conn, cursor_factory, query, pinecone_indexes[0], limit
        )
    key = _cache_key_for_indexes(pinecone_indexes)
    now = _time.monotonic()
    cached = _bm25_cache.get(key)
    if cached and (now - cached[2]) < _BM25_CACHE_TTL:
        bm25, corpus_ids = cached[0], cached[1]
    else:
        bm25, corpus_ids = _load_corpus_multi(conn, cursor_factory, pinecone_indexes)
        if bm25 is None:
            return []
        with _bm25_cache_lock:
            _bm25_cache[key] = (bm25, corpus_ids, now)

    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
    except ImportError:
        return []

    q = (query or "").strip()
    q_tokens = _tokenize(q)
    if not q or not q_tokens or limit <= 0:
        return []

    scores = bm25.get_scores(q_tokens)
    order = sorted(range(len(corpus_ids)), key=lambda i: -scores[i])
    return [corpus_ids[i] for i in order[:limit]]


def _load_corpus_multi(
    conn: Any,
    cursor_factory: Any,
    pinecone_indexes: List[str],
) -> Tuple[Any, List[str]]:
    from rank_bm25 import BM25Okapi

    max_corpus = int(os.getenv("ZMR_BM25_MAX_CORPUS", str(DEFAULT_BM25_MAX_CORPUS)))

    sql = """
    SELECT u.vid, u.body FROM (
        SELECT c.pinecone_vector_id AS vid,
               coalesce(c.chunk_lexical, c.chunk_text, '') AS body
        FROM chunks c
        WHERE c.pinecone_index = ANY(%s)
          AND length(trim(coalesce(c.chunk_lexical, c.chunk_text, ''))) > 0
        UNION ALL
        SELECT c.pinecone_vector_id AS vid,
               coalesce(c.chunk_lexical, c.chunk_text, '') AS body
        FROM chunks_v2 c
        WHERE c.pinecone_index = ANY(%s)
          AND length(trim(coalesce(c.chunk_lexical, c.chunk_text, ''))) > 0
    ) u
    LIMIT %s
    """
    with conn.cursor(cursor_factory=cursor_factory) as cur:
        try:
            cur.execute(sql, (pinecone_indexes, pinecone_indexes, max_corpus))
        except psycopg2.errors.UndefinedTable:
            conn.rollback()
            try:
                cur.execute(
                    """
                    SELECT c.pinecone_vector_id AS vid,
                           coalesce(c.chunk_lexical, c.chunk_text, '') AS body
                    FROM chunks c
                    WHERE c.pinecone_index = ANY(%s)
                      AND length(trim(coalesce(c.chunk_lexical, c.chunk_text, ''))) > 0
                    LIMIT %s
                    """,
                    (pinecone_indexes, max_corpus),
                )
            except (
                psycopg2.errors.UndefinedColumn,
                psycopg2.errors.UndefinedTable,
                Exception,
            ):
                conn.rollback()
                return None, []
        except (
            psycopg2.errors.UndefinedColumn,
            psycopg2.errors.UndefinedTable,
            Exception,
        ):
            conn.rollback()
            return None, []
        rows = cur.fetchall()

    if not rows:
        return None, []

    pairs: List[Tuple[str, List[str]]] = []
    for r in rows:
        vid = r["vid"]
        toks = _tokenize(str(r["body"]))
        if toks:
            pairs.append((vid, toks))

    if not pairs:
        return None, []

    corpus_tokens = [p[1] for p in pairs]
    corpus_ids = [p[0] for p in pairs]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, corpus_ids


def query_bm25_ranked_ids(
    conn: Any,
    cursor_factory: Any,
    query: str,
    pinecone_index: str,
    limit: int,
) -> List[str]:
    """
    Rank ``pinecone_vector_id`` by BM25Okapi over all chunks in ``pinecone_index``
    (up to ``ZMR_BM25_MAX_CORPUS`` rows). Returns top ``limit`` ids by score.

    The BM25 object and corpus IDs are **cached** per index with a configurable
    TTL (env ``ZMR_BM25_CACHE_TTL``, default 300 s) to avoid rebuilding on
    every query.
    """
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
    except ImportError:
        return []

    q = (query or "").strip()
    q_tokens = _tokenize(q)
    if not q or not q_tokens or limit <= 0:
        return []

    cache_key = f"{pinecone_index}|chunks_v2"
    now = _time.monotonic()
    cached = _bm25_cache.get(cache_key)
    if cached and (now - cached[2]) < _BM25_CACHE_TTL:
        bm25, corpus_ids = cached[0], cached[1]
    else:
        bm25, corpus_ids = _load_corpus(conn, cursor_factory, pinecone_index)
        if bm25 is None:
            return []
        with _bm25_cache_lock:
            _bm25_cache[cache_key] = (bm25, corpus_ids, now)

    scores = bm25.get_scores(q_tokens)
    order = sorted(range(len(corpus_ids)), key=lambda i: -scores[i])
    return [corpus_ids[i] for i in order[:limit]]

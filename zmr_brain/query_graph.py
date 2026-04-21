"""
LangGraph pipeline: route → (direct reply | reformulate → retrieve) → optional Claude synthesis.

``route`` decides: assistant intro, refuse general trivia, or document retrieval.
``reformulate`` rewrites the question for embedding + lexical search; the original ``query``
is still used when synthesizing the answer with Claude.

3-tier RBAC: access enforced via Pinecone metadata filter (access_tier), not separate indexes.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from zmr_brain.answer import answer_with_claude
from zmr_brain.constants import (
    DEFAULT_RERANK_POOL,
    DEFAULT_RRF_K,
    PINECONE_INDEX,
    namespaces_for_email,
    pinecone_access_filter,
)
from zmr_brain.meta_queries import chatbot_meta_reply
from zmr_brain.query_reformulate import reformulate_query_for_retrieval
from zmr_brain.query_routing import OUT_OF_SCOPE_REPLY, classify_query
from zmr_brain.embed_models import select_voyage_embed_model_for_query
from zmr_brain.retrieval import RetrievedChunk, embed_query, retrieve_for_query
from zmr_brain.tracing import init_langsmith_tracing


class QueryGraphState(TypedDict, total=False):
    query: str
    user_role: str
    user_email: str
    top_k: int
    embed_model: Optional[str]
    metadata_filter: Optional[Dict[str, Any]]
    skip_query_reformulation: bool
    generate_answer: bool
    hybrid_rrf: bool
    rrf_k: int
    candidate_pool: Optional[int]
    pinecone_rerank: bool
    rerank_pool: int
    pinecone_rerank_model: Optional[str]
    lexical_mode: str
    query_kind: str
    retrieval_query: str
    pinecone_index: str
    chunks: List[RetrievedChunk]
    answer: Optional[str]
    error: Optional[str]
    graph_trace: List[str]
    meta_intro: bool
    refuse_out_of_scope: bool


def _trace(state: QueryGraphState, step: str) -> List[str]:
    return list(state.get("graph_trace") or []) + [step]


def node_route(state: QueryGraphState) -> Dict[str, Any]:
    q = (state.get("query") or "").strip()
    kind = classify_query(q)
    return {
        "query_kind": kind,
        "graph_trace": _trace(state, f"route_{kind}"),
    }


def node_direct_reply(state: QueryGraphState) -> Dict[str, Any]:
    """Intro or out-of-scope reply — no retrieval, no Claude on passages."""
    kind = state.get("query_kind")
    if kind == "intro":
        return {
            "pinecone_index": PINECONE_INDEX,
            "chunks": [],
            "answer": chatbot_meta_reply(state.get("query") or ""),
            "error": None,
            "meta_intro": True,
            "refuse_out_of_scope": False,
            "graph_trace": _trace(state, "direct_intro"),
        }
    if kind == "refuse":
        return {
            "pinecone_index": PINECONE_INDEX,
            "chunks": [],
            "answer": OUT_OF_SCOPE_REPLY,
            "error": None,
            "meta_intro": False,
            "refuse_out_of_scope": True,
            "graph_trace": _trace(state, "direct_refuse"),
        }
    return {"graph_trace": _trace(state, "direct_empty")}


def node_reformulate(state: QueryGraphState) -> Dict[str, Any]:
    """Produce ``retrieval_query`` for Pinecone + BM25/FTS; keep ``query`` for the final answer."""
    raw = (state.get("query") or "").strip()
    if not raw:
        return {
            "retrieval_query": "",
            "graph_trace": _trace(state, "reformulate_skip_empty"),
        }
    if state.get("skip_query_reformulation"):
        return {
            "retrieval_query": raw,
            "graph_trace": _trace(state, "reformulate_skipped_fast"),
        }
    try:
        rq = reformulate_query_for_retrieval(raw)
    except Exception:
        rq = raw
    rq = (rq or raw).strip() or raw
    changed = rq.lower() != raw.lower()
    return {
        "retrieval_query": rq,
        "graph_trace": _trace(
            state, "reformulate_ok" if changed else "reformulate_unchanged"
        ),
    }


def node_retrieve(state: QueryGraphState) -> Dict[str, Any]:
    q = (state.get("retrieval_query") or state.get("query") or "").strip()
    top_k = int(state.get("top_k") or 8)
    embed_model = state.get("embed_model")
    user_email = (state.get("user_email") or "").strip().lower()

    if not q:
        return {
            "error": "query is empty",
            "chunks": [],
            "pinecone_index": PINECONE_INDEX,
            "meta_intro": False,
            "refuse_out_of_scope": False,
            "graph_trace": _trace(state, "retrieve_skip"),
        }

    # Resolve namespaces for RBAC
    user_namespaces = namespaces_for_email(user_email)
    explicit_flt = state.get("metadata_filter")

    # Select embedding model
    if embed_model:
        _model = embed_model.strip()
        _domain = "override"
    else:
        _model, _domain = select_voyage_embed_model_for_query(q)
    t0 = time.perf_counter()
    qvec = embed_query(q, _model)
    t_embed = time.perf_counter() - t0

    try:
        t1 = time.perf_counter()
        chunks = retrieve_for_query(
            q,
            PINECONE_INDEX,
            top_k=top_k,
            embed_model=embed_model,
            query_vector=qvec,
            metadata_filter=explicit_flt,
            namespaces=user_namespaces,
            include_empty_text=False,
            hybrid_rrf=bool(state.get("hybrid_rrf", True)),
            rrf_k=int(state.get("rrf_k") or DEFAULT_RRF_K),
            candidate_pool=state.get("candidate_pool"),
            pinecone_rerank=bool(state.get("pinecone_rerank", True)),
            rerank_pool=int(state.get("rerank_pool") or DEFAULT_RERANK_POOL),
            pinecone_rerank_model=state.get("pinecone_rerank_model"),
            lexical_mode=str(state.get("lexical_mode") or "bm25"),
        )
        t_retrieve = time.perf_counter() - t1
        if os.getenv("ZMR_RETRIEVE_PROFILE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            print(
                f"[ZMR_RETRIEVE_PROFILE] embed={t_embed:.3f}s "
                f"retrieve_for_query={t_retrieve:.3f}s model={_model} "
                f"chunks={len(chunks)}",
                file=sys.stderr,
                flush=True,
            )
    except Exception as e:
        return {
            "error": str(e),
            "pinecone_index": PINECONE_INDEX,
            "chunks": [],
            "meta_intro": False,
            "refuse_out_of_scope": False,
            "graph_trace": _trace(state, "retrieve_error"),
        }

    return {
        "pinecone_index": PINECONE_INDEX,
        "chunks": chunks,
        "error": None,
        "meta_intro": False,
        "refuse_out_of_scope": False,
        "embed_model": _model,
        "graph_trace": _trace(state, f"retrieve_ok(embed={_model},domain={_domain})"),
    }


def node_synthesize(state: QueryGraphState) -> Dict[str, Any]:
    if state.get("error"):
        return {"graph_trace": _trace(state, "synthesize_skip_error")}
    chunks = state.get("chunks") or []
    try:
        text = answer_with_claude(state["query"], chunks)
        return {
            "answer": text,
            "graph_trace": _trace(state, "synthesize_ok"),
        }
    except Exception as e:
        return {
            "answer": None,
            "error": str(e),
            "graph_trace": _trace(state, "synthesize_error"),
        }


def _route_after_route(state: QueryGraphState) -> str:
    k = state.get("query_kind", "document")
    if k in ("intro", "refuse"):
        return "direct_reply"
    return "reformulate"


def _route_after_retrieve(state: QueryGraphState) -> str:
    if state.get("error"):
        return "end"
    if state.get("generate_answer"):
        return "synthesize"
    return "end"


def build_query_graph() -> StateGraph:
    g = StateGraph(QueryGraphState)
    g.add_node("route", node_route)
    g.add_node("direct_reply", node_direct_reply)
    g.add_node("reformulate", node_reformulate)
    g.add_node("retrieve", node_retrieve)
    g.add_node("synthesize", node_synthesize)
    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        _route_after_route,
        {"direct_reply": "direct_reply", "reformulate": "reformulate"},
    )
    g.add_edge("direct_reply", END)
    g.add_edge("reformulate", "retrieve")
    g.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"synthesize": "synthesize", "end": END},
    )
    g.add_edge("synthesize", END)
    return g


_compiled: Any = None


def get_compiled_query_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_query_graph().compile()
    return _compiled


def run_query_graph(
    query: str,
    user_role: str,
    *,
    user_email: str = "",
    top_k: int = 8,
    embed_model: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    skip_query_reformulation: bool = False,
    generate_answer: bool = False,
    hybrid_rrf: bool = True,
    rrf_k: int = DEFAULT_RRF_K,
    candidate_pool: Optional[int] = None,
    pinecone_rerank: bool = True,
    rerank_pool: int = DEFAULT_RERANK_POOL,
    pinecone_rerank_model: Optional[str] = None,
    lexical_mode: str = "bm25",
) -> QueryGraphState:
    """Run route → direct reply or retrieve → optional answer; returns final state."""
    init_langsmith_tracing()
    initial: QueryGraphState = {
        "query": query,
        "user_role": user_role,
        "user_email": user_email,
        "top_k": top_k,
        "embed_model": embed_model,
        "metadata_filter": metadata_filter,
        "skip_query_reformulation": skip_query_reformulation,
        "generate_answer": generate_answer,
        "hybrid_rrf": hybrid_rrf,
        "rrf_k": rrf_k,
        "candidate_pool": candidate_pool,
        "pinecone_rerank": pinecone_rerank,
        "rerank_pool": rerank_pool,
        "pinecone_rerank_model": pinecone_rerank_model,
        "lexical_mode": lexical_mode,
        "graph_trace": [],
    }
    return get_compiled_query_graph().invoke(initial)


def stream_query_graph(
    query: str,
    user_role: str,
    *,
    user_email: str = "",
    top_k: int = 8,
    embed_model: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    skip_query_reformulation: bool = False,
    generate_answer: bool = False,
    hybrid_rrf: bool = True,
    rrf_k: int = DEFAULT_RRF_K,
    candidate_pool: Optional[int] = None,
    pinecone_rerank: bool = True,
    rerank_pool: int = DEFAULT_RERANK_POOL,
    pinecone_rerank_model: Optional[str] = None,
    lexical_mode: str = "bm25",
) -> Generator[Tuple[str, QueryGraphState], None, None]:
    """Yield ``(node_name, accumulated_state)`` as each node completes.

    Use this from Streamlit to update the status widget in real-time.
    The **last** yielded state is the final result (same as ``run_query_graph``).
    """
    init_langsmith_tracing()
    initial: QueryGraphState = {
        "query": query,
        "user_role": user_role,
        "user_email": user_email,
        "top_k": top_k,
        "embed_model": embed_model,
        "metadata_filter": metadata_filter,
        "skip_query_reformulation": skip_query_reformulation,
        "generate_answer": generate_answer,
        "hybrid_rrf": hybrid_rrf,
        "rrf_k": rrf_k,
        "candidate_pool": candidate_pool,
        "pinecone_rerank": pinecone_rerank,
        "rerank_pool": rerank_pool,
        "pinecone_rerank_model": pinecone_rerank_model,
        "lexical_mode": lexical_mode,
        "graph_trace": [],
    }
    accumulated: QueryGraphState = dict(initial)  # type: ignore[arg-type]
    for event in get_compiled_query_graph().stream(initial):
        for node_name, update in event.items():
            accumulated.update(update)  # type: ignore[arg-type]
            yield node_name, dict(accumulated)  # type: ignore[arg-type]

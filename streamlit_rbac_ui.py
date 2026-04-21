#!/usr/bin/env python3
"""
Streamlit UI: 3-tier RBAC chat (Pinecone + Postgres + Claude answer).

Run from repository root:

  streamlit run streamlit_rbac_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import streamlit as st

from zmr_brain.answer import stream_answer_to_placeholder
from zmr_brain.constants import (
    DEFAULT_RERANK_POOL,
    DEFAULT_RRF_K,
    PINECONE_INDEX,
    TEAM_EMAILS,
    access_tier_for_email,
    namespaces_for_email,
)
from zmr_brain.query_graph import stream_query_graph
from zmr_brain.tracing import init_langsmith_tracing

TEAM_MEMBERS: Dict[str, str] = {
    "Zamir Kazi (CEO)": "zamir@zmrcapital.com",
    "Mike Regan (CIO)": "mregan@zmrcapital.com",
    "Nicole Chang (VP AM)": "nicole@zmrcapital.com",
    "Mike Weiner (Dir AM)": "mikew@zmrcapital.com",
    "Richard Naccarato (Dir AM)": "richard@zmrcapital.com",
    "Chip Gates (VP Acquisitions)": "chip@zmrcapital.com",
    "Kevin Mawby (Sr Analyst)": "kevin@zmrcapital.com",
    "Zach Oseland (Legal)": "zach@zmrcapital.com",
    "Sid Martins (VP Construction)": "sid@zmrcapital.com",
    "Megan Burrows (MD)": "megan@zmrcapital.com",
}
TEAM_NAMES = list(TEAM_MEMBERS.keys())

CHAT_CSS = """
<style>
    .block-container { padding-top: 1rem; }
    [data-testid="stChatMessage"] { padding: 0.75rem 0; }
    div[data-testid="stSidebar"] .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    @keyframes blink { 50% { opacity: 0; } }
    .streaming-cursor { animation: blink 1s step-end infinite; }
</style>
"""


def _rows_simple(chunks: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in chunks:
        text = (c.text or "").strip()
        preview = text[:400] + ("\u2026" if len(text) > 400 else "")
        pm = c.pinecone_metadata or {}
        doc_label = c.doc_name or "\u2014"
        if pm.get("deal_name"):
            doc_label = f"{pm['deal_name']} / {doc_label}"
        row: Dict[str, Any] = {
            "#": c.rank,
            "Document": doc_label,
            "Type": pm.get("doc_type", "\u2014"),
            "Dept": pm.get("department", "\u2014"),
            "Path": (c.source_path or "")[:120],
            "Preview": preview,
        }
        if pm.get("property_name"):
            row["Property"] = pm["property_name"]
        if pm.get("email_from"):
            row["Sent By"] = pm["email_from"].split("<")[0].strip()
        rows.append(row)
    return rows


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi \u2014 I'm **ZMR Brain**. I answer from **your team's ingested documents**. "
                    "Ask about deals, models, memos, rent rolls, agreements, and more. "
                    "I don't answer unrelated general-knowledge questions."
                ),
                "chunks": None,
                "error": None,
            }
        ]


def main() -> None:
    init_langsmith_tracing()
    _init_session()
    st.set_page_config(
        page_title="ZMR Brain",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<p class="sidebar-title">ZMR Brain</p>', unsafe_allow_html=True)
        st.caption("3-tier access \u00b7 Pinecone + Postgres")

        selected_name = st.selectbox(
            "Logged in as",
            TEAM_NAMES,
            index=0,
            help="Determines your access tier (Executive / Full Access).",
        )
        user_email = TEAM_MEMBERS[selected_name]
        tier = access_tier_for_email(user_email)
        tier_label = {
            "executive_only": "Executive (Full + Private)",
            "restricted_accounting": "Full + Accounting",
            "full": "Full Access",
        }.get(tier, tier)
        st.caption(f"Access: **{tier_label}** \u00b7 `{user_email}`")

        top_k = st.slider(
            "Passages to use (top_k)",
            1,
            50,
            8,
            help=(
                "Chunks after rerank, same as API max (50). "
                "Sparse corpora (e.g. HelloData in zmr-brain-full) often benefit from 12–24. "
                "Higher values increase rerank + LLM latency and token use."
            ),
        )

        with st.expander("Speed (latency)"):
            skip_query_reformulation = st.checkbox(
                "Skip query rewrite before search",
                value=os.getenv("ZMR_UI_SKIP_REFORMULATION", "").strip().lower()
                in ("1", "true", "yes"),
                help=(
                    "Skips the extra Claude (Haiku) call that rewrites your question for embedding + "
                    "keyword search. Much faster; use off if retrieval misses on vague phrasing."
                ),
            )
        with st.expander("Advanced retrieval"):
            hybrid_rrf = st.checkbox("Hybrid (semantic + keyword + RRF)", value=True)
            rrf_k = st.number_input("RRF k", 1, 200, DEFAULT_RRF_K)
            pinecone_rerank = st.checkbox("Rerank with Pinecone", value=True)
            rerank_pool = st.number_input("Rerank pool", 5, 200, DEFAULT_RERANK_POOL)
            embed_model = st.text_input(
                "Voyage query model (blank = auto)", value=""
            )

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Conversation cleared. What would you like to know?",
                    "chunks": None,
                    "error": None,
                }
            ]
            st.rerun()

    st.title("Chat")
    st.caption(
        "Ask about **ZMR** deals, documents, and portfolio context. "
        "General trivia (geography, unrelated products, etc.) is out of scope."
    )

    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"], avatar="\U0001f9d1" if msg["role"] == "user" else "\u2728"
        ):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("chunks"):
                with st.expander("View sources", expanded=False):
                    st.dataframe(
                        _rows_simple(msg["chunks"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    with st.expander("Full passage text"):
                        for c in msg["chunks"]:
                            pm = c.pinecone_metadata or {}
                            header = f"**#{c.rank} \u2014 {c.doc_name or c.vector_id}**"
                            if pm.get("deal_name"):
                                header += f"  |  Deal: {pm['deal_name']}"
                            if pm.get("property_name"):
                                header += f"  |  Property: {pm['property_name']}"
                            if pm.get("email_from"):
                                header += f"  |  From: {pm['email_from'].split('<')[0].strip()}"
                            st.markdown(header)
                            st.text((c.text or "(empty)")[:8000])
                            st.divider()

    prompt = st.chat_input("Message ZMR Brain\u2026")
    if not prompt or not prompt.strip():
        return

    user_text = prompt.strip()
    st.session_state.messages.append(
        {"role": "user", "content": user_text, "chunks": None, "error": None}
    )
    with st.chat_message("user", avatar="\U0001f9d1"):
        st.markdown(user_text)

    em = embed_model.strip() or None

    with st.chat_message("assistant", avatar="\u2728"):
        status = st.status("Thinking\u2026", expanded=True)
        status.write("\U0001f50d Understanding your question\u2026")

        try:
            final = None
            for node_name, state in stream_query_graph(
                user_text,
                "executive",
                user_email=user_email,
                top_k=top_k,
                embed_model=em,
                skip_query_reformulation=skip_query_reformulation,
                # False: synthesize in the UI with stream_answer_to_placeholder (Claude token stream).
                # True: full answer inside the graph (blocking); no token streaming in Streamlit.
                generate_answer=False,
                hybrid_rrf=hybrid_rrf,
                rrf_k=int(rrf_k),
                pinecone_rerank=pinecone_rerank,
                rerank_pool=int(rerank_pool),
                pinecone_rerank_model=None,
                lexical_mode="bm25",
            ):
                final = state
                if node_name == "route":
                    status.update(label="Routing\u2026")
                elif node_name == "reformulate":
                    rq = (state.get("retrieval_query") or "").strip()
                    uq = (state.get("query") or "").strip()
                    if rq and uq and rq.lower() != uq.lower():
                        status.write(f"\u270f\ufe0f Reformulated search \u2192 `{rq}`")
                    else:
                        status.write("\u270f\ufe0f Search query ready")
                elif node_name == "retrieve":
                    status.update(label="Gathering information\u2026")
                    ch = state.get("chunks") or []
                    err = state.get("error")
                    if err and not ch:
                        status.write(f"\U0001f4da Gathering information\u2026 _{err}_")
                    else:
                        idx_label = ", ".join(namespaces_for_email(user_email)) or PINECONE_INDEX
                        status.write(
                            f"\U0001f4da Retrieved **{len(ch)}** passage(s) from `{idx_label}`"
                        )
                    if ch and not err:
                        trace = state.get("graph_trace") or []
                        if any("rerank" in t for t in trace):
                            status.write("\U0001f504 Reranked results")
                        status.write("\U0001f4ac Generating answer\u2026")
                        status.update(label="Generating answer\u2026")
                elif node_name == "direct_reply":
                    status.update(label="Preparing reply\u2026")
                    status.write("\u2728 Finishing response\u2026")

            if final is None:
                raise RuntimeError("Query pipeline produced no result")
        except Exception as e:
            status.update(label="Error", state="error", expanded=False)
            st.error(f"**Something went wrong:** {e}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"**Something went wrong:** {e}",
                    "chunks": None,
                    "error": str(e),
                }
            )
            return

        chunks = final.get("chunks") or []
        err = final.get("error")
        graph_answer = (final.get("answer") or "").strip()

        if final.get("meta_intro") or final.get("refuse_out_of_scope"):
            status.update(label="Done", state="complete", expanded=False)
            st.markdown(graph_answer)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": graph_answer,
                    "chunks": None,
                    "error": None,
                }
            )
            return

        if err and not chunks:
            status.update(label="Search failed", state="error", expanded=False)
            body = f"**Could not search:** {err}"
            st.markdown(body)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": body,
                    "chunks": None,
                    "error": err,
                }
            )
            return

        rq = (final.get("retrieval_query") or "").strip()
        uq = (final.get("query") or "").strip()

        status.update(label="Done", state="complete", expanded=False)

        if graph_answer:
            # Intro / refuse / any future graph path that sets ``answer`` without UI-side streaming.
            st.markdown(graph_answer)
            display_answer = graph_answer
        else:
            answer_placeholder = st.empty()
            display_answer = stream_answer_to_placeholder(
                user_text, chunks, answer_placeholder
            )

        if chunks:
            with st.expander("View sources", expanded=False):
                st.dataframe(
                    _rows_simple(chunks),
                    use_container_width=True,
                    hide_index=True,
                )
                with st.expander("Full passage text"):
                    for c in chunks:
                        pm = c.pinecone_metadata or {}
                        header = f"**#{c.rank} \u2014 {c.doc_name or c.vector_id}**"
                        if pm.get("deal_name"):
                            header += f"  |  Deal: {pm['deal_name']}"
                        if pm.get("property_name"):
                            header += f"  |  Property: {pm['property_name']}"
                        if pm.get("email_from"):
                            header += f"  |  From: {pm['email_from'].split('<')[0].strip()}"
                        st.markdown(header)
                        st.text((c.text or "(empty)")[:8000])
                        st.divider()

        header_parts: List[str] = []
        if rq and uq and rq.lower() != uq.lower():
            header_parts.append(f"_Retrieval query:_ `{rq}`")
        if err:
            header_parts.append(f"\u26a0\ufe0f _{err}_")
        header = "\n".join(header_parts)
        full_body = (header + "\n\n" + display_answer).strip() if header else display_answer

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_body,
                "chunks": chunks if chunks else None,
                "error": err,
            }
        )


if __name__ == "__main__":
    main()

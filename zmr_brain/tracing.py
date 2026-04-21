"""LangSmith tracing for LangGraph (env-driven; no secrets in code)."""

from __future__ import annotations

import os

# Default project in LangSmith UI (override with LANGCHAIN_PROJECT).
DEFAULT_LANGCHAIN_PROJECT = "Zmr-brain-dev"


def init_langsmith_tracing() -> None:
    """
    Enable LangSmith when LANGCHAIN_TRACING_V2=true and an API key is set.

    Accepts either LANGCHAIN_API_KEY or LANGSMITH_API_KEY (latter is copied for LangChain).
    Sets LANGCHAIN_PROJECT default to Zmr-brain-dev if unset.

    Safe to call on every graph run (idempotent, cheap).
    """
    sm_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    lc_key = (os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if sm_key and not lc_key:
        os.environ["LANGCHAIN_API_KEY"] = sm_key

    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    if not tracing_on:
        return

    if not (os.getenv("LANGCHAIN_API_KEY") or "").strip():
        return

    os.environ.setdefault("LANGCHAIN_PROJECT", DEFAULT_LANGCHAIN_PROJECT)

"""Detect chatbot / identity questions that must not be answered from RAG passages."""

from __future__ import annotations

import re

# Whole-message patterns only (avoids matching "help me understand this deal").
_META_RE = re.compile(
    r"^[\s]*("
    r"who\s+are\s+you|what\s+are\s+you|what\s+is\s+zmr\s+brain|"
    r"what\s+can\s+you\s+do|what\s+do\s+you\s+do|how\s+do\s+you\s+work|how\s+does\s+this\s+work|"
    r"tell\s+me\s+about\s+yourself|introduce\s+yourself|"
    r"hello|hi|hey|good\s+(morning|afternoon|evening)|"
    r"help"
    r")[\s!?.,]*$",
    re.IGNORECASE | re.DOTALL,
)


def is_chatbot_meta_query(question: str) -> bool:
    """True if the user is asking about the assistant itself or generic greetings/help."""
    q = (question or "").strip()
    if not q:
        return False
    if _META_RE.match(q):
        return True
    # Short line, no document-looking substance
    if len(q) <= 24 and q.lower().rstrip("?!.") in ("hi", "hey", "hello", "help", "yo"):
        return True
    return False


def chatbot_meta_reply(_question: str) -> str:
    """Fixed copy: who we are without using retrieved documents."""
    return (
        "I'm **ZMR Brain**, ZMR Capital's internal assistant. "
        "I search **your team's ingested documents**—which files I can see depends on the **role** "
        "you select in the sidebar (each role maps to its own secure knowledge index).\n\n"
        "Ask me about deals, underwriting models, offering memos, rent rolls, financials, "
        "confidentiality agreements, and anything else we've loaded for that role. "
        "I combine semantic and keyword search, then summarize what I find and point you to the source files.\n\n"
        "What would you like to know?"
    )

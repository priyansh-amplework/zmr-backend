"""
Route user messages: assistant intro, out-of-scope general knowledge, or document retrieval.

Only ``document`` runs Pinecone/Postgres search. Expand heuristics carefully—when unsure,
we default to ``document`` so real work questions are not blocked.
"""

from __future__ import annotations

import re
from typing import Literal

from zmr_brain.meta_queries import is_chatbot_meta_query

QueryKind = Literal["intro", "refuse", "document"]

# If the query mentions ZMR / real-estate / ingested-doc context, always use retrieval (not general refuse).
_ZMR_WORK_HINT = re.compile(
    r"\b("
    r"zmr|zmr\s+capital|rent\s*roll|offering\s+memorandum|confidentiality|underwriting|"
    r"hellodata|noi|cap\s*rate|acquisition|portfolio|asset\s+management|deal|property|"
    r"memorandum|tenant|lease|refinance|loan|spreadsheet|\.xls|xlsx|xlsm|model|"
    r"om\b|ca\b|agreement|jv\b|joint\s+venture"
    r")\b",
    re.IGNORECASE,
)

# Obvious general-knowledge patterns (only used when no ZMR work hint matches).
_GENERAL_GEO = re.compile(
    r"\b(where\s+is|where\s+are)\s+(the\s+)?("
    r"london|paris|tokyo|berlin|sydney|madrid|rome|england|france|japan|germany|"
    r"mars|the\s+moon|california|texas"
    r")\b",
    re.IGNORECASE,
)
_CAPITAL_WEATHER = re.compile(
    r"\b(what\s+is\s+the\s+capital\s+of|weather\s+in|temperature\s+in)\b",
    re.IGNORECASE,
)
_UNITY_STATUS = re.compile(
    r"\b(status\s+of\s+unity|unity\s+engine|unity\s+3d|unity\s+game)\b",
    re.IGNORECASE,
)
_SPORTS_POLITICS_TRIVIA = re.compile(
    r"\b(who\s+won\s+the\s+super\s+bowl|who\s+is\s+the\s+president\s+of|nba\s+finals|world\s+cup\s+winner)\b",
    re.IGNORECASE,
)

# "Who is Narendra Modi" / "who was Einstein" — general biography, not a deal party (no retrieval).
_WHO_IS_LEAD = re.compile(
    r"^\s*who\s+(is|are|was|were)\s+",
    re.IGNORECASE,
)
# If the user names a work/deal role, send to document search (e.g. who is the borrower).
_WORK_SUBJECT_AFTER_WHO = re.compile(
    r"\b(the\s+)?("
    r"sponsor|borrower|guarantor|seller|buyer|landlord|tenant|lender|issuer|signatory|counterparty|"
    r"party|operator|manager|owner|vendor|obligor|grantor|trustee|ceo|cfo|coo|principal|director|"
    r"contact|author|preparer|signer|witness|custodian|escrow|responsible|signing|obligated|"
    r"undersigned|recipient|addressee"
    r")\b",
    re.IGNORECASE,
)


OUT_OF_SCOPE_REPLY = (
    "I'm **ZMR Brain**, and I only answer questions using **ZMR Capital's ingested documents** "
    "for your selected role—deals, models, legal agreements, rent rolls, offering memos, and similar. "
    "I can't help with general knowledge (geography, public figures, world news, unrelated products, etc.).\n\n"
    "Ask something about our materials or your portfolio context."
)


def classify_query(question: str) -> QueryKind:
    """
    ``intro`` — who are you / what do you do / hello (no retrieval).
    ``refuse`` — clearly general knowledge unrelated to ZMR work (no retrieval).
    ``document`` — run normal RAG retrieval (+ optional Claude on passages).
    """
    q = (question or "").strip()
    if not q:
        return "document"
    if is_chatbot_meta_query(q):
        return "intro"
    if _ZMR_WORK_HINT.search(q):
        return "document"
    if _who_is_general_biography(q):
        return "refuse"
    if _looks_like_general_trivia(q):
        return "refuse"
    return "document"


def _who_is_general_biography(q: str) -> bool:
    """
    ``who is / who was …`` about a person or name, without deal/work roles → general knowledge.
    Avoids retrieval + empty-chunk noise for questions like "who is Narendra Modi".
    """
    if not _WHO_IS_LEAD.match(q):
        return False
    if _WORK_SUBJECT_AFTER_WHO.search(q):
        return False
    return True


def _looks_like_general_trivia(q: str) -> bool:
    if _GENERAL_GEO.search(q):
        return True
    if _CAPITAL_WEATHER.search(q):
        return True
    if _UNITY_STATUS.search(q):
        return True
    if _SPORTS_POLITICS_TRIVIA.search(q):
        return True
    return False

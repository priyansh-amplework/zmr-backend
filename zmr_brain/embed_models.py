"""
Voyage embedding model selection — for both **document** ingestion and **query** time.

**Document** model: chosen from the Drive-relative file path (legal / finance / general).

**Query** model: chosen by analyzing the user's question for domain-specific keywords.
This ensures the query vector lands in the same embedding space as the stored vectors.

- **Law** — ``VOYAGE_EMBED_MODEL_LAW`` (default ``voyage-law-2``)
- **Finance** — ``VOYAGE_EMBED_MODEL_FINANCE`` (default ``voyage-finance-2``)
- **General** — ``VOYAGE_EMBED_MODEL_GENERAL`` (default ``voyage-3-large``)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

# Sample-docs root folders (paths are relative to each Drive root, POSIX-style).
_LAW_FOLDER_PREFIXES: tuple[str, ...] = ("confidentiality agreement/",)

_FINANCE_FOLDER_PREFIXES: tuple[str, ...] = (
    "models/",
    "trailing financials/",
    "rent rolls/",
    "offering memorandums/",
)

# Deal-path or filename hints: financial / deal materials (checked before generic legal keywords).
_FINANCE_MARKERS: tuple[str, ...] = (
    "offering memorandum",
    "offering_memorandum",
    "offering memorandums",
    "_om.pdf",
    "_om_ipa",
    " om.pdf",
    " om ",
    " om_",
    " om-",
    "rent roll",
    "rentroll",
    "rent_roll",
    "t12",
    "trailing 12",
    "trailing12",
    "trailing financial",
    "operating statement",
    "income statement",
    "box score",
    "financial statement",
    "debt matrix",
    "debt indication",
    "underwriting",
    "fee sheet",
    "aged receivable",
    "amenity report",
    "demographic",
    ".xlsm",
    ".xlsx",
    ".csv",
)

# Legal agreements and similar (avoid bare "agreement" — matches many OM titles).
_LAW_MARKERS: tuple[str, ...] = (
    "confidentiality",
    "non-disclosure",
    "non_disclosure",
    "nda",
    "lease agreement",
    "loan agreement",
    "credit agreement",
    "promissory note",
    "purchase and sale agreement",
    "joint venture agreement",
    "confidentiality agreement",
    "subscription agreement",
    "side letter",
)


def select_voyage_embed_model_for_source_path(
    path: str,
    *,
    law_model: Optional[str] = None,
    finance_model: Optional[str] = None,
    general_model: Optional[str] = None,
) -> str:
    """
    Return the Voyage model name used to embed **document** chunks for this source path.

    Order: known legal folder → known finance folders → finance filename heuristics
    → legal text heuristics → general.
    """
    law_model = law_model or os.getenv("VOYAGE_EMBED_MODEL_LAW", "voyage-law-2")
    finance_model = finance_model or os.getenv(
        "VOYAGE_EMBED_MODEL_FINANCE", "voyage-finance-2"
    )
    general_model = general_model or os.getenv(
        "VOYAGE_EMBED_MODEL_GENERAL", "voyage-3-large"
    )

    pl = (path or "").replace("\\", "/").strip().lower()
    if not pl:
        return general_model

    if any(pl.startswith(p) for p in _LAW_FOLDER_PREFIXES):
        return law_model

    for prefix in _FINANCE_FOLDER_PREFIXES:
        if pl.startswith(prefix):
            return finance_model

    if any(m in pl for m in _FINANCE_MARKERS):
        return finance_model

    if any(m in pl for m in _LAW_MARKERS):
        return law_model

    return general_model


# ── Query-time embedding model classification ────────────────────────────────

_FINANCE_QUERY_KEYWORDS: tuple[str, ...] = (
    "rent roll", "rentroll", "rent_roll",
    "t12", "t-12", "trailing 12", "trailing twelve",
    "trailing financial", "operating statement",
    "income statement", "financial statement",
    "offering memorandum", " om ", "proforma", "pro forma",
    "underwriting", "underwrite",
    "cap rate", "noi", "net operating income",
    "purchase price", "asking price",
    "debt", "loan", "mortgage", "leverage",
    "occupancy", "vacancy", "effective rent",
    "revenue", "expense", "gross income",
    "capital expenditure", "capex",
    "cash flow", "irr", "return on",
    "yield", "spread", "basis point",
    "unit mix", "square footage", "sf",
    "model", "excel", "spreadsheet", "workbook",
    "budget", "forecast", "projection",
    "fee sheet", "demographic", "box score",
    "aged receivable",
    "property financials", "deal financials",
    "operating budget", "capital budget",
)

_LAW_QUERY_KEYWORDS: tuple[str, ...] = (
    "confidentiality agreement", "confidentiality",
    "non-disclosure", "nda",
    "lease agreement", "loan agreement",
    "credit agreement", "promissory note",
    "purchase and sale agreement", "psa",
    "joint venture agreement", "jv agreement",
    "subscription agreement", "side letter",
    "legal", "contract", "amendment", "addendum",
    "deed", "title", "closing document",
    "settlement", "estoppel",
    "indemnification", "liability",
    "tenant lease", "lease term",
    "compliance", "regulatory",
)


def select_voyage_embed_model_for_query(
    query: str,
    *,
    law_model: Optional[str] = None,
    finance_model: Optional[str] = None,
    general_model: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Classify a user query and return ``(model_name, domain)``
    where domain is ``'finance'``, ``'law'``, or ``'general'``.

    Uses keyword matching against the query text to pick the embedding model
    that matches the vectors stored during ingestion.
    """
    law_model = law_model or os.getenv("VOYAGE_EMBED_MODEL_LAW", "voyage-law-2")
    finance_model = finance_model or os.getenv(
        "VOYAGE_EMBED_MODEL_FINANCE", "voyage-finance-2"
    )
    general_model = general_model or os.getenv(
        "VOYAGE_EMBED_MODEL_GENERAL", "voyage-3-large"
    )

    ql = (query or "").lower()
    if not ql.strip():
        return general_model, "general"

    finance_hits = sum(1 for kw in _FINANCE_QUERY_KEYWORDS if kw in ql)
    law_hits = sum(1 for kw in _LAW_QUERY_KEYWORDS if kw in ql)

    if finance_hits > law_hits and finance_hits >= 1:
        return finance_model, "finance"
    if law_hits > finance_hits and law_hits >= 1:
        return law_model, "law"
    if finance_hits == law_hits and finance_hits >= 1:
        return finance_model, "finance"

    return general_model, "general"

"""
ZMR Brain constants — 3-tier access model.

RBAC uses **three separate Pinecone indexes** (one per tier):

  - ``zmr-brain-full``                  → public / team content
  - ``zmr-brain-executive-only``        → private / HR / Corporate Ops
  - ``zmr-brain-restricted-accounting`` → sensitive accounting subfolders

Query-time access: :func:`namespaces_for_email` / :func:`pinecone_indexes_for_email`
return which **index names** a user may search (1–3 indexes). Ingestion routes
vectors to the index for the content’s ``access_tier``; vectors use the default
namespace (empty string) within each index.

Field names for Pinecone / Drive / Postgres metadata: :mod:`zmr_brain.metadata_schema`.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set

PINECONE_INDEX_BY_TIER: Dict[str, str] = {
    "full": "zmr-brain-full",
    "executive_only": "zmr-brain-executive-only",
    "restricted_accounting": "zmr-brain-restricted-accounting",
}

# Default single-index alias (public tier) — backward compat for scripts using one name.
PINECONE_INDEX = PINECONE_INDEX_BY_TIER["full"]

ACCESS_TIERS = ("full", "executive_only", "restricted_accounting")

# Legacy: tier “label” used in metadata; not a Pinecone namespace name when using 3 indexes.
TIER_NAMESPACE: Dict[str, str] = {
    "full": "full",
    "executive_only": "executive_only",
    "restricted_accounting": "restricted_accounting",
}

# Legacy mapping kept for backward compatibility during migration.
INDEX_BY_ROLE: Dict[str, str] = {
    "executive": PINECONE_INDEX,
    "acquisitions": PINECONE_INDEX,
    "asset-management": PINECONE_INDEX,
    "investor-relations": PINECONE_INDEX,
    "legal": PINECONE_INDEX,
    "compliance": PINECONE_INDEX,
}

DEFAULT_USER_ROLE = "executive"

EXECUTIVE_EMAILS: FrozenSet[str] = frozenset({
    "zamir@zmrcapital.com",
    "mregan@zmrcapital.com",
})

ACCOUNTING_EMAILS: FrozenSet[str] = frozenset({
    # Property Accountant email TBD — add when onboarded
})

TEAM_EMAILS: FrozenSet[str] = frozenset({
    "zamir@zmrcapital.com",
    "mregan@zmrcapital.com",
    "nicole@zmrcapital.com",
    "mikew@zmrcapital.com",
    "richard@zmrcapital.com",
    "chip@zmrcapital.com",
    "kevin@zmrcapital.com",
    "zach@zmrcapital.com",
    "sid@zmrcapital.com",
    "megan@zmrcapital.com",
})

DEFAULT_VOYAGE_QUERY_MODEL = "voyage-3-large"

DEFAULT_RRF_K = 30
DEFAULT_HYBRID_CANDIDATE_POOL = 9
DEFAULT_RERANK_POOL = 9
DEFAULT_BM25_MAX_CORPUS = 2000

# Stored in Postgres ``pinecone_namespace`` when using per-tier indexes (default namespace).
DEFAULT_PINECONE_NAMESPACE = ""


def access_tier_for_email(email: str) -> str:
    """Map a user email to their access tier."""
    e = (email or "").strip().lower()
    if e in EXECUTIVE_EMAILS:
        return "executive_only"
    if e in ACCOUNTING_EMAILS:
        return "restricted_accounting"
    return "full"


def pinecone_index_for_tier(access_tier: str) -> str:
    """Pinecone index name for ingesting/querying a given ``access_tier``."""
    return PINECONE_INDEX_BY_TIER.get(access_tier, PINECONE_INDEX_BY_TIER["full"])


def pinecone_indexes_for_email(user_email: str) -> List[str]:
    """Return the Pinecone **index names** a user is allowed to query.

    - Executive  → all 3 indexes
    - Restricted accounting → ``full`` + ``restricted_accounting`` indexes
    - Full (regular team)   → ``full`` index only
    """
    tier = access_tier_for_email(user_email)
    if tier == "executive_only":
        return list(PINECONE_INDEX_BY_TIER.values())
    if tier == "restricted_accounting":
        return [
            PINECONE_INDEX_BY_TIER["full"],
            PINECONE_INDEX_BY_TIER["restricted_accounting"],
        ]
    return [PINECONE_INDEX_BY_TIER["full"]]


def namespaces_for_email(user_email: str) -> List[str]:
    """Return Pinecone index names the user may query (3-tier RBAC).

    Name kept for backward compatibility; values are **index names**, not logical
    namespace strings inside a single index.
    """
    return pinecone_indexes_for_email(user_email)


def namespace_for_tier(access_tier: str) -> str:
    """Map an access tier to a Pinecone **index** name (for ingestion).

    Deprecated name: historically returned a *namespace* string; now returns the
    per-tier index name. Use :func:`pinecone_index_for_tier` in new code.
    """
    return pinecone_index_for_tier(access_tier)


def pinecone_access_filter(user_email: str) -> Dict | None:
    """Legacy metadata-filter approach (kept for backward compatibility).

    Prefer :func:`pinecone_indexes_for_email` for multi-index RBAC.
    """
    tier = access_tier_for_email(user_email)
    if tier == "executive_only":
        return None
    if tier == "restricted_accounting":
        return {"access_tier": {"$nin": ["executive_only"]}}
    return {"access_tier": {"$nin": ["executive_only", "restricted_accounting"]}}

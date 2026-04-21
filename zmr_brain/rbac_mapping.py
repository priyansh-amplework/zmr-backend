"""
3-tier RBAC mapping: Drive path → access_tier, department, property_name.

Access tiers (from client spec):
  - ``full``                  → Public Drive content (everyone)
  - ``executive_only``        → Private Drive (HR, Corporate Ops)
  - ``restricted_accounting`` → Sensitive accounting subcategories

Used by ingestion scripts to tag Pinecone metadata and Postgres rows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

# ── Private Drive departments (executive_only) ───────────────────────────────

_PRIVATE_DEPARTMENTS: tuple[str, ...] = (
    "human resources",
    "corporate operations",
)

# ── Restricted accounting subcategories ──────────────────────────────────────
# Path is lowercased before matching. A path must also look like **Accounting &
# Tax** (see ``_accounting_path_context``) so "tax return" in unrelated folders
# does not flip to restricted.

_RESTRICTED_ACCOUNTING_SUBCATEGORIES: tuple[str, ...] = (
    "bank statements",
    "bank statement",
    "bankstatement",
    "bank_statements",
    "tax returns",
    "tax return",
    "taxreturns",
    "audit & review",
    "audit and review",
    "audit review",
    "audit/review",
)


def _accounting_path_context(pl: str) -> bool:
    """True if path is under an Accounting & Tax style branch (public)."""
    if "accounting & tax" in pl or "accounting and tax" in pl:
        return True
    if "/accounting/" in pl or pl.endswith("/accounting"):
        return True
    if "corporate" in pl and "accounting" in pl:
        return True
    return False

# ── Department names (canonical, from client spec) ───────────────────────────

DEPARTMENTS: tuple[str, ...] = (
    "Acquisitions",
    "Due Diligence",
    "Closing",
    "Capital Markets & Debt",
    "Investor Relations & Capital Raising",
    "JV Partners & Co-Investors",
    "Asset Management",
    "3rd Party Property Management",
    "Construction & Capital Projects",
    "Insurance & Risk Management",
    "Legal & Compliance",
    "Accounting & Tax",
    "Marketing & Leasing",
    "Dispositions & Exits",
    "Corporate Operations",
    "Human Resources",
)

_DEPARTMENT_KEYWORDS: Dict[str, str] = {
    "acquisitions": "Acquisitions",
    "due diligence": "Due Diligence",
    "closing": "Closing",
    "capital markets": "Capital Markets & Debt",
    "debt": "Capital Markets & Debt",
    "investor relations": "Investor Relations & Capital Raising",
    "capital raising": "Investor Relations & Capital Raising",
    "jv partners": "JV Partners & Co-Investors",
    "co-investors": "JV Partners & Co-Investors",
    "asset management": "Asset Management",
    "property management": "3rd Party Property Management",
    "pm reports": "3rd Party Property Management",
    "construction": "Construction & Capital Projects",
    "capital projects": "Construction & Capital Projects",
    "insurance": "Insurance & Risk Management",
    "risk management": "Insurance & Risk Management",
    "legal": "Legal & Compliance",
    "compliance": "Legal & Compliance",
    "accounting": "Accounting & Tax",
    "tax": "Accounting & Tax",
    "marketing": "Marketing & Leasing",
    "leasing": "Marketing & Leasing",
    "dispositions": "Dispositions & Exits",
    "exits": "Dispositions & Exits",
    "corporate operations": "Corporate Operations",
    "human resources": "Human Resources",
    "hr": "Human Resources",
    # Folder names from existing Drive structure
    "models": "Acquisitions",
    "offering memorandums": "Acquisitions",
    "trailing financials": "Asset Management",
    "rent rolls": "Asset Management",
    "confidentiality agreement": "Legal & Compliance",
}

# ── Asset alias registry ─────────────────────────────────────────────────────

ASSET_ALIASES: Dict[str, str] = {
    "sonoma pointe": "Skye at Hunter's Creek",
    "sunridge": "Skye Ridge",
    "pecan square": "Skye Isle",
    "bayou bend": "Skye at Love",
    "park place": "The Boardwalk",
    "palms at palisades": "Skye Oaks",
    "brandon oaks": "Skye Oaks",
}

CANONICAL_ASSETS: tuple[str, ...] = (
    "Skye at Hunter's Creek",
    "Skye Ridge",
    "Skye Isle",
    "Skye at Love",
    "Skye at Conway",
    "Skye Reserve",
    "Crossing at Palm Aire",
    "Preserve at Riverwalk",
    "Julia",
    "Nightingale",
    "Parks at Walnut",
    "9944",
    "Skye Oaks",
    "Hanley Place",
    "The Boardwalk",
    "Upland Townhomes",
    "Park at Peachtree Hills",
    "Flats",
    # Disposed
    "Las Lomas",
    "Laurel at Altamonte",
    "Camila",
    "Alexandria Landings",
    "Desert Peaks",
)

_CANONICAL_ASSETS_LOWER = {a.lower(): a for a in CANONICAL_ASSETS}

PORTFOLIO_ASSETS: Dict[str, List[str]] = {
    "Walnut Portfolio": ["Parks at Walnut", "9944"],
    "Slate Portfolio": [
        "Skye Oaks", "Hanley Place", "The Boardwalk",
        "Upland Townhomes", "Park at Peachtree Hills", "Flats",
    ],
}


# ── Core mapping functions ───────────────────────────────────────────────────

def access_tier_for_path(
    path: str,
    *,
    source_drive: str = "public",
) -> str:
    """
    Determine access_tier from a file path.

    - Private Drive → ``executive_only``
    - Public paths under **Accounting & Tax** with a restricted subfolder name
      (bank statements, tax returns, audit review, etc.) → ``restricted_accounting``
    - Everything else → ``full``
    """
    pl = (path or "").replace("\\", "/").strip().lower().replace("%20", " ")
    if source_drive == "private":
        return "executive_only"
    for dept in _PRIVATE_DEPARTMENTS:
        if dept in pl:
            return "executive_only"
    if source_drive != "private" and _accounting_path_context(pl):
        for sub in _RESTRICTED_ACCOUNTING_SUBCATEGORIES:
            if sub in pl:
                return "restricted_accounting"
    return "full"


def source_drive_for_path(path: str) -> str:
    """Infer source_drive from path. Private Drive content has HR/Corporate Ops keywords."""
    pl = (path or "").replace("\\", "/").strip().lower()
    for dept in _PRIVATE_DEPARTMENTS:
        if dept in pl:
            return "private"
    return "public"


def infer_department_from_path(path: str) -> str:
    """Extract department name from a folder path."""
    pl = (path or "").replace("\\", "/").strip().lower()
    parts = [p.strip() for p in pl.split("/") if p.strip()]
    for part in parts:
        if part in _DEPARTMENT_KEYWORDS:
            return _DEPARTMENT_KEYWORDS[part]
    for kw, dept in _DEPARTMENT_KEYWORDS.items():
        if kw in pl:
            return dept
    return "General"


def infer_property_name_from_path(path: str) -> str:
    """
    Extract canonical property/asset name from a file path.
    Checks path segments against the asset registry and aliases.
    """
    pl = (path or "").replace("\\", "/").strip().lower()
    parts = [p.strip() for p in pl.split("/") if p.strip()]

    for part in parts:
        if part in _CANONICAL_ASSETS_LOWER:
            return _CANONICAL_ASSETS_LOWER[part]
        if part in ASSET_ALIASES:
            return ASSET_ALIASES[part]
        for alias, canonical in ASSET_ALIASES.items():
            if alias in part:
                return canonical
        for canon_lower, canon in _CANONICAL_ASSETS_LOWER.items():
            if canon_lower in part:
                return canon

    return ""


def infer_doc_type_from_path(path: str) -> str:
    """Classify document type from its file path."""
    lower = (path or "").replace("\\", "/").lower()
    if "confidentiality" in lower or "non-disclosure" in lower or "nda" in lower:
        return "legal_agreement"
    if any(x in lower for x in ("lease agreement", "loan agreement", "credit agreement")):
        return "legal_agreement"
    if "offering" in lower or "memorandum" in lower:
        return "offering_memorandum"
    if "rent" in lower and ("roll" in lower or "rr" in lower):
        return "rent_roll"
    if any(x in lower for x in ("trailing", "t12", "financial statement", "income statement")):
        return "financial_statement"
    if any(x in lower for x in ("model", ".xlsm")):
        return "underwriting_model"
    if "insurance" in lower or "policy" in lower:
        return "insurance_policy"
    if any(x in lower for x in ("jv", "joint venture", "operating agreement")):
        return "jv_agreement"
    if "budget" in lower:
        return "budget"
    return "general"


# ── Legacy compatibility helpers ─────────────────────────────────────────────

def map_drive_path_to_roles(file_path: str) -> List[str]:
    """Legacy: return roles list. Now always returns ["full"] + executive for
    backward compat with ingestion scripts during migration."""
    tier = access_tier_for_path(file_path)
    if tier == "executive_only":
        return ["executive"]
    return ["executive", "acquisitions"]


def index_by_role_dict() -> Dict[str, str]:
    """Legacy: all roles map to the single index."""
    from zmr_brain.constants import INDEX_BY_ROLE
    return dict(INDEX_BY_ROLE)


def csp_lite_metadata(roles: List[str]) -> Tuple[bool, List[str], str]:
    """Legacy stub for backward compat."""
    return False, ["full"], "csp_lite_v2"


def sub_role_for_index(role: str) -> Optional[str]:
    """Legacy stub — no sub-roles in 3-tier model."""
    return None

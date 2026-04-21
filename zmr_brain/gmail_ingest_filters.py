"""
Gmail body ingestion — exclusion rules (spam/promo, calendar bots, system mail).

Used by ``scripts/ingest_gmail_bodies.py``. Tune with env (see ``exclusion_reason``).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set


def _env_csv(name: str) -> Set[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _walk_mime_types(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if not payload:
        return out
    mt = (payload.get("mimeType") or "").strip()
    if mt:
        out.append(mt.lower())
    for part in payload.get("parts") or []:
        out.extend(_walk_mime_types(part))
    return out


def _header_map(msg: Dict[str, Any]) -> Dict[str, str]:
    return {
        h["name"].lower(): h.get("value") or ""
        for h in msg.get("payload", {}).get("headers", [])
        if h.get("name")
    }


def _ics_attachment_names(payload: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    if not payload:
        return names
    fn = (payload.get("filename") or "").strip()
    if fn.lower().endswith(".ics"):
        names.append(fn)
    for part in payload.get("parts") or []:
        names.extend(_ics_attachment_names(part))
    return names


# Default: obvious Google / Workspace system + common file-share bots.
_DEFAULT_SYSTEM_SENDERS = frozenset(
    {
        "google-workspace-alerts-noreply@google.com",
        "workspace-noreply@google.com",
        "drive-shares-noreply@google.com",
        "calendar-notification@google.com",
        "group-calendar@google.com",
        "noreply@google.com",
        "mailer-daemon@google.com",
        "no-reply@accounts.google.com",
        "mail-noreply@google.com",
        "notifications@github.com",
        "build@github.com",
    }
)

_DEFAULT_SUBJECT_SKIP = (
    "password reset",
    "password has been changed",
    "security issue",
    "shared drive",
    "2-step verification",
    "two-step verification",
    "security alert",
    "sign-in attempt",
    "new sign-in",
    "your google account",
    "verify your email",
    "subscription confirmed",
    "order confirmation",
    "receipt for your",
    "unsubscribe",
    "newsletter",
    "digest:",
    "weekly digest",
    "daily digest",
)

# Meeting / calendar style (English) — skip automated invites & RSVPs.
_MEETING_SUBJECT_PREFIXES = (
    "accepted:",
    "declined:",
    "tentative:",
    "needs action:",
    "invitation:",
    "updated invitation",
    "canceled:",
    "cancelled:",
    "new meeting",
    "meeting request",
    "microsoft teams",
    "zoom meeting",
    "webex meeting",
    "google meet",
    "join with google meet",
)

# Senders that are almost always transactional / promos (extend via env).
_DEFAULT_PROMO_FROM_PATTERNS = (
    "@mailchimp.com",
    "@sendgrid.net",
    "@hubspot.com",
    "@hs-send.com",
    "@amazonses.com",
    "@constantcontact.com",
    "@ccsend.com",
    "@mailgun.org",
    "@mailgun.net",
    "@bounce.",
    "@email.zoom.us",
    "no-reply@zoom.us",
    "noreply@zoom.us",
    "notifications@slack.com",
    "mailer@linkedin.com",
    "jobs-listings@linkedin.com",
)


def _promo_from_match(from_addr: str, pat: str) -> bool:
    pat = pat.strip().lower()
    if not pat or not from_addr:
        return False
    if pat.startswith("@"):
        return from_addr.endswith(pat)
    if "@" in pat:
        dom = pat.split("@", 1)[1]
        return from_addr == pat or from_addr.endswith("@" + dom)
    return pat in from_addr


def _extract_email_addr(raw: str) -> str:
    raw = (raw or "").strip()
    if "<" in raw and ">" in raw:
        return raw.split("<")[1].split(">")[0].strip().lower()
    return raw.lower()


def exclusion_reason(
    msg: Dict[str, Any],
    parsed: Dict[str, Any],
    *,
    extract_addr: Any = _extract_email_addr,
) -> Optional[str]:
    """
    If this message should **not** be ingested, return a short reason label; else ``None``.

    Env overrides (optional, comma-separated where noted):
    - ``GMAIL_INGEST_EXTRA_SYSTEM_SENDERS`` — extra from-addresses (full lower-case addr).
    - ``GMAIL_INGEST_EXTRA_PROMO_FROM_SUFFIXES`` — extra ``@domain`` or substring to match on from.
    - ``GMAIL_INGEST_EXCLUDE_FREEMAIL_SENDERS`` — ``1``/``true``: skip when From is a consumer domain
      (gmail/yahoo/hotmail/outlook/icloud) **and** To/Cc contains no ``@zmrcapital.com`` (weak personal heuristic).
    """
    label_ids = {str(x).upper() for x in (msg.get("labelIds") or [])}
    if "SPAM" in label_ids:
        return "spam_label"
    if "TRASH" in label_ids:
        return "trash_label"
    if "DRAFT" in label_ids:
        return "draft_label"
    if "CATEGORY_PROMOTIONS" in label_ids:
        return "category_promotions"

    headers = _header_map(msg)
    from_raw = parsed.get("from") or headers.get("from", "")
    from_addr = extract_addr(from_raw)
    subject = (parsed.get("subject") or "").strip()
    subj_lower = subject.lower()

    extra_sys = _env_csv("GMAIL_INGEST_EXTRA_SYSTEM_SENDERS")
    system_senders = set(_DEFAULT_SYSTEM_SENDERS) | extra_sys
    if from_addr in system_senders:
        return "system_sender"

    for frag in _DEFAULT_SUBJECT_SKIP:
        if frag in subj_lower:
            return "subject_system_or_promo"

    extra_subj = (os.getenv("GMAIL_INGEST_EXTRA_SUBJECT_FRAGMENTS") or "").strip()
    if extra_subj:
        for frag in extra_subj.split("|"):
            f = frag.strip().lower()
            if f and f in subj_lower:
                return "subject_custom_fragment"

    for pfx in _MEETING_SUBJECT_PREFIXES:
        if subj_lower.startswith(pfx):
            return "meeting_or_invite_subject"

    auto_sub = (headers.get("auto-submitted") or "").lower()
    if auto_sub in ("auto-generated", "auto-replied", "auto-notified"):
        return "header_auto_submitted"

    prec = (headers.get("precedence") or "").lower().strip()
    if prec in ("bulk", "list", "junk"):
        # Newsletters / lists — skip; routine 1:1 mail rarely uses Precedence: bulk.
        return "header_precedence_bulk_or_list"

    # Calendar MIME / .ics
    mimes = _walk_mime_types(msg.get("payload") or {})
    if any("text/calendar" in m for m in mimes):
        return "mime_text_calendar"
    if any("application/ics" in m or m.endswith("/calendar") for m in mimes):
        return "mime_calendar"
    if _ics_attachment_names(msg.get("payload") or {}):
        return "attachment_ics"

    # Promotional / ESP patterns (sender domain)
    extra_promo = [
        x.strip().lower()
        for x in (os.getenv("GMAIL_INGEST_EXTRA_PROMO_FROM_SUFFIXES") or "").split(",")
        if x.strip()
    ]
    for pat in list(_DEFAULT_PROMO_FROM_PATTERNS) + extra_promo:
        if _promo_from_match(from_addr, pat):
            return "promo_or_esp_sender"

    # Optional: consumer freemail senders with no ZMR recipient in To/Cc (non-business heuristic)
    if os.getenv("GMAIL_INGEST_EXCLUDE_FREEMAIL_SENDERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        domain = from_addr.split("@")[-1] if "@" in from_addr else ""
        freemail = {
            "gmail.com",
            "googlemail.com",
            "yahoo.com",
            "yahoo.co.uk",
            "hotmail.com",
            "outlook.com",
            "live.com",
            "msn.com",
            "icloud.com",
            "me.com",
            "mac.com",
        }
        if domain in freemail:
            to_cc = f"{parsed.get('to', '')} {parsed.get('cc', '')}".lower()
            if "zmrcapital.com" not in to_cc:
                return "personal_freemail_no_zmr_recipient"

    return None

"""
Email ↔ document linking pipeline (Gmail API + Postgres + Drive ingest).

**Canonical path:** direct Gmail (``scripts/ingest_gmail_bodies.py``) plus client
Drive trees (``scripts/ingest_drive_full_rag_v2.py`` for public/private roots), not
an external deal-intake HTTP service.

Flow
----
1. **Gmail body ingest** — For each ``@zmrcapital.com`` mailbox, fetch messages,
   store full body text in ``documents`` / ``chunks`` / Pinecone, and insert
   ``email_attachment_links`` rows (one per attachment filename) with
   ``from_addr``, ``to_addr``, ``subject``, ``email_date``, ``gmail_message_id``.

2. **Drive ingest** — When ``ingest_drive_full_rag_v2.py`` (or v1 ``ingest_drive_full_rag.py``) processes a
   file whose ``description`` JSON or path metadata includes the same
   ``email_from`` / ``email_subject`` / ``email_date`` / original filename,
   ``try_match_attachment_link`` sets ``drive_document_id`` and ``drive_file_id``
   on the matching link row so RAG chunks can carry ``email_message_id``.

Canonical metadata keys for documents/chunks align with
:data:`zmr_brain.metadata_schema.DOCUMENT_METADATA_JSON_KEYS` and
``PINECONE_METADATA_EMAIL_KEYS``.
"""

from __future__ import annotations

from typing import Any, Dict


def build_email_document_metadata(
    parsed: Dict[str, Any],
    gmail_account: str,
    run_id: str,
    *,
    source: str = "gmail_body",
    access_tier: str = "full",
) -> Dict[str, Any]:
    """JSON-serializable ``documents.metadata`` for a Gmail-ingested email row."""
    msg_id = parsed["message_id"]
    return {
        "access_tier": access_tier,
        "run_id": run_id,
        "source": source,
        "email_from": parsed.get("from", ""),
        "email_to": parsed.get("to", ""),
        "email_cc": parsed.get("cc", ""),
        "email_subject": parsed.get("subject", ""),
        "email_date": parsed.get("date", ""),
        "email_message_id": msg_id,
        "email_thread_id": parsed.get("thread_id", ""),
        "gmail_account": gmail_account,
        "attachment_filenames": list(parsed.get("attachment_filenames") or []),
    }


def build_email_chunk_metadata_json(
    parsed: Dict[str, Any],
    gmail_account: str,
    *,
    access_tier: str = "full",
) -> Dict[str, Any]:
    """Postgres ``chunks.metadata`` jsonb: same email envelope as the parent document."""
    msg_id = parsed["message_id"]
    return {
        "access_tier": access_tier,
        "email_from": parsed.get("from", ""),
        "email_to": parsed.get("to", ""),
        "email_cc": parsed.get("cc", ""),
        "email_subject": parsed.get("subject", ""),
        "email_date": parsed.get("date", ""),
        "email_message_id": msg_id,
        "email_thread_id": parsed.get("thread_id", ""),
        "gmail_account": gmail_account,
    }

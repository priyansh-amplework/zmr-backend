"""
Canonical metadata contracts — ingestion, Pinecone, Postgres, and client Drive uploads.

Use this module as the single reference when adding fields in:
``ingest_drive_full_rag.py``, ``reingest_to_v2.py``, ``ingest_gmail_bodies.py``,
and the client Gmail→Drive pipeline (``src/main.py``).

**Why Gmail ingestion exists (discovery):** When we ingested Drive documents, we saw
that attachment files already carried **Gmail context** on the Drive side — the
client pipeline writes sender, recipient, subject, and date into each file's
``description`` field as JSON (see :data:`DRIVE_EMAIL_ATTACHMENT_DESCRIPTION_KEYS`).
That showed we could reliably tie attachments back to their email thread. We
**started the Gmail body ingestion pipeline** (``ingest_gmail_bodies.py``) and the
``email_attachment_links`` table so the **full email body** plus **Drive attachment
chunks** stay linkable in Postgres and Pinecone metadata.

Pinecone stores only **string** metadata (and numeric where supported); values are
truncated in upsert paths — keep keys short and values under a few KB.
"""

from __future__ import annotations

from typing import Final, FrozenSet, Tuple

# ── Google Drive ``file.description`` (JSON string) — client attachment pipeline ──

# Written by ``src/main.py`` when saving Gmail attachments to Drive. Ingestion parses
# this in ``parse_email_description`` / ``ingest_drive_full_rag.py``.
DRIVE_EMAIL_ATTACHMENT_DESCRIPTION_KEYS: Final[Tuple[str, ...]] = (
    "from",
    "to",
    "subject",
    "date",
    "originalFilename",
    "extractedBy",
    "classified",
)
# Optional keys after classification runs (same JSON object, updated in place):
DRIVE_EMAIL_ATTACHMENT_OPTIONAL_KEYS: Final[Tuple[str, ...]] = (
    "classifiedAt",
    "classification",
)

# ── Pinecone vector metadata (per chunk) — Engineer 2 ingestion ──

# Core RBAC / routing (namespaces enforce tier; these fields support filtering & UI).
PINECONE_METADATA_CORE_KEYS: Final[Tuple[str, ...]] = (
    "chunk_id",
    "doc_id",  # Drive file id (or logical id for Gmail-derived chunks)
    "doc_name",
    "source_path",
    "mime_type",
    "doc_type",
    "source",  # e.g. google_drive, gmail, email_intake
    "embed_model",
    "access_tier",  # full | executive_only | restricted_accounting
    "source_drive",  # public | private (inferred from path)
    "department",
    "property_name",
    "ingested_at",
    "chunk_index",
    "total_chunks",
    "extraction_quality",
    "is_superseded",
    "chunk_gcs_uri",  # gs://... or local:... chunk body pointer
    "chunk_sha256",
)

# When the source file came from an email attachment (Drive description JSON):
PINECONE_METADATA_EMAIL_KEYS: Final[Tuple[str, ...]] = (
    "email_from",
    "email_to",
    "email_subject",
    "email_date",
    "email_message_id",  # Gmail id when linked via email_attachment_links
)

# ── Postgres ``documents`` / ``documents_v2`` JSON ``metadata`` column ──

# Typical keys stored in ``metadata`` jsonb (not exhaustive — ingestion may add run_id, etc.):
DOCUMENT_METADATA_JSON_KEYS: Final[Tuple[str, ...]] = (
    "access_tier",
    "run_id",
    "extraction_quality",
    "old_source_path",  # reingest_to_v2.py
    "email_from",
    "email_to",
    "email_subject",
    "email_date",
    "email_original_filename",
    "email_extracted_by",
)

# ── Postgres ``chunks`` / ``chunks_v2`` JSON ``metadata`` column ──

CHUNK_METADATA_JSON_KEYS: Final[Tuple[str, ...]] = (
    "access_tier",
    "department",
    "property_name",
    "piece_metadata",  # per-chunk extractor hints (sheet, page, etc.)
    "email_from",
    "email_to",
    "email_cc",
    "email_subject",
    "email_date",
    "email_message_id",
    "email_thread_id",
    "gmail_account",
)

# ── ``email_attachment_links`` (migration 009) — link Gmail ↔ Drive attachment ──

EMAIL_ATTACHMENT_LINK_COLUMNS: Final[Tuple[str, ...]] = (
    "gmail_message_id",
    "gmail_thread_id",
    "gmail_account",
    "from_addr",
    "to_addr",
    "subject",
    "email_date",
    "attachment_filename",
    "email_document_id",
    "drive_document_id",
    "drive_file_id",
    "matched_at",
)

# Valid ``access_tier`` values (also Pinecone namespace names in ``zmr-brain-dev``).
ACCESS_TIER_VALUES: FrozenSet[str] = frozenset({"full", "executive_only", "restricted_accounting"})

def validate_access_tier(value: str) -> bool:
    return (value or "").strip() in ACCESS_TIER_VALUES

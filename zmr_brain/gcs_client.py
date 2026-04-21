"""
Google Cloud Storage helpers — use *your* bucket as an alternative to the client's.

Configure via .env:
  GCS_ARTIFACTS_BUCKET       — required for chunk uploads (ingest scripts do not write local bodies)
  GCS_APPLICATION_CREDENTIALS — optional JSON key path used *only* for Storage
                                (if unset, uses GOOGLE_APPLICATION_CREDENTIALS)
  GCS_PROJECT_ID             — optional; Storage client project (default: key's project_id)
  GCS_ARTIFACTS_PREFIX       — optional object prefix, e.g. zmr-dev/prakash/

Chunk rows may store gs://your-bucket/... in chunk_gcs_uri; that bucket must grant
your SA roles/storage.objectAdmin (or objectViewer for read-only retrieval).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

from google.cloud import storage


def _credentials_json_path() -> Optional[Path]:
    p = os.getenv("GCS_APPLICATION_CREDENTIALS", "").strip()
    if p and Path(p).is_file():
        return Path(p).resolve()
    p2 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if p2 and Path(p2).is_file():
        return Path(p2).resolve()
    root = Path(__file__).resolve().parent.parent
    default = root / "service-account-key.json"
    if default.is_file():
        return default.resolve()
    return None


def storage_client() -> storage.Client:
    """Client for GCS — prefers GCS_APPLICATION_CREDENTIALS, then default ADC path."""
    project = (
        os.getenv("GCS_PROJECT_ID", "").strip()
        or os.getenv("GCLOUD_PROJECT_ID", "").strip()
        or None
    )
    path = _credentials_json_path()
    if path:
        return storage.Client.from_service_account_json(str(path), project=project)
    return storage.Client(project=project)


def parse_gs_uri(uri: str) -> Tuple[str, str]:
    u = (uri or "").strip()
    if not u.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri!r}")
    rest = u[5:]
    if "/" not in rest:
        raise ValueError(f"Invalid gs:// URI (missing object path): {uri!r}")
    bucket, _, blob = rest.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid gs:// URI: {uri!r}")
    return bucket, blob


def download_blob_text(gs_uri: str, *, encoding: str = "utf-8") -> str:
    """Read object body as text (for RAG context). Uses credentials from gcs_client rules."""
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    client = storage_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.download_as_text(encoding=encoding)


def artifacts_bucket_name() -> str:
    name = os.getenv("GCS_ARTIFACTS_BUCKET", "").strip()
    if not name:
        raise RuntimeError(
            "GCS_ARTIFACTS_BUCKET is not set. Ingestion stores chunk bodies only in GCS; "
            "set this in .env to your artifacts bucket."
        )
    return name


def require_gcs_artifacts_bucket_env() -> str:
    """Validate bucket for ingest CLIs and pin stripped ``GCS_ARTIFACTS_BUCKET`` in the environment."""
    name = artifacts_bucket_name()
    os.environ["GCS_ARTIFACTS_BUCKET"] = name
    return name


def _prefix() -> str:
    p = os.getenv("GCS_ARTIFACTS_PREFIX", "").strip().strip("/")
    return f"{p}/" if p else ""


def upload_text(
    text: str,
    *,
    content_type: str = "text/plain; charset=utf-8",
    object_name: Optional[str] = None,
) -> str:
    """
    Upload UTF-8 text to your artifacts bucket. Returns gs:// URI.

    object_name: optional full key under bucket; default GCS_ARTIFACTS_PREFIX + uuid.txt
    """
    bucket_name = artifacts_bucket_name()
    if object_name:
        key = object_name.lstrip("/")
    else:
        key = f"{_prefix()}{uuid.uuid4().hex}.txt"
    client = storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    body = text.decode("utf-8") if isinstance(text, bytes) else str(text)
    blob.upload_from_string(body, content_type=content_type)
    return f"gs://{bucket_name}/{key}"

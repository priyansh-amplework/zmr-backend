"""
Read/write chunk bodies on **local disk** for legacy ``local:…`` URIs in Postgres.

Ingest scripts in this repo upload new chunk bodies to **GCS** only; they do not call
:func:`save_chunk_text_local`. Retrieval still uses :func:`load_chunk_body_from_uri` for old rows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return _ROOT


def local_chunk_base_dir() -> Path:
    rel = (
        os.getenv("LOCAL_CHUNK_BODIES_DIR", "data/chunk_bodies").strip()
        or "data/chunk_bodies"
    )
    p = Path(rel)
    base = p.resolve() if p.is_absolute() else (_ROOT / rel).resolve()
    root = _ROOT.resolve()
    try:
        base.relative_to(root)
    except ValueError as e:
        raise ValueError(
            "LOCAL_CHUNK_BODIES_DIR must resolve inside the repository root "
            f"(got {base}, root {root})"
        ) from e
    return base


def save_chunk_text_local(
    text: Union[str, bytes, bytearray],
    *,
    run_id: str,
    drive_file_id: str,
    chunk_index: int,
) -> str:
    """
    Write UTF-8 chunk file under ``LOCAL_CHUNK_BODIES_DIR`` (default ``data/chunk_bodies``).

    Returns a ``local:<relative_posix_path>`` URI stored in ``chunk_gcs_uri`` for retrieval.
    """
    sub = (
        Path("runs") / run_id / "drive" / drive_file_id / f"chunk_{chunk_index:05d}.txt"
    )
    full = local_chunk_base_dir() / sub
    full.parent.mkdir(parents=True, exist_ok=True)
    body = text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else str(text)
    full.write_text(body, encoding="utf-8")
    rel = full.relative_to(_ROOT.resolve()).as_posix()
    return f"local:{rel}"


def load_chunk_body_from_uri(uri: str) -> Optional[str]:
    """
    Load chunk text from ``local:...`` (repo-relative) or ``gs://...`` (delegates to gcs_client).
    """
    u = (uri or "").strip()
    if not u:
        return None
    if u.startswith("local:"):
        rel = u[6:].lstrip("/")
        path = (_ROOT / rel).resolve()
        try:
            path.relative_to(_ROOT.resolve())
        except ValueError:
            return None
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    if u.startswith("gs://"):
        from zmr_brain.gcs_client import download_blob_text

        return download_blob_text(u)
    return None

"""Extract text and chunk lists from local files (PDF, DOCX, plain text) for Drive RAG ingest."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# scripts/ on path for legal_doc_ingest_lib
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@dataclass
class TextChunkPiece:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _simple_windows(text: str, max_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    max_chars = max(500, max_chars)
    overlap = max(0, min(overlap, max_chars // 2))
    parts: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        parts.append(text[i:end].strip())
        if end >= n:
            break
        i = end - overlap if end - overlap > i else end
    return [p for p in parts if p]


def chunk_plaintext(
    text: str, *, max_chars: int = 3500, overlap: int = 200
) -> List[TextChunkPiece]:
    mc = int(os.getenv("DRIVE_RAG_CHUNK_CHARS", str(max_chars)))
    ov = int(os.getenv("DRIVE_RAG_CHUNK_OVERLAP", str(overlap)))
    return [
        TextChunkPiece(t, {"chunking": "char_window"})
        for t in _simple_windows(text, mc, ov)
    ]


def _use_legal_pdf_for_path(source_path: str) -> bool:
    """Legal PDF pipeline: native text, low-density → OCR, optional Claude Vision (see legal_doc_ingest_lib)."""
    if _use_legal_chunker(source_path):
        return True
    return os.getenv("DRIVE_RAG_PDF_LEGAL_PIPELINE_ALL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _use_legal_chunker(source_path: str) -> bool:
    pl = source_path.lower()
    keys = (
        "confidentiality",
        "agreement",
        "legal",
        "lease",
        "contract",
        "jll",
        "psa",
        "loan",
        # Offering memorandums: use legal PDF path (native + OCR fallback) — plain pypdf often
        # yields nothing on image-heavy / HighRes marketing PDFs.
        "offering memorandum",
        "offering memorandums",
        "rent roll",
        "trailing financial",
    )
    return any(k in pl for k in keys)


def extract_chunks_from_file(
    local_path: Path,
    *,
    source_path_for_rbac: str,
    mime_type: str,
) -> Tuple[List[TextChunkPiece], str]:
    """
    Returns (chunks, extraction_quality_label).

    **Charts / image-only PDFs:** default PDF path is pypdf (text layer only). Set
    ``DRIVE_RAG_PDF_LEGAL_PIPELINE_ALL=1`` to run **all** PDFs through the legal pipeline
    (text density → Tesseract OCR → optional Claude Vision per ``legal_doc_ingest_lib``).
    For heavy decks, ``DRIVE_RAG_PDF_FORCE_CLAUDE_VISION=1`` forces vision (costly).

    **Excel charts:** cell grid + formulas only; embedded chart *pictures* are not rasterized.
    ``DRIVE_RAG_EXCEL_WITH_LLM=1`` adds per-sheet Claude summaries (helps narrative models, not chart pixels).
    """
    local_path = local_path.resolve()
    suf = local_path.suffix.lower()

    if suf in (".txt", ".md", ".csv"):
        raw = local_path.read_text(encoding="utf-8", errors="replace")
        return chunk_plaintext(raw), "plain_text"

    if suf == ".pdf":
        if _use_legal_pdf_for_path(source_path_for_rbac):
            from legal_doc_ingest_lib import chunk_legal_path

            legal_chunks, mode = chunk_legal_path(local_path, pdf_native_only=False)
            out = [
                TextChunkPiece(c.text, {**dict(c.metadata), "legal_mode": mode})
                for c in legal_chunks
            ]
            return out, f"legal_pdf_{mode}"
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("pip install pypdf for PDF extraction") from None
        reader = PdfReader(str(local_path))
        parts = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            parts.append(t)
        raw = "\n\n".join(parts)
        return chunk_plaintext(raw), "pdf_pypdf"

    if suf in (".xlsx", ".xlsm"):
        from excel_ingest_lib import extract_workbook_chunks

        # Large models can yield 1000+ chunks; ingest uses batched Voyage embeds (VOYAGE_EMBED_BATCH_*).
        # Tighten DRIVE_RAG_EXCEL_MAX_ROWS / DRIVE_RAG_EXCEL_CHUNK_CHARS to reduce chunk count if needed.
        chunks = extract_workbook_chunks(
            local_path,
            max_rows=int(os.getenv("DRIVE_RAG_EXCEL_MAX_ROWS", "500")),
            max_cols=int(os.getenv("DRIVE_RAG_EXCEL_MAX_COLS", "60")),
            skip_hidden=True,
            with_llm=os.getenv("DRIVE_RAG_EXCEL_WITH_LLM", "").strip().lower()
            in ("1", "true", "yes"),
            chunk_target_chars=int(os.getenv("DRIVE_RAG_EXCEL_CHUNK_CHARS", "6000")),
        )
        return [
            TextChunkPiece(c.text, dict(c.metadata)) for c in chunks
        ], "excel_workbook"

    if suf == ".docx":
        if _use_legal_chunker(source_path_for_rbac):
            from legal_doc_ingest_lib import chunk_legal_path

            legal_chunks, mode = chunk_legal_path(local_path)
            return [
                TextChunkPiece(c.text, {**dict(c.metadata), "legal_mode": mode})
                for c in legal_chunks
            ], "legal_docx"
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("pip install python-docx") from None
        doc = Document(str(local_path))
        raw = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return chunk_plaintext(raw), "docx_paragraphs"

    raise ValueError(f"Unsupported extension for full RAG ingest: {suf}")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

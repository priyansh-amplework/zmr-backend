"""Conservative Voyage document embedding in batches (Drive ingest, reingest, etc.)."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, List, Optional

import voyageai


def estimate_doc_tokens(text: str) -> int:
    """
    Upper-bound-ish token estimate for **batch budgeting** (not exact).

    ``len // 4`` underestimates for spreadsheets and numeric-heavy text, which
    caused batches to exceed Voyage's per-request cap (e.g. 120k for finance).
    """
    s = text or ""
    if not s:
        return 1
    chars_per_tok = float(os.getenv("VOYAGE_EMBED_EST_CHARS_PER_TOKEN", "2.2"))
    chars_per_tok = max(1.6, min(chars_per_tok, 6.0))
    return max(1, int(len(s) / chars_per_tok))


def _api_batch_token_cap(model: str) -> int:
    """Hard per-request token ceiling we must stay under (below provider limits)."""
    m = (model or "").lower()
    if "finance" in m or "law" in m:
        return int(os.getenv("VOYAGE_EMBED_API_TOKEN_CAP_FINANCE_LAW", "118000"))
    return int(os.getenv("VOYAGE_EMBED_API_TOKEN_CAP_DEFAULT", "280000"))


def batch_token_budget(model: str, *, default_max_tokens: Optional[int] = None) -> int:
    """Target max **estimated** tokens per embed call (env, capped by model API limit)."""
    d = str(default_max_tokens) if default_max_tokens is not None else "90000"
    raw = int(os.getenv("VOYAGE_EMBED_BATCH_MAX_TOKENS", d))
    return max(2000, min(raw, _api_batch_token_cap(model)))


def embed_documents_batched(
    vclient: voyageai.Client,
    texts: List[str],
    model: str,
    *,
    input_type: str = "document",
    max_chunks_default: Optional[int] = None,
    max_tokens_default: Optional[int] = None,
) -> List[List[float]]:
    """
    Embed ``texts`` in one or more Voyage API calls.

    Uses conservative per-text token estimates and, if the API still rejects a
    batch for size, splits the batch and retries (then merges in order).
    """
    mc_env = os.getenv("VOYAGE_EMBED_BATCH_MAX_CHUNKS")
    if mc_env is not None and mc_env.strip() != "":
        max_chunks = int(mc_env)
    elif max_chunks_default is not None:
        max_chunks = int(max_chunks_default)
    else:
        max_chunks = 400
    max_chunks = max(1, min(max_chunks, 999))
    max_tokens = batch_token_budget(model, default_max_tokens=max_tokens_default)
    sleep_s = float(os.getenv("VOYAGE_EMBED_BATCH_SLEEP_SEC", "0.6"))

    def _is_batch_size_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(
            x in msg
            for x in (
                "token",
                "120000",
                "batch",
                "truncation",
                "max allowed",
                "too many",
            )
        )

    def _embed_batch_direct(batch: List[str]) -> List[List[float]]:
        resp = vclient.embed(batch, model=model, input_type=input_type)
        embs = resp.embeddings
        if len(embs) != len(batch):
            raise RuntimeError(f"Voyage embed returned {len(embs)} vectors for {len(batch)} texts")
        return embs

    def _embed_batch_recursive(batch: List[str]) -> List[List[float]]:
        if not batch:
            return []
        try:
            return _embed_batch_direct(batch)
        except BaseException as e:
            if not _is_batch_size_error(e) or len(batch) <= 1:
                if len(batch) == 1 and _is_batch_size_error(e):
                    return [_embed_oversized_single(vclient, batch[0], model, input_type, _embed_batch_direct)]
                raise
        mid = max(1, len(batch) // 2)
        return _embed_batch_recursive(batch[:mid]) + _embed_batch_recursive(batch[mid:])

    out: List[List[float]] = []
    i = 0
    n = len(texts)
    while i < n:
        batch: List[str] = []
        tok = 0
        while i < n and len(batch) < max_chunks:
            t = texts[i]
            e = estimate_doc_tokens(t)
            if batch and tok + e > max_tokens:
                break
            if not batch and e > max_tokens:
                batch.append(t)
                tok += e
                i += 1
                print(
                    f"WARN: single chunk est. {e} tokens > budget {max_tokens}; "
                    "embedding with shrink/retry if needed",
                    file=sys.stderr,
                )
                break
            batch.append(t)
            tok += e
            i += 1
        out.extend(_embed_batch_recursive(batch))
        if sleep_s > 0 and i < n:
            time.sleep(sleep_s)
    if len(out) != n:
        raise RuntimeError(f"batched embed total {len(out)} != expected {n}")
    return out


def _embed_oversized_single(
    vclient: voyageai.Client,
    text: str,
    model: str,
    input_type: str,
    embed_direct: Any,
) -> List[float]:
    """One text still rejected: shrink until the API accepts (last resort)."""
    cur = text
    floor = max(500, int(os.getenv("VOYAGE_EMBED_OVERSIZED_FLOOR_CHARS", "8000")))
    while len(cur) >= floor:
        try:
            return embed_direct([cur])[0]
        except BaseException as e:
            msg = str(e).lower()
            if "token" not in msg and "120000" not in msg and "batch" not in msg:
                raise
        cur = cur[: int(len(cur) * 3 // 4)]
    return embed_direct([cur[:floor]])[0]

"""Singleton API-client factories for Voyage, Pinecone, and Anthropic.

Every function returns a **cached** instance so TLS handshakes and object
construction happen once per process, not once per query.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any

import anthropic
import voyageai
from pinecone import Pinecone


_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_voyage_client() -> voyageai.Client:
    key = os.getenv("VOYAGE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("VOYAGE_API_KEY is not set")
    return voyageai.Client(api_key=key)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    key = os.getenv("PINECONE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=key)


_index_cache: dict[str, Any] = {}


def get_pinecone_index(index_name: str) -> Any:
    """Return a cached ``pc.Index(name)`` handle (thread-safe)."""
    if index_name in _index_cache:
        return _index_cache[index_name]
    with _lock:
        if index_name not in _index_cache:
            _index_cache[index_name] = get_pinecone_client().Index(index_name)
    return _index_cache[index_name]


@lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=key)

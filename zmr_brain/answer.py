"""Synthesize an answer with citations from retrieved chunks (Claude)."""

from __future__ import annotations

import os
import time as _time
from typing import Generator, List

import anthropic

from zmr_brain.meta_queries import chatbot_meta_reply
from zmr_brain.query_routing import OUT_OF_SCOPE_REPLY, classify_query
from zmr_brain.retrieval import (
    RetrievedChunk,
    chunks_to_context_blocks,
    chunks_with_body_text_for_llm,
)

try:
    from langsmith.run_helpers import traceable as _traceable
except ImportError:
    def _traceable(*args, **kwargs):  # noqa: ARG001
        def _decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _decorator

# Shown when retrieval returned rows but chunk_text / chunk file bodies are all empty.
NO_USABLE_CHUNK_TEXT_MESSAGE = (
    "Search matched documents, but **no readable passage text** was available for those hits "
    "(missing `chunk_text` or unreadable `chunk_gcs_uri` in Postgres—e.g. local chunk files removed, "
    "or GCS object not accessible). **Re-ingest** those documents or restore chunk storage. "
    "Answers are generated only from stored chunk text."
)


def _build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    for_llm = chunks_with_body_text_for_llm(chunks)
    context = chunks_to_context_blocks(for_llm)
    n = len(for_llm)
    return f"""You are ZMR Brain, an assistant for ZMR Capital.

Below are **{n} retrieved passages** (the full final retrieval set for this query, each with real chunk text). You must read and use **all** of them when forming your answer: combine evidence across passages where needed, and do not ignore a passage unless it is clearly irrelevant to the question.

Rules:
- Answer using **only** these passages (ZMR ingested documents). No outside general knowledge or the web.
- If the question is not addressed in these passages, say so briefly; do not invent facts.
- Cite sources by document name and path when you use a fact.

User question:
{question}

Passages [1..{n}] (chunk text):
{context}
"""


def _get_client_and_model():
    from zmr_brain.clients import get_anthropic_client

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip()
    return get_anthropic_client(), model


@_traceable(run_type="llm", name="claude_synthesize")
def answer_with_claude(question: str, chunks: List[RetrievedChunk]) -> str:
    """
    Synthesize an answer from the **full** list returned by :func:`~zmr_brain.retrieval.retrieve_for_query`
    (after hybrid fusion + rerank, length ``top_k``). Every chunk that has readable body text is
    included in the prompt—nothing is sampled or truncated here. Chunks with empty text are dropped
    so Claude never sees placeholder-only rows.
    """
    kind = classify_query(question)
    if kind == "intro":
        return chatbot_meta_reply(question)
    if kind == "refuse":
        return OUT_OF_SCOPE_REPLY

    if not chunks:
        return "No retrieved passages. Try a different question or confirm ingestion for this role index."
    for_llm = chunks_with_body_text_for_llm(chunks)
    if not for_llm:
        return NO_USABLE_CHUNK_TEXT_MESSAGE

    client, model = _get_client_and_model()
    prompt = _build_prompt(question, chunks)

    t0 = _time.monotonic()
    msg = client.messages.create(
        model=model,
        max_tokens=int(os.getenv("ANTHROPIC_ANSWER_MAX_TOKENS", "1024")),
        messages=[{"role": "user", "content": prompt}],
    )
    latency_s = _time.monotonic() - t0

    usage = getattr(msg, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    try:
        from langsmith import run_helpers as _rh

        current_run = _rh.get_current_run_tree()
        if current_run is not None:
            current_run.extra = current_run.extra or {}
            current_run.extra["metadata"] = {
                **(current_run.extra.get("metadata") or {}),
                "ls_model_name": model,
                "ls_provider": "anthropic",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_s": round(latency_s, 3),
            }
    except Exception:
        pass

    block = msg.content[0]
    if block.type != "text":
        return str(block)
    return block.text


def stream_answer_with_claude(
    question: str, chunks: List[RetrievedChunk]
) -> Generator[str, None, None]:
    """Yield answer tokens as they arrive from Claude (for Streamlit streaming).

    After the stream closes, the ``_stream_token_usage`` attribute is set on
    this generator with ``(input_tokens, output_tokens)`` from the final message.
    """
    kind = classify_query(question)
    if kind == "intro":
        yield chatbot_meta_reply(question)
        return
    if kind == "refuse":
        yield OUT_OF_SCOPE_REPLY
        return

    if not chunks:
        yield "No retrieved passages. Try a different question or confirm ingestion for this role index."
        return
    for_llm = chunks_with_body_text_for_llm(chunks)
    if not for_llm:
        yield NO_USABLE_CHUNK_TEXT_MESSAGE
        return

    client, model = _get_client_and_model()
    prompt = _build_prompt(question, chunks)
    with client.messages.stream(
        model=model,
        max_tokens=int(os.getenv("ANTHROPIC_ANSWER_MAX_TOKENS", "1024")),
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


@_traceable(run_type="chain", name="stream_synthesize")
def stream_answer_to_placeholder(
    question: str,
    chunks: List[RetrievedChunk],
    placeholder,
    chars_per_second: float = 0,
) -> str:
    """
    Stream Claude's answer into a Streamlit placeholder.

    ``chars_per_second`` caps how fast characters are painted (smooth typing effect).
    **0** (default) means no artificial delay—tokens appear as fast as the model streams.
    Override with env ``ZMR_STREAMLIT_CHARS_PER_SECOND`` (e.g. ``80`` for old behavior).
    """
    import queue
    import threading

    char_queue: queue.Queue[str | None] = queue.Queue()
    token_usage: dict = {}
    first_token_time: list = []
    t0 = _time.monotonic()

    def _producer():
        try:
            client, model = _get_client_and_model()
            prompt = _build_prompt(question, chunks)
            with client.messages.stream(
                model=model,
                max_tokens=int(os.getenv("ANTHROPIC_ANSWER_MAX_TOKENS", "1024")),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                first = True
                for text in stream.text_stream:
                    if first:
                        first_token_time.append(_time.monotonic() - t0)
                        first = False
                    for ch in text:
                        char_queue.put(ch)
                final_msg = stream.get_final_message()
                usage = getattr(final_msg, "usage", None)
                if usage:
                    token_usage["input"] = getattr(usage, "input_tokens", 0)
                    token_usage["output"] = getattr(usage, "output_tokens", 0)
                    token_usage["model"] = model
        finally:
            char_queue.put(None)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    try:
        _queue_timeout = float(os.getenv("ZMR_STREAMLIT_QUEUE_TIMEOUT_SEC", "300"))
    except ValueError:
        _queue_timeout = 300.0

    cps_env = os.getenv("ZMR_STREAMLIT_CHARS_PER_SECOND", "").strip()
    if cps_env:
        try:
            chars_per_second = float(cps_env)
        except ValueError:
            pass
    interval = (
        (1.0 / chars_per_second) if chars_per_second and chars_per_second > 0 else 0.0
    )

    full_text = ""
    cursor = " ▍"
    last_render = 0.0

    while True:
        try:
            ch = char_queue.get(timeout=_queue_timeout)
        except queue.Empty:
            if thread.is_alive():
                raise RuntimeError(
                    "Answer stream stalled: no tokens within "
                    f"{_queue_timeout:.0f}s (set ZMR_STREAMLIT_QUEUE_TIMEOUT_SEC to raise the limit)."
                ) from None
            break
        if ch is None:
            break

        full_text += ch

        now = _time.monotonic()
        elapsed = now - last_render
        if interval > 0 and elapsed < interval:
            _time.sleep(interval - elapsed)

        placeholder.markdown(full_text + cursor)
        last_render = _time.monotonic()

    placeholder.markdown(full_text)
    thread.join(timeout=120)

    try:
        from langsmith import run_helpers as _rh

        current_run = _rh.get_current_run_tree()
        if current_run is not None:
            current_run.extra = current_run.extra or {}
            meta = current_run.extra.get("metadata") or {}
            meta.update({
                "ls_model_name": token_usage.get("model", ""),
                "ls_provider": "anthropic",
                "input_tokens": token_usage.get("input", 0),
                "output_tokens": token_usage.get("output", 0),
                "total_tokens": token_usage.get("input", 0) + token_usage.get("output", 0),
                "first_token_s": round(first_token_time[0], 3) if first_token_time else None,
                "total_latency_s": round(_time.monotonic() - t0, 3),
            })
            current_run.extra["metadata"] = meta
    except Exception:
        pass

    return full_text

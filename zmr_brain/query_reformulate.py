"""
Rewrite the user question for dense + lexical retrieval (embed + BM25/FTS).

The final answer still uses the original ``query`` in :func:`zmr_brain.answer.answer_with_claude`.
"""

from __future__ import annotations

import os


def reformulate_query_for_retrieval(user_query: str) -> str:
    """
    Return a search-focused string for Pinecone + Postgres. On skip, missing key, or errors,
    returns the stripped original question.
    """
    q = (user_query or "").strip()
    if not q:
        return q

    if os.getenv("ZMR_SKIP_QUERY_REFORMULATION", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return q

    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return q

    model = (
        os.getenv("ANTHROPIC_REFORMULATE_MODEL", "").strip()
        or "claude-3-5-haiku-20241022"
    )
    max_tokens = int(os.getenv("ZMR_REFORMULATE_MAX_TOKENS", "80"))

    from zmr_brain.clients import get_anthropic_client

    client = get_anthropic_client()
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rewrite this user question into ONE short search-focused query for a corporate "
                        "real-estate and finance document index (Excel models, offering memoranda, "
                        "legal agreements, rent rolls, trailing financials).\n"
                        "Expand acronyms when helpful (e.g. OM → offering memorandum, CA → confidentiality agreement).\n"
                        "Output ONLY the rewritten query. No quotes, labels, or explanation.\n\n"
                        f"Question:\n{q}"
                    ),
                }
            ],
        )
        block = msg.content[0]
        if block.type != "text":
            return q
        line = (block.text or "").strip().split("\n")[0].strip()
        if not line:
            return q
        if len(line) > 2000:
            line = line[:2000]
        return line
    except Exception:
        return q

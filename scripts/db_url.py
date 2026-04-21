"""Shared DATABASE_URL helpers for Render / cloud Postgres."""

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def ensure_ssl_for_managed(url: str) -> str:
    """Add sslmode=require for hosts that need TLS (e.g. Render external connections)."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if not any(x in host for x in ("render.com", "supabase.co", "neon.tech")):
        return url
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    if "sslmode" not in q and "ssl" not in q:
        q["sslmode"] = "require"
        new_query = urlencode(q)
        return urlunparse((p.scheme, p.netloc, p.path, "", new_query, ""))
    return url


def apply_managed_postgres_keepalive(url: str) -> str:
    """
    Merge libpq TCP keepalive + connect_timeout for managed Postgres.

    Long-running ingest workers otherwise sit idle on DB while embedding / GCS /
    Pinecone run; hosts (e.g. Render) may close the server side of idle connections,
    causing ``connection already closed`` on the next ``commit``.
    """
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if not any(x in host for x in ("render.com", "supabase.co", "neon.tech")):
        return url
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    hints = {
        "connect_timeout": "30",
        "keepalives": "1",
        "keepalives_idle": "30",
        "keepalives_interval": "10",
        "keepalives_count": "6",
    }
    for k, v in hints.items():
        if k not in q:
            q[k] = v
    if "application_name" not in q:
        q["application_name"] = "zmr_ingest_drive_v2"
    new_query = urlencode(q)
    return urlunparse((p.scheme, p.netloc, p.path, "", new_query, ""))

# ZMR Brain — FastAPI + Streamlit on one port (nginx) for Cloud Run.
#
# Clients:
#   - Browser chat:  https://<service-url>/
#   - REST API:      https://<service-url>/v1/query/graph  (and /health, /docs, /openapi.json)
#
# Build:  docker build -t zmr-brain-chat .
# Run:    docker run --rm -p 8080:8080 --env-file .env zmr-brain-chat
#
# Cloud Run sets PORT. Wire secrets (Secret Manager → env): DATABASE_URL, VOYAGE_API_KEY,
# PINECONE_API_KEY, ANTHROPIC_API_KEY; use a runtime service account for GCS chunk reads.
#
# Suggested: --memory 2Gi --cpu 2 --timeout 300 --concurrency 2
# (Streamlit + LangGraph + Claude are memory-heavy; lower concurrency avoids OOM.)

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx gettext-base curl tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY zmr_brain/ ./zmr_brain/
COPY scripts/db_url.py ./scripts/
COPY streamlit_rbac_ui.py ./
COPY docker/nginx.conf.template docker/entrypoint.sh ./docker/

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /tmp/streamlit-cache \
    && chmod 1777 /tmp/streamlit-cache

EXPOSE 8080

# tini reaps zombies and forwards signals to the shell entrypoint.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/docker/entrypoint.sh"]

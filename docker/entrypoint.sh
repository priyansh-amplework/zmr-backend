#!/bin/sh
# PID 1 (with tini): FastAPI + Streamlit on localhost, nginx on $PORT (Cloud Run).

set -e

# Writable home for Streamlit cache when the container user has no home (e.g. Cloud Run non-root).
export HOME=/tmp/streamlit-cache

# Railway / hosts with no key file on disk: set GCP_SERVICE_ACCOUNT_JSON (full JSON) or
# GCP_SERVICE_ACCOUNT_JSON_B64 (base64 of JSON) as a secret; we write /tmp/gcp-sa.json.
if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
    if [ -n "${GCP_SERVICE_ACCOUNT_JSON_B64:-}" ]; then
        echo "entrypoint: GOOGLE_APPLICATION_CREDENTIALS from GCP_SERVICE_ACCOUNT_JSON_B64" >&2
        printf '%s' "$GCP_SERVICE_ACCOUNT_JSON_B64" | base64 -d > /tmp/gcp-sa.json
        chmod 600 /tmp/gcp-sa.json
        export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-sa.json
    elif [ -n "${GCP_SERVICE_ACCOUNT_JSON:-}" ]; then
        echo "entrypoint: GOOGLE_APPLICATION_CREDENTIALS from GCP_SERVICE_ACCOUNT_JSON" >&2
        printf '%s' "$GCP_SERVICE_ACCOUNT_JSON" > /tmp/gcp-sa.json
        chmod 600 /tmp/gcp-sa.json
        export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-sa.json
    fi
fi

PORT="${PORT:-8080}"
API_PORT=8081
ST_PORT=8501
export PORT API_PORT ST_PORT

cleanup() {
    echo "entrypoint: shutting down..."
    kill "$UV_PID" "$ST_PID" "$NGINX_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup TERM INT

/usr/local/bin/python -m uvicorn zmr_brain.api:app --host 127.0.0.1 --port "$API_PORT" &
UV_PID=$!

/usr/local/bin/streamlit run /app/streamlit_rbac_ui.py \
    --server.headless=true \
    --server.address=127.0.0.1 \
    --server.port="$ST_PORT" \
    --browser.gatherUsageStats=false \
    &
ST_PID=$!

echo "entrypoint: waiting for API (:$API_PORT) and Streamlit (:$ST_PORT)..."
i=0
while [ "$i" -lt 120 ]; do
    if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1 \
        && curl -sf "http://127.0.0.1:${ST_PORT}/" >/dev/null 2>&1; then
        echo "entrypoint: backends up, starting nginx on :${PORT}"
        break
    fi
    i=$((i + 1))
    sleep 0.5
done

if [ "$i" -ge 120 ]; then
    echo "entrypoint: timeout waiting for backends" >&2
    cleanup
    exit 1
fi

envsubst '${PORT} ${API_PORT} ${ST_PORT}' < /app/docker/nginx.conf.template > /tmp/nginx-zmr.conf
nginx -c /tmp/nginx-zmr.conf -g "daemon off;" &
NGINX_PID=$!

wait "$NGINX_PID"
cleanup

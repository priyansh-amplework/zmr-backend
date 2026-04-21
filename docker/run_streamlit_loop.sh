#!/bin/sh
# Keep Streamlit on ST_PORT across crashes/OOM (nginx stays up; without this, 502 until redeploy).

ST_PORT="${ST_PORT:-8501}"
CHILD=""

cleanup_loop() {
    if [ -n "${CHILD}" ]; then
        kill "$CHILD" 2>/dev/null || true
        wait "$CHILD" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup_loop TERM INT

while true; do
    echo "entrypoint: starting Streamlit on :${ST_PORT}..." >&2
    /usr/local/bin/streamlit run /app/streamlit_rbac_ui.py \
        --server.headless=true \
        --server.address=127.0.0.1 \
        --server.port="$ST_PORT" \
        --server.fileWatcherType none \
        --browser.gatherUsageStats=false &
    CHILD=$!
    wait "$CHILD"
    EXIT_CODE=$?
    CHILD=""
    echo "entrypoint: Streamlit exited (code ${EXIT_CODE}), restarting in 2s..." >&2
    sleep 2
done

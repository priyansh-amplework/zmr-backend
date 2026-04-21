#!/usr/bin/env bash
# HelloData ingestion pipeline (default: migrate + property reports).
# Requires .env with HELLO_DATA_API_KEY and DATABASE_URL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ -x "${ROOT}/venv/bin/python" ]]; then
  exec "${ROOT}/venv/bin/python" "${ROOT}/scripts/pipelines/run_hellodata_ingestion.py" "$@"
else
  exec python3 "${ROOT}/scripts/pipelines/run_hellodata_ingestion.py" "$@"
fi

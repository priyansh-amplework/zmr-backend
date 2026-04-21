# ZMR Brain

Role-aware **RAG** (retrieval-augmented generation) stack for ZMR Capital: Google Drive documents are chunked, embedded with **Voyage AI**, stored in **PostgreSQL** and **Pinecone**, and queried through a **FastAPI** backend with an optional **Streamlit** chat UI. Answers are synthesized with **Anthropic Claude** using only retrieved passages.

## Features

- **Hybrid retrieval**: dense vectors (Pinecone) + lexical search (BM25 / Postgres FTS) merged with **Reciprocal Rank Fusion (RRF)**; optional **Pinecone Inference reranking**.
- **RBAC by role**: each user role maps to a dedicated Pinecone index (`executive`, `acquisitions`, `asset-management`, `legal` / `compliance`, etc.).
- **Document embeddings**: paths map to **Voyage** models via `zmr_brain.embed_models` — e.g. law-oriented vs finance-oriented corpora (`VOYAGE_EMBED_MODEL_LAW`, `VOYAGE_EMBED_MODEL_FINANCE`, `VOYAGE_EMBED_MODEL_GENERAL`).
- **LangGraph pipeline** (`POST /v1/query/graph`): routing (intro / out-of-scope / document), optional query reformulation, retrieve, synthesize. **LangSmith** tracing when configured.
- **Chunk storage**: **Google Cloud Storage only** for current Drive v2 (and related) ingests — chunk bodies are **`gs://…` objects**, not under `data/chunk_bodies/`. `GCS_ARTIFACTS_BUCKET` is required at ingest time. Retrieval still supports legacy `local:` rows in older tables until you migrate them.

## Architecture

```text
Streamlit / HTTP clients
        │
        ▼
   FastAPI (zmr_brain.api)
        │
        ├── LangGraph (query_graph) ──► Voyage (query embed) + Postgres + Pinecone
        └── Claude (answer_with_claude) ← chunk text from PG / local / GCS
```

## Requirements

- **Python 3.10+** (3.11+ recommended)
- **PostgreSQL** with migrations applied (`migrations/versions/*.sql`)
- Accounts / keys: **Anthropic**, **Voyage AI**, **Pinecone**, **Google Cloud** (Drive + optional GCS), optional **LangSmith**

## Quick start

### 1. Clone and virtual environment

```bash
git clone <your-repo-url>
cd <repo-directory>
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment variables

Copy the template and edit (do **not** commit real secrets):

```bash
copy env.postgres.example .env
# or: cp env.postgres.example .env
```

Minimum typical variables:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string |
| `VOYAGE_API_KEY` | Embeddings |
| `PINECONE_API_KEY` | Vector index |
| `ANTHROPIC_API_KEY` | Answers + optional query reformulation |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON (Drive; same or other key for GCS uploads) |
| `GCS_ARTIFACTS_BUCKET` | **Required** for ingest scripts: bucket for chunk body objects (`gs://…` in `chunk_gcs_uri`). |
| `GDRIVE_FOLDER_ID` / `GDRIVE_FOLDER_IDS` | Optional override for Drive roots; **RAG v2** (`ingest_drive_full_rag_v2.py`) defaults to ZMR public/private Shared Drives when unset. |

See **`env.postgres.example`** for optional tuning (LangSmith, rerank, Excel caps, embedding model names).

### 3. Database

```bash
python scripts/bootstrap_db.py
```

### 4. Run the API

From the repository root:

```bash
python -m uvicorn zmr_brain.api:app --host 127.0.0.1 --port 8080
```

- Health: `GET http://127.0.0.1:8080/health`
- Query: `POST http://127.0.0.1:8080/v1/query`
- LangGraph query: `POST http://127.0.0.1:8080/v1/query/graph`

### 5. Run the Streamlit UI

```bash
python -m streamlit run streamlit_rbac_ui.py --server.port 8501
```

Open **http://localhost:8501**. Point the UI at the same API base URL if not default.

---

## Client testing guide: ingestion → query (step-by-step)

Run these **from the repository root** with the virtual environment activated. On Windows, use **PowerShell** for the `GCS_ARTIFACTS_BUCKET` line.

### Stage 1 — Install and configure

| Step | Command / action |
|------|------------------|
| Install dependencies | `pip install -r requirements.txt` |
| Environment file | Copy `env.postgres.example` to `.env` and set at least: `DATABASE_URL`, `VOYAGE_API_KEY`, `PINECONE_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, **`GCS_ARTIFACTS_BUCKET`**. Drive roots are optional for **v2** ingest (defaults in code); set `GDRIVE_*` to override. |
| Service account | JSON key must have **Google Drive** access to the shared folder(s) you ingest. |

### Stage 2 — PostgreSQL schema

Creates the database (if needed) and applies all SQL under `migrations/versions/`.

```bash
python scripts/bootstrap_db.py
```

Use `python scripts/bootstrap_db.py --dry-run` to print planned migrations without applying.

### Stage 3 — Pinecone indexes (one-time per environment)

Vectors are stored in role-scoped indexes (names are in `zmr_brain/constants.py`, e.g. `zmr-executive-dev`, `zmr-legal-compliance-dev`). Create or validate them in the Pinecone console, or use:

```bash
python scripts/init_pinecone.py
```

Ensure index **names** match `zmr_brain/constants.py` (`zmr-executive-dev`, `zmr-acquisitions-dev`, …, `zmr-legal-compliance-dev` for both `legal` and `compliance`). If `init_pinecone.py` lists different names in your checkout, align with the constants file or create the missing indexes in the Pinecone UI.

Embedding **dimension** must match the Voyage models you use (often **1024** for `voyage-3-large` / `voyage-law-2` / `voyage-finance-2` — confirm in Voyage docs for your chosen models).

### Stage 4 — Ingest documents (full RAG pipeline)

This is the main path: **Drive download → text extraction → chunking → Voyage document embeddings → Postgres + Pinecone**.

**Chunk bodies:** Ingest scripts **require** `GCS_ARTIFACTS_BUCKET` and write only `gs://…` URIs. Grant **Storage Object Admin** (or object read/write) on that bucket. Optional `GCS_APPLICATION_CREDENTIALS` if the Storage key differs from `GOOGLE_APPLICATION_CREDENTIALS`. Legacy `local:…` rows still work until you run `scripts/migrate_local_chunks_to_gcs.py`.

**Tiered RBAC ingest (current):** `scripts/ingest_drive_full_rag_v2.py` writes `documents_v2` / `chunks_v2` and routes vectors to per-tier Pinecone indexes.

```powershell
python scripts/ingest_drive_full_rag_v2.py --per-folder-limit 10 --max-xlsm 5
```

**Legacy v1 table ingest:** `scripts/ingest_drive_full_rag.py` (also requires GCS).

| Flag | Purpose |
|------|---------|
| `--dry-run` | List candidate files only; no download/embed. |
| `--force-reingest` | Re-process files already ingested (refreshes vectors/chunks). |
| `--max-xlsm N` | Cap **macro Excel** `.xlsm` count (slow); `0` = no cap. |
| `--per-folder-limit K` | Cap **non-.xlsm** files per subfolder (e.g. PDFs per folder). |
| `--prefix-limit "FolderName/:N"` | (v1 script where supported) Only paths under that prefix, max `N` files each. |

**Optional — lighter smoke test (metadata-only chunks, faster):**  
`python scripts/ingest_drive_batch_csp.py --drive-id <folder-id> --per-folder-limit 2`  
Use full RAG when you need real passage text for Claude.

**Migrating older `local:` rows to GCS:**  
`python scripts/migrate_local_chunks_to_gcs.py --dry-run` then run without `--dry-run` (see script help for `--tables` / `--update-pinecone`).

### Stage 5 — Verify chunks (optional)

```bash
python scripts/debug_local_chunk_coverage.py
```

Reports document/chunk counts in Postgres, local `local:` URIs vs missing files, and how many chunk body files exist on disk.

### Stage 6 — Run the API (required for Streamlit)

```bash
python -m uvicorn zmr_brain.api:app --host 127.0.0.1 --port 8080
```

### Stage 7 — Run the Streamlit UI

```bash
python -m streamlit run streamlit_rbac_ui.py --server.port 8501
```

Open **http://localhost:8501**, choose a **role** in the sidebar (maps to a Pinecone index), then ask a question. The UI calls the LangGraph path (`/v1/query/graph`) with retrieval + Claude.

### Stage 8 — Quick API checks (without the browser)

**Health**

```bash
curl -s http://127.0.0.1:8080/health
```

**Sample graph query** (replace `user_role` with a role you ingested, e.g. `executive`, `legal`):

```bash
curl -s -X POST http://127.0.0.1:8080/v1/query/graph -H "Content-Type: application/json" -d "{\"query\":\"What does the confidentiality agreement say about use of information?\",\"user_role\":\"legal\",\"top_k\":8}"
```

On **Windows PowerShell**, you can use:

`curl.exe` with the same line as above, or `Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/query/graph -Method POST -ContentType "application/json" -Body '{"query":"Hello","user_role":"executive","top_k":8}'`.

---

## Ingestion scripts (reference)

| Script | When to use it |
|--------|----------------|
| `scripts/bootstrap_db.py` | Apply Postgres migrations |
| `scripts/init_pinecone.py` | Create/validate Pinecone indexes |
| `scripts/ingest_drive_full_rag_v2.py` | **Primary** Drive RAG: v2 tables, per-tier Pinecone, **GCS-only** chunk bodies (`chunks_v2.chunk_gcs_uri`) |
| `scripts/ingest_hellodata_to_rag_v2.py` | After `ingest_hellodata.py`: push HelloData JSON snapshots into **same** v2 RAG path (Pinecone `zmr-brain-full` + GCS). Optional `--include-property-reports` |
| `scripts/ingest_drive_full_rag.py` | Legacy v1 tables + single-index style flows |
| `scripts/ingest_drive_batch_csp.py` | Quick **metadata-only** spike for RBAC / index wiring tests |
| `scripts/debug_local_chunk_coverage.py` | Health check: Postgres + local chunk files |
| `scripts/migrate_local_chunks_to_gcs.py` | Move existing `local:` chunk files into GCS and update DB (and optionally Pinecone metadata) |

**Summary:** Set **`GCS_ARTIFACTS_BUCKET`** in every environment that runs ingest; the API needs the same variable (and IAM) to load chunk bodies from `gs://` URIs. **v2 ingest does not write new chunk bodies to local disk** (unlike older tooling that used `local:` URIs).

## Repository layout

```text
zmr_brain/          # API, retrieval, LangGraph, routing, answer synthesis
scripts/            # Ingestion, DB bootstrap, utilities
migrations/versions/ # PostgreSQL migrations
streamlit_rbac_ui.py  # Chat UI
requirements.txt
env.postgres.example  # Environment template (copy to .env)
```

## Security and GitHub

- **Never commit** `.env`, API keys, or `service-account-key.json`. Use `env.postgres.example` only as a template.
- If a secret was ever committed, rotate the key and remove it from git history before pushing.
- This repository is intended for **private / internal** use unless your organization defines otherwise.

## License

Proprietary — ZMR Capital (internal use). Add an explicit `LICENSE` file if your legal team requires it.

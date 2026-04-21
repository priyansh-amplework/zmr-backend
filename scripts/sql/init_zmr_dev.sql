-- Deprecated: use migrations + scripts/bootstrap_db.py instead.
-- Source of truth: migrations/versions/001_initial_schema.sql
--
-- ZMR spike database — run in pgAdmin connected to postgres (or any DB), then create DB if needed:
-- CREATE DATABASE zmr_dev;
-- Then connect to zmr_dev and run the rest of this file.

-- Database name for pgAdmin: zmr_dev

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    drive_id TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    manifest JSONB,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    drive_file_id TEXT NOT NULL,
    name TEXT NOT NULL,
    mime_type TEXT,
    source_path TEXT NOT NULL,
    doc_type TEXT,
    source TEXT NOT NULL DEFAULT 'google_drive',
    has_mixed_sensitivity BOOLEAN NOT NULL DEFAULT false,
    access_tiers TEXT[] NOT NULL DEFAULT '{}',
    csp_profile TEXT NOT NULL DEFAULT 'csp_lite_v1',
    property_slug TEXT,
    jv TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (drive_file_id)
);

CREATE TABLE IF NOT EXISTS document_roles (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    sub_role TEXT,
    PRIMARY KEY (document_id, role)
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL DEFAULT 0,
    total_chunks INT NOT NULL DEFAULT 1,
    pinecone_index TEXT NOT NULL,
    pinecone_vector_id TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    extraction_quality TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(source_path);

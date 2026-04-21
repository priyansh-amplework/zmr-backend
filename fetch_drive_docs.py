"""
fetch_drive_docs.py
--------------------
Recursively fetches documents from the Week 1 Sample Docs Drive root (GDRIVE_FOLDER_ID).

Per-folder cap (MAX_PER_FOLDER) limits how many files are downloaded in
each individual subfolder — all subfolders are still visited.

Output structure:
  downloads_test/
  └── Sample Docs/
      ├── Models/               ← 2 files
      ├── Confidentiality Agreement/  ← 2 files
      ├── Trailing Financials/  ← 2 files
      ├── Rent Rolls/           ← 2 files
      └── Offering Memorandums/ ← 2 files
"""

import io
import os
import argparse
import json
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Config ───────────────────────────────────────────────────────────────────

SERVICE_ACCOUNT_FILE = "service-account-key.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
DOWNLOAD_DIR = Path("downloads_test")
MAX_PER_FOLDER = 2  # max files downloaded per individual subfolder

# Week 1: single root — Sample Docs shared drive folder (see zmr_rbac_mapping.md §1).
ENV_DRIVE_KEYS = ("GDRIVE_FOLDER_ID",)

# Google Workspace types that need export (not direct download)
GOOGLE_EXPORT_MAP = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
}

# ── Auth ──────────────────────────────────────────────────────────────────────


def build_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    svc = build("drive", "v3", credentials=creds)
    print(f"✅ Authenticated as: {creds.service_account_email}")
    return svc


# ── Config helpers ─────────────────────────────────────────────────────────────


def load_root_folders_from_env() -> dict:
    """Single Week 1 root from GDRIVE_FOLDER_ID (Sample Docs)."""
    load_dotenv()
    roots = {}
    for key in ENV_DRIVE_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            roots["sample-docs"] = value
            break
    if not roots:
        raise ValueError("Set GDRIVE_FOLDER_ID in .env to the Sample Docs folder ID.")
    return roots


# ── Drive helpers ─────────────────────────────────────────────────────────────


def list_folder(service, folder_id: str) -> list:
    """Return all children of a Drive folder (files + subfolders)."""
    items, page_token = [], None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType)",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                pageToken=page_token,
            )
            .execute()
        )
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def download_file(
    service, file_id: str, name: str, mime: str, dest: Path
) -> Optional[Path]:
    """Download one file to dest/. Returns saved path, or None for folders/skips."""
    dest.mkdir(parents=True, exist_ok=True)

    if mime in GOOGLE_EXPORT_MAP:
        export_mime, ext = GOOGLE_EXPORT_MAP[mime]
        path = dest / (name + ext)
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    elif mime == "application/vnd.google-apps.folder":
        return None
    else:
        path = dest / name
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    path.write_bytes(buf.getvalue())
    return path


# ── Recursive crawler ─────────────────────────────────────────────────────────


def crawl(
    service,
    folder_id: str,
    local_dir: Path,
    stats: dict,
    max_per_folder: int,
    depth: int = 0,
):
    """Visit every subfolder; download up to MAX_PER_FOLDER files per folder."""
    indent = "  " * depth
    items = list_folder(service, folder_id)

    # Separate subfolders and files so we always recurse into ALL subfolders
    subfolders = [
        i for i in items if i["mimeType"] == "application/vnd.google-apps.folder"
    ]
    files = [i for i in items if i["mimeType"] != "application/vnd.google-apps.folder"]

    # Download up to max_per_folder files in this folder
    downloaded = 0
    for f in files:
        if downloaded >= max_per_folder:
            remaining = len(files) - downloaded
            print(
                f"{indent}  ⏭️  +{remaining} more file(s) skipped (limit {max_per_folder}/folder)"
            )
            stats["skipped_due_to_cap"] += remaining
            break
        print(f"{indent}  ⬇️  {f['name']}", end=" ... ", flush=True)
        try:
            path = download_file(service, f["id"], f["name"], f["mimeType"], local_dir)
            if path:
                downloaded += 1
                stats["downloaded"] += 1
                print(f"✅  [{downloaded}/{max_per_folder}]  →  {path}")
            else:
                print("⏭️  skipped")
        except Exception as e:
            stats["errors"] += 1
            print(f"❌  {e}")

    # Always recurse into ALL subfolders
    for sf in subfolders:
        stats["folders_visited"] += 1
        print(f"{indent}📂 {sf['name']}/")
        crawl(
            service, sf["id"], local_dir / sf["name"], stats, max_per_folder, depth + 1
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download files from configured Google Drive folders."
    )
    parser.add_argument(
        "--max-per-folder",
        type=int,
        default=MAX_PER_FOLDER,
        help="Maximum files to download per folder level.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DOWNLOAD_DIR),
        help="Local output directory for downloaded files.",
    )
    args = parser.parse_args()

    root_folders = load_root_folders_from_env()
    output_dir = Path(args.output_dir)
    service = build_service()
    print(f"\n📥 Downloading into: {output_dir.resolve()}")
    output_dir.mkdir(exist_ok=True)
    run_stats = {
        "drives": len(root_folders),
        "folders_visited": 0,
        "downloaded": 0,
        "skipped_due_to_cap": 0,
        "errors": 0,
    }

    for drive_name, folder_id in root_folders.items():
        root_dir = output_dir / drive_name
        root_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"📁  {drive_name}  (ID: {folder_id})")
        print(f"{'='*60}")
        run_stats["folders_visited"] += 1
        crawl(service, folder_id, root_dir, run_stats, args.max_per_folder)

    summary_path = output_dir / "ingestion_summary.json"
    summary_path.write_text(json.dumps(run_stats, indent=2), encoding="utf-8")
    print(f"\n🧾 Summary written: {summary_path}")
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check whether domain-wide delegation is working for the service account used by main.py.

Domain-wide delegation cannot be read from the JSON file alone; you verify by
requesting OAuth tokens with .with_subject(user@domain). Success means GCP has
delegation enabled and Google Admin has authorized this SA's Client ID with
the same scopes as below.

Run from repo root (so service-account-key.json resolves like the Cloud Function):

  cd path/to/zmr-document-hub
  python src/check_domain_wide_delegation.py

Multi-mailbox (Gmail impersonation only, one line per user):

  python src/check_domain_wide_delegation.py --users a@zmrcapital.com,b@zmrcapital.com
  python src/check_domain_wide_delegation.py --users-file path/to/emails.txt
  python src/check_domain_wide_delegation.py --discover-max 10 --domain zmrcapital.com

Exit code: 0 if DWD checks pass for every tested user; 1 otherwise (or key missing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Keep in sync with main.py SCOPES — Admin must authorize this exact set for the SA Client ID.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/admin.directory.user.readonly",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEY = REPO_ROOT / "service-account-key.json"
DEFAULT_ADMIN = "zamir@zmrcapital.com"
DEFAULT_DOMAIN = "zmrcapital.com"


def resolve_key_path(explicit: str) -> Path:
    if explicit.strip():
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Key not found: {p}")
        return p.resolve()
    env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p.resolve()
    if DEFAULT_KEY.is_file():
        return DEFAULT_KEY.resolve()
    raise FileNotFoundError(
        "No key: set GOOGLE_APPLICATION_CREDENTIALS or place service-account-key.json at repo root."
    )


def print_key_meta(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        meta = json.load(f)
    print(f"Key file:     {path}")
    print(f"project_id:   {meta.get('project_id')}")
    print(f"client_email: {meta.get('client_email')}")
    print(f"client_id:    {meta.get('client_id')}  (use in Admin -> Domain-wide delegation)")
    return meta


def check_gmail_impersonated(key_path: Path, user_email: str, *, verbose: bool = True) -> bool:
    creds = service_account.Credentials.from_service_account_file(
        str(key_path), scopes=SCOPES
    ).with_subject(user_email)
    try:
        gmail = build("gmail", "v1", credentials=creds, cache_discovery=False)
        prof = gmail.users().getProfile(userId="me").execute()
        addr = prof.get("emailAddress", "?")
        ok = addr.lower() == user_email.lower()
        if verbose:
            print(f"  OK  Gmail (impersonate): profile emailAddress = {addr}")
        return ok
    except Exception as e:
        if verbose:
            print(f"  FAIL Gmail (impersonate): {e}")
        return False


def check_gmail_impersonated_one_line(key_path: Path, user_email: str) -> bool:
    creds = service_account.Credentials.from_service_account_file(
        str(key_path), scopes=SCOPES
    ).with_subject(user_email)
    try:
        gmail = build("gmail", "v1", credentials=creds, cache_discovery=False)
        prof = gmail.users().getProfile(userId="me").execute()
        addr = (prof.get("emailAddress") or "").strip()
        ok = addr.lower() == user_email.lower()
        if ok:
            print(f"  OK   {user_email}")
        else:
            print(f"  FAIL {user_email}  (profile returned {addr!r})")
        return ok
    except Exception as e:
        msg = str(e).replace("\n", " ")[:160]
        print(f"  FAIL {user_email}  ({msg})")
        return False


def check_admin_impersonated(key_path: Path, user_email: str) -> bool:
    creds = service_account.Credentials.from_service_account_file(
        str(key_path), scopes=SCOPES
    ).with_subject(user_email)
    try:
        admin = build("admin", "directory_v1", credentials=creds, cache_discovery=False)
        admin.users().get(userKey=user_email).execute()
        print(f"  OK  Admin Directory (impersonate): user lookup for {user_email}")
        return True
    except Exception as e:
        print(f"  FAIL Admin Directory (impersonate): {e}")
        return False


def check_drive_service_account(key_path: Path) -> bool:
    creds = service_account.Credentials.from_service_account_file(
        str(key_path), scopes=SCOPES
    )
    try:
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        drive.about().get(fields="user").execute()
        print("  OK  Drive (service account): about.get")
        return True
    except Exception as e:
        print(f"  FAIL Drive (service account): {e}")
        return False


def load_users_file(path: Path) -> List[str]:
    out: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def discover_workspace_emails(
    key_path: Path, admin_email: str, domain: str, max_users: int
) -> List[str]:
    creds = service_account.Credentials.from_service_account_file(
        str(key_path), scopes=SCOPES
    ).with_subject(admin_email)
    admin = build("admin", "directory_v1", credentials=creds, cache_discovery=False)
    out: List[str] = []
    page_token = None
    while len(out) < max_users:
        results = (
            admin.users()
            .list(
                domain=domain,
                maxResults=200,
                pageToken=page_token,
                query="isSuspended=false",
                fields="users(primaryEmail),nextPageToken",
            )
            .execute()
        )
        for user in results.get("users", []):
            email = (user.get("primaryEmail") or "").strip()
            if email and email not in out:
                out.append(email)
            if len(out) >= max_users:
                return out[:max_users]
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    return out


def parse_users_csv(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify domain-wide delegation for the same SA/scopes as src/main.py."
    )
    ap.add_argument(
        "--key",
        default="",
        help="Path to service account JSON (else GOOGLE_APPLICATION_CREDENTIALS or repo root key)",
    )
    ap.add_argument(
        "--impersonate",
        default="",
        help=f"Single user: Workspace user to impersonate (default: {DEFAULT_ADMIN})",
    )
    ap.add_argument(
        "--users",
        default="",
        help="Comma-separated list of Workspace emails to test (Gmail impersonation each)",
    )
    ap.add_argument(
        "--users-file",
        default="",
        help="File with one email per line (# comments allowed); tests each with Gmail impersonation",
    )
    ap.add_argument(
        "--discover-max",
        type=int,
        default=0,
        metavar="N",
        help="List up to N active users in --domain via Admin SDK (impersonating --admin), then Gmail-test each",
    )
    ap.add_argument(
        "--domain",
        default=os.getenv("WORKSPACE_DOMAIN", DEFAULT_DOMAIN).strip() or DEFAULT_DOMAIN,
        help=f"Workspace domain for --discover-max (default: {DEFAULT_DOMAIN})",
    )
    ap.add_argument(
        "--admin",
        default=os.getenv("ADMIN_EMAIL", DEFAULT_ADMIN).strip() or DEFAULT_ADMIN,
        help=f"Admin user for Directory user list when using --discover-max (default: {DEFAULT_ADMIN})",
    )
    args = ap.parse_args()

    try:
        key_path = resolve_key_path(args.key)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    modes = sum(
        1
        for x in (
            bool(args.users.strip()),
            bool(args.users_file.strip()),
            args.discover_max > 0,
        )
        if x
    )
    if modes > 1:
        print(
            "ERROR: use only one of --users, --users-file, or --discover-max",
            file=sys.stderr,
        )
        return 1

    print("=== Domain-wide delegation probe (matches src/main.py SCOPES) ===\n")
    print_key_meta(key_path)
    print()

    # ----- multi-mailbox: Gmail impersonation per user -----
    if args.discover_max > 0:
        n = args.discover_max
        print(f"Discover mode: list up to {n} active users in domain {args.domain!r} (Admin as {args.admin})")
        print("Then Gmail users.getProfile for each (proves DWD per mailbox).\n")
        try:
            users = discover_workspace_emails(
                key_path, args.admin, args.domain, n
            )
        except Exception as e:
            print(f"FAIL discover/list users: {e}")
            return 1
        if not users:
            print("No users returned (check domain, admin rights, or filters).")
            return 1
        print(f"Sampled {len(users)} user(s):\n")
        ok_drive = check_drive_service_account(key_path)
        print()
        passed = 0
        for email in users:
            if check_gmail_impersonated_one_line(key_path, email):
                passed += 1
        total = len(users)
        print()
        print(f"DWD_MULTI: {passed}/{total} mailboxes OK (Gmail impersonation)")
        if passed == total:
            print("DWD_STATUS: ON (all sampled users)")
        else:
            print("DWD_STATUS: MIXED or OFF (at least one sampled mailbox failed)")
        if ok_drive:
            print("DRIVE_SA: OK")
        else:
            print("DRIVE_SA: FAIL")
        return 0 if passed == total else 1

    if args.users_file.strip():
        uf = Path(args.users_file).expanduser()
        if not uf.is_file():
            print(f"ERROR: users-file not found: {uf}", file=sys.stderr)
            return 1
        users = load_users_file(uf)
        if not users:
            print("ERROR: users-file has no addresses", file=sys.stderr)
            return 1
        print(f"Users file: {len(users)} address(es)\n")
        ok_drive = check_drive_service_account(key_path)
        print()
        passed = sum(1 for e in users if check_gmail_impersonated_one_line(key_path, e))
        total = len(users)
        print()
        print(f"DWD_MULTI: {passed}/{total} mailboxes OK (Gmail impersonation)")
        if passed == total:
            print("DWD_STATUS: ON (all listed users)")
        else:
            print("DWD_STATUS: MIXED or OFF (at least one listed mailbox failed)")
        if ok_drive:
            print("DRIVE_SA: OK")
        else:
            print("DRIVE_SA: FAIL")
        return 0 if passed == total else 1

    if args.users.strip():
        users = parse_users_csv(args.users)
        if not users:
            print("ERROR: --users is empty", file=sys.stderr)
            return 1
        print(f"Explicit --users: {len(users)} address(es)\n")
        ok_drive = check_drive_service_account(key_path)
        print()
        passed = sum(1 for e in users if check_gmail_impersonated_one_line(key_path, e))
        total = len(users)
        print()
        print(f"DWD_MULTI: {passed}/{total} mailboxes OK (Gmail impersonation)")
        if passed == total:
            print("DWD_STATUS: ON (all listed users)")
        else:
            print("DWD_STATUS: MIXED or OFF (at least one listed mailbox failed)")
        if ok_drive:
            print("DRIVE_SA: OK")
        else:
            print("DRIVE_SA: FAIL")
        return 0 if passed == total else 1

    # ----- single user (original): Gmail + Admin + Drive -----
    user = (args.impersonate or "").strip() or DEFAULT_ADMIN
    print("DWD test = real token + API calls with .with_subject (Gmail + Admin).")
    print("Drive without impersonation only checks the SA key + Drive scope (not DWD by itself).\n")

    ok_gmail = check_gmail_impersonated(key_path, user)
    ok_admin = check_admin_impersonated(key_path, user)
    ok_drive = check_drive_service_account(key_path)
    dwd_on = ok_gmail and ok_admin

    print()
    if dwd_on:
        print("DWD_STATUS: ON")
        print("  (Gmail + Admin impersonation succeeded: delegation is working for this user.)")
    else:
        print("DWD_STATUS: OFF")
        print("  (Impersonation failed: delegation off, wrong Client ID/scopes in Admin, or bad user.)")

    if ok_drive:
        print("DRIVE_SA: OK (service account can call Drive)")
    else:
        print("DRIVE_SA: FAIL (key or Drive API / scope issue)")

    if dwd_on:
        return 0
    print("\nTypical fixes when DWD_STATUS is OFF:")
    print("  - GCP: Service account -> Enable Google Workspace domain-wide delegation")
    print("  - Admin: Domain-wide delegation -> same client_id + scopes as main.py SCOPES")
    print("  - Use --impersonate with a real Workspace user in the authorized domain")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
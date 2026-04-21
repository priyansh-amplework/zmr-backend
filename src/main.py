import functools
import builtins

builtins.print = functools.partial(print, flush=True)
"""
ZMR Capital — Central Document Hub
Google Cloud Function: Service Account with Domain-Wide Delegation

Pipeline 1: Extract attachments from ALL org Gmail accounts
Pipeline 2: Classify with Claude AI and organize into Shared Drives

Deploy: gcloud functions deploy zmr-document-hub --runtime python312 --trigger-http --timeout 540
Schedule: Cloud Scheduler runs every 15 minutes
"""

import os
import json
import base64
import logging
import time
import functions_framework
from datetime import datetime, timezone
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import anthropic
import io
from dotenv import load_dotenv
load_dotenv()
# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "SERVICE_ACCOUNT_FILE": "service-account-key.json",
    "DOMAIN": "zmrcapital.com",
    "PUBLIC_DRIVE_ID": "0APZqblunRQdpUk9PVA",
    "PRIVATE_DRIVE_ID": "0AHjq041jBwVIUk9PVA",
    "CLAUDE_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "CLAUDE_MODEL": "claude-sonnet-4-20250514",
    # Processing limits
    "MAX_MESSAGES_PER_USER": 100,  # Per run, per user
    "MAX_FILES_TO_CLASSIFY": 50,  # Per run total
    "BACKFILL_MAX_PER_USER": 200,  # For backfill mode (per run)
    # File filters
    "SKIP_EXTENSIONS": {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",  # Images (signatures/marketing)
        ".ics",
        ".vcf",  # Calendar invites, contacts
        ".htm",
        ".html",  # HTML email artifacts
        ".eml",  # Forwarded emails
        ".zip",  # Archives
    },
    "MIN_FILE_SIZE": 5000,  # Skip files under 5KB (signatures, tiny images)
    "MAX_FILE_SIZE": 100_000_000,  # Skip files over 100MB
    # Folder names
    "RAW_FOLDER": "Email Attachments",
    "ORGANIZED_FOLDER": "Organized by Deal",
    # Gmail label to mark processed messages
    "PROCESSED_LABEL": "_AttachmentsSaved",
    # Archive settings
    "ARCHIVE_DRIVE_PREFIX": "ZMR Archive",  # New drives named "ZMR Archive", "ZMR Archive 2", etc.
    "ARCHIVE_AGE_MONTHS": 12,  # Move files older than this to archive
    "DRIVE_ITEM_LIMIT": 380000,  # Create new drive when approaching 400K
    "ADMIN_EMAIL": "zamir@zmrcapital.com",
}

# Scopes needed for domain-wide delegation
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",  # Read Gmail + manage labels
    "https://www.googleapis.com/auth/drive",  # Full Drive access
    "https://www.googleapis.com/auth/admin.directory.user.readonly",  # List org users
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zmr-doc-hub")


# =============================================================================
# ASSET REGISTRY
# =============================================================================
ASSETS = {
    "standalone": [
        {"name": "Skye at Hunter's Creek", "aliases": ["Sonoma Pointe"], "jv": None},
        {"name": "Skye Ridge", "aliases": ["Sunridge"], "jv": "Atlantic Creek"},
        {"name": "Skye Isle", "aliases": ["Pecan Square"], "jv": "Atlantic Creek"},
        {"name": "Skye at Love", "aliases": ["Bayou Bend"], "jv": "Atlantic Creek"},
        {"name": "Skye at Conway", "aliases": [], "jv": None},
        {"name": "Skye Reserve", "aliases": [], "jv": None},
        {"name": "Crossing at Palm Aire", "aliases": [], "jv": None},
        {"name": "Preserve at Riverwalk", "aliases": [], "jv": None},
        {"name": "Julia", "aliases": [], "jv": None},
        {"name": "Nightingale", "aliases": [], "jv": "Atlantic Creek"},
    ],
    "portfolios": [
        {
            "name": "Walnut Portfolio",
            "subAssets": [
                {"name": "Parks at Walnut", "aliases": ["Park at Walnut"]},
                {"name": "9944", "aliases": []},
            ],
        },
        {
            "name": "Slate Portfolio",
            "subAssets": [
                {
                    "name": "Skye Oaks",
                    "aliases": ["Palms at Palisades", "Brandon Oaks"],
                },
                {"name": "Hanley Place", "aliases": []},
                {"name": "The Boardwalk", "aliases": ["Park Place"]},
                {"name": "Upland Townhomes", "aliases": []},
                {"name": "Park at Peachtree Hills", "aliases": []},
                {"name": "Flats", "aliases": []},
            ],
        },
    ],
    "disposed": [
        {"name": "Las Lomas", "aliases": []},
        {"name": "Laurel at Altamonte", "aliases": []},
        {"name": "Camila", "aliases": []},
        {"name": "Alexandria Landings", "aliases": []},
        {"name": "Desert Peaks", "aliases": []},
    ],
}

DEPARTMENTS = {
    "Acquisitions": {
        "access": "public",
        "subs": [
            "Offering Memorandums",
            "Letters of Intent",
            "Purchase & Sale Agreements",
            "Underwriting Models",
            "Market Research",
            "Broker Correspondence",
            "Deal Screening",
        ],
    },
    "Due Diligence": {
        "access": "public",
        "subs": [
            "Physical Inspections",
            "Environmental",
            "Appraisals",
            "Title & Survey",
            "Zoning & Entitlements",
            "Financial Due Diligence",
            "Insurance Review",
            "Lease Audits",
        ],
    },
    "Closing": {
        "access": "public",
        "subs": [
            "Closing Statements",
            "Title Policies",
            "Transfer Documents",
            "Post-Close Adjustments",
            "Closing Checklists",
        ],
    },
    "Capital Markets & Debt": {
        "access": "public",
        "subs": [
            "Loan Applications",
            "Term Sheets & Commitments",
            "Loan Documents",
            "Rate Lock & Hedging",
            "Loan Compliance",
            "Payoff & Defeasance",
            "Lender Correspondence",
            "Draw Requests",
        ],
    },
    "Investor Relations & Capital Raising": {
        "access": "public",
        "subs": [
            "Investor Decks & Teasers",
            "Subscription Documents",
            "Distribution Notices",
            "K-1s & Tax Documents",
            "Quarterly Reports",
            "Capital Call Notices",
            "Investor Correspondence",
            "Investor Pipeline",
        ],
    },
    "JV Partners & Co-Investors": {
        "access": "public",
        "subs": [
            "JV Operating Agreements",
            "Major Decision Approvals",
            "Partner Reports",
            "Promote & Waterfall",
            "Partner Correspondence",
            "Exit & Disposition",
        ],
    },
    "Asset Management": {
        "access": "public",
        "subs": [
            "Business Plans",
            "Annual Budgets",
            "Financial Reports",
            "Rent Rolls",
            "NOI & Valuation",
            "Operational KPIs",
            "Market Benchmarking",
            "IC & Board Materials",
            "Recovery Plans",
        ],
    },
    "3rd Party Property Management": {
        "access": "public",
        "subs": [
            "Management Agreements",
            "Monthly PM Reports",
            "Leasing Reports",
            "Maintenance & Work Orders",
            "Vendor Contracts",
            "Staffing & Personnel",
            "Compliance & Inspections",
            "Delinquency & Collections",
            "Resident Communications",
            "PM Correspondence",
        ],
    },
    "Construction & Capital Projects": {
        "access": "public",
        "subs": [
            "Scope of Work",
            "GC & Sub Contracts",
            "Bids & Proposals",
            "Construction Draws",
            "Progress Reports",
            "Permits & Approvals",
            "Budget vs. Actuals",
            "Warranties & Closeout",
            "Unit Renovation Tracking",
        ],
    },
    "Insurance & Risk Management": {
        "access": "public",
        "subs": [
            "Insurance Policies",
            "ACORD Certificates",
            "Claims",
            "Premium & Renewals",
            "Loss Runs",
            "Risk Assessments",
        ],
    },
    "Legal & Compliance": {
        "access": "public",
        "subs": [
            "Entity Formation",
            "Litigation & Disputes",
            "Regulatory Compliance",
            "Leases & Amendments",
            "Corporate Governance",
            "External Counsel",
        ],
    },
    "Accounting & Tax": {
        "access": "conditional",
        "subs": [
            "Bank Statements",
            "Invoices & AP",
            "Tax Returns",
            "Audit & Review",
            "Cost Segregation",
            "1031 Exchange",
            "Property Tax Appeals",
            "Accounts Receivable",
        ],
    },
    "Marketing & Leasing": {
        "access": "public",
        "subs": [
            "Marketing Materials",
            "Photography & Media",
            "Website & ILS",
            "Market Surveys",
            "Signage & Branding",
        ],
    },
    "Corporate Operations (OpCo)": {
        "access": "private",
        "subs": [
            "OpCo Contracts",
            "Vendor & Software Agreements",
            "Corporate Entity Documents",
            "Banking & Treasury",
            "Office Lease & Facilities",
            "Corporate Insurance",
            "Legal Retainers & Outside Counsel",
            "Licenses & Registrations",
        ],
    },
    "Human Resources": {
        "access": "private",
        "subs": [
            "Employment Agreements",
            "Benefits & Payroll",
            "Company Policies & Handbook",
            "Performance Reviews",
            "Recruiting & Onboarding",
        ],
    },
    "Dispositions & Exits": {
        "access": "public",
        "subs": [
            "Sale Documents",
            "Final Financials",
            "Investor Closeout",
            "Tax & 1031",
            "Historical Records",
            "Loan Payoff",
        ],
    },
}


# =============================================================================
# AUTHENTICATION HELPERS
# =============================================================================
def get_credentials(impersonate_user=None):
    """Get service account credentials, optionally impersonating a user."""
    creds = service_account.Credentials.from_service_account_file(
        CONFIG["SERVICE_ACCOUNT_FILE"], scopes=SCOPES
    )
    if impersonate_user:
        creds = creds.with_subject(impersonate_user)
    return creds


def get_gmail_service(user_email):
    """Get Gmail API service impersonating a specific user."""
    creds = get_credentials(impersonate_user=user_email)
    return build("gmail", "v1", credentials=creds)


def get_drive_service():
    """Get Drive API service (uses service account directly, not impersonation)."""
    creds = get_credentials()
    return build("drive", "v3", credentials=creds)


def get_admin_service():
    """Get Admin SDK service to list org users."""
    # Impersonate an admin user to list directory users
    admin_email = "zamir@zmrcapital.com"
    creds = get_credentials(impersonate_user=admin_email)
    return build("admin", "directory_v1", credentials=creds)


# =============================================================================
# LIST ALL ORG USERS
# =============================================================================
def get_all_users():
    """Get all active users in the Google Workspace domain."""
    service = get_admin_service()
    users = []
    page_token = None

    while True:
        results = (
            service.users()
            .list(
                domain=CONFIG["DOMAIN"],
                maxResults=200,
                pageToken=page_token,
                query="isSuspended=false",
                fields="users(primaryEmail,name),nextPageToken",
            )
            .execute()
        )

        for user in results.get("users", []):
            users.append(user["primaryEmail"])

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    print(f"Found {len(users)} active users in {CONFIG['DOMAIN']}")
    return users


# =============================================================================
# PIPELINE 1: EXTRACT ATTACHMENTS
# =============================================================================
def get_or_create_label(gmail_service, user_email, label_name):
    """Get or create a Gmail label for tracking processed messages."""
    labels = gmail_service.users().labels().list(userId="me").execute()
    for label in labels.get("labels", []):
        if label["name"] == label_name:
            return label["id"]

    # Create label
    label = (
        gmail_service.users()
        .labels()
        .create(
            userId="me",
            body={
                "name": label_name,
                "labelListVisibility": "labelHide",
                "messageListVisibility": "hide",
            },
        )
        .execute()
    )
    return label["id"]


def get_or_create_folder(drive_service, parent_id, folder_name):
    """Get or create a folder in Drive."""
    safe_name = folder_name.replace("'", "\\'")
    query = (
        f"name = '{safe_name}' and '{parent_id}' in parents "
        f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    results = (
        drive_service.files()
        .list(
            q=query,
            spaces="drive",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            fields="files(id, name)",
        )
        .execute()
    )

    files = results.get("files", [])
    if files:
        return files[0]["id"]

    # Create folder
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = (
        drive_service.files()
        .create(
            body=metadata,
            supportsAllDrives=True,
            fields="id",
        )
        .execute()
    )
    return folder["id"]


def should_skip_file(filename, size):
    """Check if file should be skipped based on extension and size."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in CONFIG["SKIP_EXTENSIONS"]:
        return True
    if size < CONFIG["MIN_FILE_SIZE"]:
        return True
    if size > CONFIG["MAX_FILE_SIZE"]:
        return True
    return False


def extract_attachments_for_user(
    user_email, drive_service, max_messages=None, backfill=False
):
    """Extract all attachments from a user's Gmail."""
    max_messages = max_messages or (
        CONFIG["BACKFILL_MAX_PER_USER"] if backfill else CONFIG["MAX_MESSAGES_PER_USER"]
    )

    try:
        gmail = get_gmail_service(user_email)
        # Quick test - if this fails, it's not accessible
        gmail.users().getProfile(userId="me").execute()
    except Exception as e:
        print(f"Cannot access Gmail for {user_email}: {e}")
        return 0

    label_id = get_or_create_label(gmail, user_email, CONFIG["PROCESSED_LABEL"])

    # Search for messages with attachments that haven't been processed
    query = f"has:attachment -label:{CONFIG['PROCESSED_LABEL']}"
    saved_count = 0
    page_token = None
    messages_processed = 0

    while messages_processed < max_messages:
        try:
            results = (
                gmail.users()
                .messages()
                .list(
                    userId="me",
                    q=query,
                    maxResults=min(50, max_messages - messages_processed),
                    pageToken=page_token,
                )
                .execute()
            )
        except Exception as e:
            print(f"Error listing messages for {user_email}: {e}")
            break

        messages = results.get("messages", [])
        if not messages:
            break

        for msg_ref in messages:
            try:
                msg = (
                    gmail.users()
                    .messages()
                    .get(
                        userId="me",
                        id=msg_ref["id"],
                        format="full",
                    )
                    .execute()
                )

                saved = process_gmail_message(gmail, msg, user_email, drive_service)
                saved_count += saved

                # Add processed label
                gmail.users().messages().modify(
                    userId="me",
                    id=msg_ref["id"],
                    body={"addLabelIds": [label_id]},
                ).execute()

                messages_processed += 1

            except Exception as e:
                print(f"Error processing message {msg_ref['id']} for {user_email}: {e}")
                messages_processed += 1

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    if saved_count > 0:
        print(
            f"  {user_email}: saved {saved_count} attachments from {messages_processed} messages"
        )

    return saved_count


def process_gmail_message(gmail, msg, user_email, drive_service):
    """Process a single Gmail message and save its attachments."""
    headers = {
        h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])
    }
    sender = headers.get("from", "Unknown")
    subject = headers.get("subject", "No Subject")
    date_str = headers.get("date", "")

    # Parse date
    try:
        # Try to get internal date from Gmail
        internal_date = int(msg.get("internalDate", 0)) / 1000
        dt = datetime.fromtimestamp(internal_date, tz=timezone.utc)
        date_prefix = dt.strftime("%Y-%m-%d")
        month_folder = dt.strftime("%Y-%m")
    except:
        date_prefix = "unknown-date"
        month_folder = "unknown"

    # Extract sender name
    sender_name = extract_sender_name(sender)
    sender_email = extract_sender_email(sender)

    # Get or create folder structure: Raw Root / Month / Sender
    raw_root_id = get_or_create_folder(
        drive_service, CONFIG["PUBLIC_DRIVE_ID"], CONFIG["RAW_FOLDER"]
    )
    month_id = get_or_create_folder(drive_service, raw_root_id, month_folder)
    sender_folder_name = f"{sender_name} ({sender_email})"
    sender_id = get_or_create_folder(drive_service, month_id, sender_folder_name)

    # Find and save attachments
    saved = 0
    parts = get_all_parts(msg.get("payload", {}))

    for part in parts:
        filename = part.get("filename", "")
        if not filename:
            continue

        body = part.get("body", {})
        size = body.get("size", 0)

        if should_skip_file(filename, size):
            continue

        attachment_id = body.get("attachmentId")
        if not attachment_id:
            continue

        try:
            # Download attachment
            att = (
                gmail.users()
                .messages()
                .attachments()
                .get(
                    userId="me",
                    messageId=msg["id"],
                    id=attachment_id,
                )
                .execute()
            )

            file_data = base64.urlsafe_b64decode(att["data"])
            base_name, ext = os.path.splitext(filename)
            saved_name = f"{date_prefix} - {filename}"

            # Check for duplicates — if same filename exists on same day from same sender,
            # append a counter (e.g., "2026-02-15 - Rent Roll (2).xlsx")
            # Duplicate check - fail-safe: if query errors due to special chars, just upload
            try:
                safe_base = base_name.replace("'", "\\'").replace("&", " ")
                safe_prefix = date_prefix.replace("'", "\\'")
                dup_query = f"name contains '{safe_prefix} - {safe_base}' and '{sender_id}' in parents and trashed = false"
                existing = (
                    drive_service.files()
                    .list(
                        q=dup_query,
                        spaces="drive",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                        fields="files(id, name, md5Checksum)",
                    )
                    .execute()
                )
                existing_files = existing.get("files", [])
                if existing_files:
                    name_matches = [
                        f for f in existing_files if f["name"] == saved_name
                    ]
                    if name_matches:
                        counter = len(existing_files) + 1
                        saved_name = f"{date_prefix} - {base_name} ({counter}){ext}"
            except Exception as dup_err:
                print(
                    f"  Duplicate check failed for {filename}, uploading anyway: {dup_err}"
                )

            # Upload to Drive. ``description`` must stay JSON-parseable; keys are the contract
            # for Engineer 2 ingestion (email ↔ attachment linking). See ``zmr_brain.metadata_schema``
            # (DRIVE_EMAIL_ATTACHMENT_DESCRIPTION_KEYS) in the document-hub repo.
            metadata = {
                "name": saved_name,
                "parents": [sender_id],
                "description": json.dumps(
                    {
                        "from": sender,
                        "to": headers.get("to", ""),
                        "subject": subject,
                        "date": date_prefix,
                        "originalFilename": filename,
                        "extractedBy": user_email,
                        "classified": False,
                    }
                ),
            }

            media = MediaIoBaseUpload(
                io.BytesIO(file_data),
                mimetype=part.get("mimeType", "application/octet-stream"),
            )

            drive_service.files().create(
                body=metadata,
                media_body=media,
                supportsAllDrives=True,
                fields="id",
            ).execute()

            saved += 1

        except Exception as e:
            print(f"Error saving attachment {filename}: {e}")

    return saved


def get_all_parts(payload):
    """Recursively get all parts from a Gmail message payload."""
    parts = []
    if "parts" in payload:
        for part in payload["parts"]:
            parts.extend(get_all_parts(part))
    if payload.get("filename"):
        parts.append(payload)
    return parts


def extract_sender_name(from_field):
    """Extract display name from email From field."""
    import re

    match = re.match(r'^"?(.+?)"?\s*<', from_field)
    if match:
        return match.group(1).strip().replace('"', "")
    return extract_sender_email(from_field).split("@")[0]


def extract_sender_email(from_field):
    """Extract email address from From field."""
    import re

    match = re.search(r"<(.+?)>", from_field)
    return match.group(1) if match else from_field.strip()


# =============================================================================
# PIPELINE 2: AI CLASSIFICATION
# =============================================================================
def build_classification_prompt():
    """Build the Claude system prompt with full taxonomy."""
    asset_text = "STANDALONE ASSETS:\n"
    for a in ASSETS["standalone"]:
        asset_text += f"- {a['name']}"
        if a["aliases"]:
            asset_text += f" (aliases: {', '.join(a['aliases'])})"
        if a["jv"]:
            asset_text += f" [JV: {a['jv']}]"
        asset_text += "\n"

    asset_text += "\nPORTFOLIOS:\n"
    for p in ASSETS["portfolios"]:
        asset_text += f"- {p['name']}:\n"
        for s in p["subAssets"]:
            asset_text += f"    - {s['name']}"
            if s["aliases"]:
                asset_text += f" (aliases: {', '.join(s['aliases'])})"
            asset_text += "\n"

    asset_text += "\nDISPOSED ASSETS (route to Dispositions & Exits):\n"
    for a in ASSETS["disposed"]:
        asset_text += f"- {a['name']}\n"

    dept_text = ""
    for name, info in DEPARTMENTS.items():
        dept_text += f"\n{name} [{info['access'].upper()}]:\n"
        for s in info["subs"]:
            dept_text += f"  - {s}\n"

    return f"""You are a document classifier for ZMR Capital, an institutional multifamily real estate investment firm.

Given a file's metadata, classify it into the correct asset, department, and subcategory.

{asset_text}

JV PARTNERSHIP: Atlantic Creek is the JV equity partner on Skye Isle, Skye at Love, Nightingale, and Skye Ridge.

DEPARTMENTS:{dept_text}

ACCESS RULES:
- PUBLIC departments → team Shared Drive
- PRIVATE departments (HR, Corporate Operations) → confidential Shared Drive
- CONDITIONAL (Accounting & Tax) → asset-level = PUBLIC, OpCo/personal-level = PRIVATE

RULES:
1. Match file to ASSET using filename, sender, subject. Use legacy aliases.
2. Portfolio sub-assets: property-specific → Portfolio/Sub-Asset/Dept/Sub; portfolio-wide → Portfolio/Dept/Sub
3. Disposed asset docs → Dispositions & Exits department
4. No asset match → "Corporate"
5. Atlantic Creek docs → relevant asset's JV Partners department

Respond ONLY with valid JSON:
{{"asset":"name","portfolio":"name or null","sub_asset":"name or null","department":"dept","subcategory":"sub","access":"public or private","confidence":"high/medium/low","reasoning":"one sentence"}}"""


def classify_file(metadata):
    """Send file metadata to Claude for classification."""
    client = anthropic.Anthropic(api_key=CONFIG["CLAUDE_API_KEY"])

    user_msg = f"""Classify this document:
Filename: {metadata.get('originalFilename', 'Unknown')}
From: {metadata.get('from', 'Unknown')}
Subject: {metadata.get('subject', 'Unknown')}
Date: {metadata.get('date', 'Unknown')}"""

    try:
        response = client.messages.create(
            model=CONFIG["CLAUDE_MODEL"],
            max_tokens=500,
            system=build_classification_prompt(),
            messages=[{"role": "user", "content": user_msg}],
        )

        text = response.content[0].text.strip()
        clean = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)

    except Exception as e:
        print(f"Claude classification error: {e}")
        return None


def classify_unprocessed_files(drive_service):
    """Find and classify unprocessed files in the raw attachments folder."""
    raw_root_id = get_or_create_folder(
        drive_service, CONFIG["PUBLIC_DRIVE_ID"], CONFIG["RAW_FOLDER"]
    )

    # Search for files with classified=false in description
    # We search recursively through the raw folder
    unclassified = find_unclassified_files(
        drive_service, raw_root_id, CONFIG["MAX_FILES_TO_CLASSIFY"]
    )

    if not unclassified:
        print("Pipeline 2: No unclassified files found.")
        return 0

    print(f"Pipeline 2: Classifying {len(unclassified)} files...")
    classified_count = 0

    for file_info in unclassified:
        file_id = file_info["id"]
        metadata = file_info["metadata"]

        classification = classify_file(metadata)
        if not classification:
            continue

        try:
            # Create organized shortcut/copy
            create_organized_file(
                drive_service, file_id, file_info["name"], classification
            )

            # Mark as classified
            metadata["classified"] = True
            metadata["classifiedAt"] = datetime.now(timezone.utc).isoformat()
            metadata["classification"] = classification
            drive_service.files().update(
                fileId=file_id,
                body={"description": json.dumps(metadata)},
                supportsAllDrives=True,
            ).execute()

            classified_count += 1
            print(
                f"  Classified: {file_info['name']} → {classification['asset']} / {classification['department']} / {classification['subcategory']} [{classification['access']}]"
            )

        except Exception as e:
            print(f"Error organizing {file_info['name']}: {e}")

    return classified_count


def find_unclassified_files(drive_service, folder_id, max_files):
    """Recursively find unclassified files in the raw folder."""
    results = []

    # List subfolders
    query = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    folders = (
        drive_service.files()
        .list(
            q=query,
            spaces="drive",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            fields="files(id)",
        )
        .execute()
        .get("files", [])
    )

    for folder in folders:
        if len(results) >= max_files:
            break
        results.extend(
            find_unclassified_files(
                drive_service, folder["id"], max_files - len(results)
            )
        )

    # List files in this folder
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    files = (
        drive_service.files()
        .list(
            q=query,
            spaces="drive",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            fields="files(id, name, description)",
        )
        .execute()
        .get("files", [])
    )

    for f in files:
        if len(results) >= max_files:
            break
        try:
            desc = f.get("description", "")
            if desc:
                metadata = json.loads(desc)
                if not metadata.get("classified"):
                    results.append(
                        {"id": f["id"], "name": f["name"], "metadata": metadata}
                    )
        except (json.JSONDecodeError, KeyError):
            pass

    return results


def create_organized_file(drive_service, source_file_id, filename, classification):
    """Move the original file to the organized folder structure."""
    # Determine target drive
    if classification["access"] == "private":
        target_drive_id = CONFIG["PRIVATE_DRIVE_ID"]
    else:
        target_drive_id = CONFIG["PUBLIC_DRIVE_ID"]

    # Build folder path
    organized_root = get_or_create_folder(
        drive_service, target_drive_id, CONFIG["ORGANIZED_FOLDER"]
    )

    if classification.get("portfolio") and classification.get("sub_asset"):
        portfolio_folder = get_or_create_folder(
            drive_service, organized_root, classification["portfolio"]
        )
        asset_folder = get_or_create_folder(
            drive_service, portfolio_folder, classification["sub_asset"]
        )
    elif classification.get("portfolio"):
        asset_folder = get_or_create_folder(
            drive_service, organized_root, classification["portfolio"]
        )
    else:
        asset_folder = get_or_create_folder(
            drive_service, organized_root, classification["asset"]
        )

    dept_folder = get_or_create_folder(
        drive_service, asset_folder, classification["department"]
    )
    sub_folder = get_or_create_folder(
        drive_service, dept_folder, classification["subcategory"]
    )

    # Move the original file to the organized folder
    file_info = (
        drive_service.files()
        .get(
            fileId=source_file_id,
            fields="parents",
            supportsAllDrives=True,
        )
        .execute()
    )
    previous_parents = ",".join(file_info.get("parents", []))

    drive_service.files().update(
        fileId=source_file_id,
        addParents=sub_folder,
        removeParents=previous_parents,
        supportsAllDrives=True,
        fields="id",
    ).execute()


# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================
@functions_framework.http
def process_all(request):
    """Main function: extract attachments from users + classify. Processes in batches to avoid timeout."""
    import time as _time

    start_time = _time.time()
    MAX_RUNTIME = 420  # Stop after 8 minutes (buffer before 540s timeout)

    print("=== ZMR Document Hub: Starting run ===")

    drive_service = get_drive_service()
    users = get_all_users()
    total_saved = 0
    users_processed = 0

    # Pipeline 1: Extract attachments, but watch the clock
    for user_email in users:
        elapsed = _time.time() - start_time
        if elapsed > MAX_RUNTIME:
            print(
                f"Time limit reached after {users_processed} users. Remaining users will be processed next run."
            )
            break
        try:
            saved = extract_attachments_for_user(
                user_email, drive_service, max_messages=20
            )
            total_saved += saved
            users_processed += 1
        except Exception as e:
            print(f"Error processing user {user_email}: {e}")
            users_processed += 1

    print(
        f"Pipeline 1: {total_saved} attachments saved from {users_processed}/{len(users)} users"
    )

    # Pipeline 2: Classify if we have time left
    classified = 0
    elapsed = _time.time() - start_time
    if elapsed < MAX_RUNTIME:
        classified = classify_unprocessed_files(drive_service)
        print(f"Pipeline 2: {classified} files classified")
    else:
        print("Pipeline 2: Skipped (no time remaining)")

    result = {
        "saved": total_saved,
        "classified": classified,
        "users_processed": users_processed,
        "total_users": len(users),
    }
    print(f"=== Run complete: {json.dumps(result)} ===")

    return json.dumps(result), 200


@functions_framework.http
def backfill_all(request):
    """Backfill: process historical emails. Runs in batches, call repeatedly until done."""
    import time as _time

    start_time = _time.time()
    MAX_RUNTIME = 420

    print("=== ZMR Document Hub: BACKFILL MODE ===")

    drive_service = get_drive_service()
    users = get_all_users()
    total_saved = 0
    users_processed = 0

    for user_email in users:
        elapsed = _time.time() - start_time
        if elapsed > MAX_RUNTIME:
            print(
                f"Time limit reached after {users_processed} users. Run backfill again for remaining users."
            )
            break
        print(f"Backfilling {user_email}...")
        try:
            saved = extract_attachments_for_user(
                user_email, drive_service, backfill=True
            )
            total_saved += saved
            users_processed += 1
        except Exception as e:
            print(f"Error backfilling {user_email}: {e}")
            users_processed += 1

    print(
        f"Backfill: {total_saved} attachments from {users_processed}/{len(users)} users"
    )

    classified = 0
    elapsed = _time.time() - start_time
    if elapsed < MAX_RUNTIME:
        classified = classify_unprocessed_files(drive_service)

    result = {
        "saved": total_saved,
        "classified": classified,
        "users_processed": users_processed,
        "total_users": len(users),
    }
    print(f"=== Backfill batch complete: {json.dumps(result)} ===")

    return json.dumps(result), 200


@functions_framework.http
def cleanup_duplicates(request):
    """One-time cleanup: replace shortcuts with moved originals, delete empty folders."""
    import time as _time

    start_time = _time.time()
    MAX_RUNTIME = 420

    print("=== Cleanup: Starting (v13 - empty folder cleanup) ===")
    drive_service = get_drive_service()
    admin_creds = get_credentials(impersonate_user=CONFIG["ADMIN_EMAIL"])
    admin_drive = build("drive", "v3", credentials=admin_creds)

    # Phase 1: Delete any remaining shortcuts
    query = "mimeType = 'application/vnd.google-apps.shortcut' and trashed = false"
    r = (
        drive_service.files()
        .list(
            q=query,
            corpora="drive",
            driveId=CONFIG["PUBLIC_DRIVE_ID"],
            spaces="drive",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            fields="files(id)",
            pageSize=1000,
        )
        .execute()
    )
    remaining_shortcuts = r.get("files", [])
    if remaining_shortcuts:
        print(f"Deleting {len(remaining_shortcuts)} remaining shortcuts...")
        batch = admin_drive.new_batch_http_request()
        for s in remaining_shortcuts:
            batch.add(
                admin_drive.files().delete(fileId=s["id"], supportsAllDrives=True)
            )
        try:
            batch.execute()
            print(f"Deleted {len(remaining_shortcuts)} shortcuts")
        except Exception as e:
            print(f"Shortcut batch delete error: {e}")

    # Phase 2: Find and delete empty folders
    # Strategy: list all folders, check each for children, batch delete empty ones
    # Repeat passes until no more empty folders (deleting may create new empty parents)
    total_deleted = 0
    errors = 0
    timed_out = False
    pass_num = 0

    while True:
        pass_num += 1
        if _time.time() - start_time > MAX_RUNTIME:
            timed_out = True
            break

        # Get a batch of folders
        query = "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        page_token = None
        pass_deleted = 0
        folders_checked = 0

        while True:
            if _time.time() - start_time > MAX_RUNTIME:
                timed_out = True
                break

            results = (
                drive_service.files()
                .list(
                    q=query,
                    corpora="drive",
                    driveId=CONFIG["PUBLIC_DRIVE_ID"],
                    spaces="drive",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    fields="files(id, name), nextPageToken",
                    pageSize=500,
                    pageToken=page_token,
                )
                .execute()
            )

            folders = results.get("files", [])
            if not folders:
                break

            # Check each folder for children and collect empty ones
            empty_ids = []
            for folder in folders:
                if _time.time() - start_time > MAX_RUNTIME:
                    timed_out = True
                    break
                folders_checked += 1
                children = (
                    drive_service.files()
                    .list(
                        q=f"'{folder['id']}' in parents and trashed = false",
                        spaces="drive",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                        fields="files(id)",
                        pageSize=1,
                    )
                    .execute()
                )
                if not children.get("files"):
                    empty_ids.append(folder["id"])

            # Batch delete empty folders
            for i in range(0, len(empty_ids), 100):
                if _time.time() - start_time > MAX_RUNTIME:
                    timed_out = True
                    break
                batch = admin_drive.new_batch_http_request()
                chunk = empty_ids[i : i + 100]
                batch_results = {"ok": 0, "err": 0}

                def make_callback(br):
                    def cb(request_id, response, exception):
                        if exception:
                            br["err"] += 1
                        else:
                            br["ok"] += 1

                    return cb

                callback = make_callback(batch_results)
                for fid in chunk:
                    batch.add(
                        admin_drive.files().delete(fileId=fid, supportsAllDrives=True),
                        callback=callback,
                    )

                try:
                    batch.execute()
                except Exception as batch_err:
                    print(f"Batch error: {batch_err}")

                pass_deleted += batch_results["ok"]
                errors += batch_results["err"]

            if timed_out:
                break
            page_token = results.get("nextPageToken")
            if not page_token:
                break

        total_deleted += pass_deleted
        elapsed = int(_time.time() - start_time)
        print(
            f"Pass {pass_num}: checked {folders_checked} folders, deleted {pass_deleted} empty ({elapsed}s)"
        )

        # If no empty folders found this pass, we're done
        if pass_deleted == 0 or timed_out:
            break

    result = {
        "folders_deleted": total_deleted,
        "errors": errors,
        "passes": pass_num,
        "timed_out": timed_out,
    }
    print(f"=== Cleanup complete: {json.dumps(result)} ===")

    # Auto-pause when done
    if total_deleted == 0 and not timed_out:
        try:
            from google.oauth2 import service_account as sa
            from google.auth.transport.requests import Request as AuthRequest
            import urllib.request

            creds = sa.Credentials.from_service_account_file(
                CONFIG["SERVICE_ACCOUNT_FILE"],
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            creds.refresh(AuthRequest())
            req = urllib.request.Request(
                "https://cloudscheduler.googleapis.com/v1/projects/zmr-document-hub/locations/us-central1/jobs/zmr-cleanup-scheduler:pause",
                data=b"{}",
                headers={
                    "Authorization": f"Bearer {creds.token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            urllib.request.urlopen(req)
            print("=== Folder cleanup done! Auto-paused zmr-cleanup-scheduler ===")
        except Exception as e:
            print(f"Failed to auto-pause scheduler: {e}")

    return json.dumps(result), 200


# =============================================================================
# ARCHIVE OLD FILES
# =============================================================================
def count_drive_items(drive_service, drive_id):
    """Count total non-trashed items in a Shared Drive (files + folders)."""
    total = 0
    page_token = None
    while True:
        results = (
            drive_service.files()
            .list(
                q="trashed = false",
                corpora="drive",
                driveId=drive_id,
                spaces="drive",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                fields="files(id), nextPageToken",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute()
        )
        total += len(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    return total


def get_or_create_archive_drive(admin_drive):
    """Find the current archive drive or create a new one if approaching limit."""
    import uuid

    prefix = CONFIG["ARCHIVE_DRIVE_PREFIX"]
    limit = CONFIG["DRIVE_ITEM_LIMIT"]

    # List all drives matching our archive prefix
    drives = admin_drive.drives().list(pageSize=100).execute().get("drives", [])
    archive_drives = sorted(
        [d for d in drives if d["name"].startswith(prefix)],
        key=lambda d: d["name"],
    )

    if archive_drives:
        # Use the latest archive drive
        current = archive_drives[-1]
        # Check if it has room (quick estimate — count items)
        count = count_drive_items(admin_drive, current["id"])
        print(f"Archive drive '{current['name']}' has {count} items")
        if count < limit:
            return current["id"], current["name"]

        # Current archive is full, create next one
        num = len(archive_drives) + 1
        new_name = f"{prefix} {num}"
    else:
        # No archive drive exists, use the one we already created or create new
        new_name = prefix

    # Create new archive drive
    result = (
        admin_drive.drives()
        .create(
            requestId=str(uuid.uuid4()),
            body={"name": new_name},
        )
        .execute()
    )
    print(f"Created new archive drive: {new_name} (id={result['id']})")
    return result["id"], new_name


def get_or_create_folder_in_drive(drive_service, drive_id, parent_id, folder_name):
    """Get or create a folder in a specific drive, under a specific parent."""
    safe_name = folder_name.replace("'", "\\'")
    query = f"name = '{safe_name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = (
        drive_service.files()
        .list(
            q=query,
            spaces="drive",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            fields="files(id)",
            pageSize=1,
        )
        .execute()
    )
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    # Create the folder
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = (
        drive_service.files()
        .create(
            body=metadata,
            supportsAllDrives=True,
            fields="id",
        )
        .execute()
    )
    return folder["id"]


@functions_framework.http
def archive_old_files(request):
    """Move files older than ARCHIVE_AGE_MONTHS to archive drive.

    Uses filename date prefix (YYYY-MM-DD) since modifiedTime reflects upload date.
    Files are organized in archive by year-month folders.
    """
    import time as _time
    import re as _re
    from datetime import datetime, timedelta

    start_time = _time.time()
    MAX_RUNTIME = 420

    print("=== Archive: Starting ===")

    cutoff = datetime.utcnow() - timedelta(days=CONFIG["ARCHIVE_AGE_MONTHS"] * 30)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    print(f"Archiving files with name date before {cutoff_str}")

    drive_service = get_drive_service()
    admin_creds = get_credentials(impersonate_user=CONFIG["ADMIN_EMAIL"])
    admin_drive = build("drive", "v3", credentials=admin_creds)

    archive_drive_id, archive_name = get_or_create_archive_drive(admin_drive)

    # Query ALL non-folder, non-shortcut files (we filter by filename date in code)
    query = "mimeType != 'application/vnd.google-apps.folder' and mimeType != 'application/vnd.google-apps.shortcut' and trashed = false"
    date_pattern = _re.compile(r"^(\d{4}-\d{2}-\d{2})\s*-\s*")
    archived = 0
    skipped = 0
    errors = 0
    page_token = None
    folder_cache = {}  # year-month -> folder_id in archive drive

    while True:
        if _time.time() - start_time > MAX_RUNTIME:
            print(f"Time limit reached. Archived so far: {archived}")
            break

        results = (
            drive_service.files()
            .list(
                q=query,
                spaces="drive",
                corpora="drive",
                driveId=CONFIG["PUBLIC_DRIVE_ID"],
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                fields="files(id, name, parents), nextPageToken",
                pageSize=500,
                pageToken=page_token,
            )
            .execute()
        )

        files = results.get("files", [])
        if not files:
            print("No more files in drive.")
            break

        # Filter to files with old dates in their name
        to_archive = []
        for f in files:
            m = date_pattern.match(f.get("name", ""))
            if m:
                file_date = m.group(1)
                if file_date < cutoff_str:
                    to_archive.append((f, file_date[:7]))  # (file, "YYYY-MM")
            # Files without date prefix are skipped (too recent or not from pipeline)

        skipped += len(files) - len(to_archive)

        if to_archive:
            print(f"Page: {len(files)} files, {len(to_archive)} to archive")

        for f, year_month in to_archive:
            if _time.time() - start_time > MAX_RUNTIME:
                break

            try:
                file_id = f["id"]
                old_parents = ",".join(f.get("parents", []))

                # Get or create year-month folder in archive drive
                if year_month not in folder_cache:
                    folder_cache[year_month] = get_or_create_folder_in_drive(
                        admin_drive,
                        archive_drive_id,
                        archive_drive_id,
                        year_month,
                    )
                archive_folder = folder_cache[year_month]

                # Move file to archive drive
                admin_drive.files().update(
                    fileId=file_id,
                    addParents=archive_folder,
                    removeParents=old_parents,
                    supportsAllDrives=True,
                    fields="id",
                ).execute()
                archived += 1
                if archived % 200 == 0:
                    elapsed = int(_time.time() - start_time)
                    rate = archived / max(elapsed, 1) * 60
                    print(f"Progress: {archived} archived ({rate:.0f}/min, {elapsed}s)")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Archive error for {f.get('name','?')}: {e}")

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    result = {
        "archived": archived,
        "skipped": skipped,
        "errors": errors,
        "archive_drive": archive_name,
    }
    print(f"=== Archive complete: {json.dumps(result)} ===")
    return json.dumps(result), 200


@functions_framework.http
def check_status(request):
    """Health check: verify all connections work."""
    status = {}

    # Check Drive access
    try:
        drive = get_drive_service()
        pub = (
            drive.files()
            .get(fileId=CONFIG["PUBLIC_DRIVE_ID"], supportsAllDrives=True)
            .execute()
        )
        status["public_drive"] = f"✅ {pub.get('name', 'Connected')}"
    except Exception as e:
        status["public_drive"] = f"❌ {e}"

    try:
        drive = get_drive_service()
        priv = (
            drive.files()
            .get(fileId=CONFIG["PRIVATE_DRIVE_ID"], supportsAllDrives=True)
            .execute()
        )
        status["private_drive"] = f"✅ {priv.get('name', 'Connected')}"
    except Exception as e:
        status["private_drive"] = f"❌ {e}"

    # Check Admin API (user listing)
    try:
        users = get_all_users()
        status["admin_api"] = f"✅ {len(users)} users found"
    except Exception as e:
        status["admin_api"] = f"❌ {e}"

    # Check Claude API
    try:
        client = anthropic.Anthropic(api_key=CONFIG["CLAUDE_API_KEY"])
        resp = client.messages.create(
            model=CONFIG["CLAUDE_MODEL"],
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}],
        )
        status["claude_api"] = f"✅ Connected ({CONFIG['CLAUDE_MODEL']})"
    except Exception as e:
        status["claude_api"] = f"❌ {e}"

    for k, v in status.items():
        print(f"{k}: {v}")

    if request:
        return json.dumps(status, indent=2), 200
    return status


# =============================================================================
# LOCAL TESTING
# =============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "status":
            check_status()
        elif cmd == "backfill":
            backfill_all()
        elif cmd == "run":
            process_all()
        else:
            print(f"Usage: python main.py [status|run|backfill]")
    else:
        print("ZMR Document Hub")
        print("Commands: status, run, backfill")

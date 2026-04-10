#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python3"

# ── Config ──────────────────────────────────────────────────────
# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
elif [ -f "${SCRIPT_DIR}/.env" ]; then
    export $(grep -v '^#' "${SCRIPT_DIR}/.env" | xargs)
fi

# R2 Credentials should be exported in the environment:
# export R2_ACCESS_KEY_ID="..."
# export R2_SECRET_ACCESS_KEY="..."

R2_BUCKET_NAME="pandects-bulk"
R2_ENDPOINT="https://7b5e7846d94ee35b35e21999fc4fad5b.r2.cloudflarestorage.com"
PUBLIC_DEV_BASE="https://bulk.pandects.org"

# Local paths
BACKUP_ROOT="/tmp/db_sync_artifacts"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
SESSION_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

SQL_DUMP_FILE="${SESSION_DIR}/public_${TIMESTAMP}.sql.gz"
TARGET_DB="${MARIADB_DATABASE:-pdx}"

# API-facing table allowlist (all and only what backend API currently requires).
API_TABLES=(
  agreement_counsel
  agreement_buyer_type_matrix_summary
  agreement_deal_type_summary
  agreement_industry_pairing_summary
  agreement_overview_summary
  agreement_ownership_deal_size_summary
  agreement_ownership_mix_summary
  agreement_status_summary
  agreement_target_industry_summary
  agreements
  clauses
  counsel
  latest_sections_search
  latest_sections_search_standard_ids
  naics_sectors
  naics_sub_sectors
  sections
  summary_data
  tax_clause_assignments
  tax_clause_taxonomy_l1
  tax_clause_taxonomy_l2
  tax_clause_taxonomy_l3
  taxonomy_l1
  taxonomy_l2
  taxonomy_l3
  xml
)

TABLES_LIST=""
for table in "${API_TABLES[@]}"; do
  if [ -n "$TABLES_LIST" ]; then
    TABLES_LIST+=","
  fi
  TABLES_LIST+="${TARGET_DB}.${table}"
done

# ── Checks ──────────────────────────────────────────────────────
if ! command -v mydumper &> /dev/null; then
    echo "❌ Error: 'mydumper' is not installed."
    echo "   Install it (macOS): brew install mydumper"
    exit 1
fi

if [ -z "${R2_ACCESS_KEY_ID:-}" ] || [ -z "${R2_SECRET_ACCESS_KEY:-}" ]; then
    echo "❌ Error: R2 credentials are missing."
    echo "   Please create a .env file in bulk/ with R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY."
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "❌ Error: bulk virtualenv is missing: $PYTHON_BIN"
    echo "   Create it with:"
    echo "   python3 -m venv bulk/.venv"
    echo "   bulk/.venv/bin/python3 -m pip install -r bulk/requirements.txt"
    exit 1
fi

echo "🚀 Starting Full Sync: Local -> R2 (Logical + SQL)"
echo "📚 Export table allowlist (${#API_TABLES[@]} tables): ${API_TABLES[*]}"

LOGICAL_DIR="${SESSION_DIR}/logical"
LOGICAL_ARCHIVE="${SESSION_DIR}/logical_backup_${TIMESTAMP}.tar.gz"

mkdir -p "$LOGICAL_DIR"

# ── 1. Create Logical Backup (For Fly Restore) ──────────────────
echo "📦 [1/4] Taking Logical Backup (mydumper)..."
mydumper \
  --host="${MARIADB_HOST:-127.0.0.1}" \
  --port="${MARIADB_PORT:-3306}" \
  --user="${MARIADB_USER:-root}" \
  --password="${MARIADB_PASSWORD:-}" \
  --database="${TARGET_DB}" \
  --tables-list="${TABLES_LIST}" \
  --outputdir="$LOGICAL_DIR" \
  --threads="${MYDUMPER_THREADS:-6}" \
  --rows="${MYDUMPER_ROWS:-500000}" \
  --triggers \
  --routines \
  --events

find "$LOGICAL_DIR" -name '._*' -delete
find "$LOGICAL_DIR" -name '.DS_Store' -delete

if [ ! -f "$LOGICAL_DIR/metadata" ]; then
    echo "❌ Error: mydumper output missing metadata file."
    echo "   Contents of $LOGICAL_DIR:"
    ls -la "$LOGICAL_DIR" || true
    exit 1
fi

if ! find "$LOGICAL_DIR" -maxdepth 1 -type f -name "*.sql" | grep -q .; then
    echo "❌ Error: mydumper output contains no .sql files."
    echo "   Contents of $LOGICAL_DIR:"
    ls -la "$LOGICAL_DIR" || true
    exit 1
fi

echo "✅ Logical dump sanity checks passed"

echo "🗜️  Compressing Logical Backup..."
COPYFILE_DISABLE=1 tar -czf "$LOGICAL_ARCHIVE" -C "$LOGICAL_DIR" .
echo "✅ Logical Archive Ready: $(du -h "$LOGICAL_ARCHIVE" | cut -f1)"

# ── 2. Create SQL Dump (For Public Access) ──────────────────────
echo "📄 [2/4] Taking SQL Dump (for Public Access)..."
mysqldump \
  --host="${MARIADB_HOST:-127.0.0.1}" \
  --port="${MARIADB_PORT:-3306}" \
  --user="${MARIADB_USER:-root}" \
  --password="${MARIADB_PASSWORD:-}" \
  --single-transaction \
  --quick \
  --lock-tables=false \
  --routines \
  "${TARGET_DB}" \
  "${API_TABLES[@]}" \
  | gzip > "$SQL_DUMP_FILE"

echo "✅ SQL Dump Ready: $(du -h "$SQL_DUMP_FILE" | cut -f1)"

# ── 2b. Create Checksum ─────────────────────────────────────────
echo "🔐 [2b/4] Generating checksum..."
CHECKSUM_FILE="${SQL_DUMP_FILE}.sha256"
sha256sum "$SQL_DUMP_FILE" > "$CHECKSUM_FILE"
echo "✅ Checksum file created: $CHECKSUM_FILE"

# ── 3. Upload to R2 ─────────────────────────────────────────────
echo "☁️  [3/4] Uploading artifacts to R2..."

# Ensure boto3 is installed
if ! "$PYTHON_BIN" -c "import boto3" 2>/dev/null; then
    echo "⚠️  boto3 not found. Installing..."
    "$PYTHON_BIN" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

"$PYTHON_BIN" - <<EOF
import boto3
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

endpoint        = "${R2_ENDPOINT}"
public_dev_base = "${PUBLIC_DEV_BASE}"
bucket          = "${R2_BUCKET_NAME}"

session = boto3.session.Session()
client  = session.client(
    service_name='s3',
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    endpoint_url=endpoint,
)


class ProgressPrinter:
    def __init__(self, label: str, path: Path) -> None:
        self.label = label
        self.path = path
        self.total_bytes = path.stat().st_size
        self.seen_bytes = 0
        self.last_percent = -1
        self.lock = threading.Lock()

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(num_bytes)
        unit = units[0]
        for unit in units:
            if value < 1024 or unit == units[-1]:
                break
            value /= 1024
        if unit == "B":
            return f"{int(value)} {unit}"
        return f"{value:.1f} {unit}"

    def __call__(self, chunk_size: int) -> None:
        with self.lock:
            self.seen_bytes += chunk_size
            if self.total_bytes == 0:
                percent = 100
            else:
                percent = min(int(self.seen_bytes * 100 / self.total_bytes), 100)

            if percent != self.last_percent or self.seen_bytes >= self.total_bytes:
                progress = (
                    f"\r   {self.label}: {percent:3d}% "
                    f"({self._format_bytes(self.seen_bytes)}/"
                    f"{self._format_bytes(self.total_bytes)})"
                )
                sys.stdout.write(progress)
                sys.stdout.flush()
                self.last_percent = percent

    def finish(self) -> None:
        with self.lock:
            if self.last_percent < 100:
                self.seen_bytes = self.total_bytes
                sys.stdout.write(
                    f"\r   {self.label}: 100% "
                    f"({self._format_bytes(self.total_bytes)}/"
                    f"{self._format_bytes(self.total_bytes)})"
                )
            sys.stdout.write("\n")
            sys.stdout.flush()


def upload_with_progress(path: Path, key: str, acl: str, label: str) -> None:
    print(f"📤 Uploading {label}: {key}")
    progress = ProgressPrinter(label, path)
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ACL": acl},
        Callback=progress,
    )
    progress.finish()

# ── Upload Logical Backup ────────────────────────────────────────
logical_key = f"logical_backups/backup_${TIMESTAMP}.tar.gz"
upload_with_progress(Path("${LOGICAL_ARCHIVE}"), logical_key, "private", "logical backup")
print(f"   ✅ Logical backup uploaded: {logical_key}")

# ── Upload SQL Dump and Related Files ─────────────────────────────
dump_path     = Path("${SQL_DUMP_FILE}")
checksum_path = Path("${CHECKSUM_FILE}")

dump_key     = f"dumps/{dump_path.name}"
checksum_key = f"dumps/{checksum_path.name}"

upload_with_progress(dump_path, dump_key, "public-read", "dump")

upload_with_progress(checksum_path, checksum_key, "public-read", "checksum")

# ── Generate Manifest ─────────────────────────────────────────────
sha256_digest = hashlib.sha256(dump_path.read_bytes()).hexdigest()
dump_url       = f"{endpoint}/{bucket}/{dump_key}"
checksum_url   = f"{endpoint}/{bucket}/{checksum_key}"
dump_url_dev   = f"{public_dev_base}/{dump_key}"
checksum_url_dev = f"{public_dev_base}/{checksum_key}"

manifest = {
    "filename":          dump_path.name,
    "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "size_bytes":        dump_path.stat().st_size,
    "sha256":            sha256_digest,
    "download_url":      dump_url,
    "checksum_url":      checksum_url,
    "download_url_dev":  dump_url_dev,
    "checksum_url_dev":  checksum_url_dev,
}

manifest_path = dump_path.with_name(dump_path.name + ".manifest.json")
print(f"📝 Writing manifest to {manifest_path}")
manifest_path.write_text(json.dumps(manifest, indent=2))

manifest_key = f"dumps/{manifest_path.name}"
upload_with_progress(manifest_path, manifest_key, "public-read", "manifest")

# ── Update latest.* Pointers ─────────────────────────────────────
print("🔁 Updating latest.* symlinks...")
for src_key, dst_key in [
    (dump_key,      "dumps/latest.sql.gz"),
    (checksum_key,  "dumps/latest.sql.gz.sha256"),
    (manifest_key,  "dumps/latest.json"),
    (logical_key,   "logical_backups/latest.tar.gz"),
]:
    client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        ACL="public-read"
    )

print("✅ All uploads successful.")
EOF

# ── Cleanup ─────────────────────────────────────────────────────
echo "🧹 [4/4] Cleaning up local artifacts..."
rm -rf "$SESSION_DIR"

echo "🎉 Sync Complete! Artifacts are on R2."
echo "   - Internal Restore: s3://${R2_BUCKET_NAME}/logical_backups/backup_${TIMESTAMP}.tar.gz"
echo "   - Public Dump:    s3://${R2_BUCKET_NAME}/dumps/latest.sql.gz"

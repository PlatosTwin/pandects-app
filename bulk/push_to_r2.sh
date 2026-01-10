#!/bin/bash
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────
# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
elif [ -f bulk/.env ]; then
    export $(grep -v '^#' bulk/.env | xargs)
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

echo "🚀 Starting Full Sync: Local -> R2 (Logical + SQL)"

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
  --database="${MARIADB_DATABASE:-mna}" \
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
  --databases "${MARIADB_DATABASE:-mna}" \
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
if ! $(which python3) -c "import boto3" 2>/dev/null; then
    echo "⚠️  boto3 not found. Installing..."
    $(which pip3) install boto3
fi

$(which python3) - <<EOF
import boto3
import hashlib
import json
import os
import sys
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

# ── Upload Logical Backup ────────────────────────────────────────
print(f"📤 Uploading logical backup...")
logical_key = f"logical_backups/backup_${TIMESTAMP}.tar.gz"
client.upload_file("${LOGICAL_ARCHIVE}", bucket, logical_key, ExtraArgs={"ACL":"private"})
print(f"   ✅ Logical backup uploaded: {logical_key}")

# ── Upload SQL Dump and Related Files ─────────────────────────────
dump_path     = Path("${SQL_DUMP_FILE}")
checksum_path = Path("${CHECKSUM_FILE}")

dump_key     = f"dumps/{dump_path.name}"
checksum_key = f"dumps/{checksum_path.name}"

print(f"📤 Uploading dump: {dump_key}")
client.upload_file(str(dump_path), bucket, dump_key, ExtraArgs={"ACL":"public-read"})

print(f"📤 Uploading checksum: {checksum_key}")
client.upload_file(str(checksum_path), bucket, checksum_key, ExtraArgs={"ACL":"public-read"})

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
print(f"📤 Uploading manifest: {manifest_key}")
client.upload_file(str(manifest_path), bucket, manifest_key, ExtraArgs={"ACL":"public-read"})

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

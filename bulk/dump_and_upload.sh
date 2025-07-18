#!/bin/bash

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────
MARIADB_HOST="pandects-db.internal"
MARIADB_USER="panda"
MARIADB_DATABASE="mna"
R2_BUCKET_NAME="pandects-bulk"
R2_ENDPOINT="https://34730161d8a80dadcd289d6774ffff3d.r2.cloudflarestorage.com"

# ── Dump DB ─────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")
DUMP_FILE="/tmp/db_dump_${TIMESTAMP}.sql.gz"
CHECKSUM_FILE="${DUMP_FILE}.sha256"

echo "🔄 Dumping MariaDB and compressing..."

mysqldump -h "$MARIADB_HOST" -u "$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE" \
  --single-transaction --skip-lock-tables --routines | gzip > "$DUMP_FILE"

# ── Create Checksum ─────────────────────────────────────────────
echo "🔐 Generating checksum..."
sha256sum "$DUMP_FILE" > "$CHECKSUM_FILE"
echo "Checksum file created at $CHECKSUM_FILE"

# ── Upload to R2 ────────────────────────────────────────────────
echo "☁️ Uploading to R2..."

python3 - <<EOF
import boto3
from pathlib import Path
import hashlib
import json
import time

# ── Setup ───────────────────────────────────────────────────────
session = boto3.session.Session()
client = session.client(
    service_name='s3',
    aws_access_key_id="${R2_ACCESS_KEY_ID}",
    aws_secret_access_key="${R2_SECRET_ACCESS_KEY}",
    endpoint_url="${R2_ENDPOINT}",
)

dump_path = Path("$DUMP_FILE")
checksum_path = Path("$CHECKSUM_FILE")

dump_key = f"dumps/{dump_path.name}"
checksum_key = f"dumps/{checksum_path.name}"

print(f"📤 Uploading dump: {dump_key}")
client.upload_file(str(dump_path), "${R2_BUCKET_NAME}", dump_key, ExtraArgs={"ACL": "public-read"})

print(f"📤 Uploading checksum: {checksum_key}")
client.upload_file(str(checksum_path), "${R2_BUCKET_NAME}", checksum_key, ExtraArgs={"ACL": "public-read"})

# ── Metadata ──────────────────────────────────────────────────────
dump_url = f"{R2_ENDPOINT}/${R2_BUCKET_NAME}/{dump_key}"
checksum_url = f"{R2_ENDPOINT}/${R2_BUCKET_NAME}/{checksum_key}"

with open(str(dump_path), "rb") as f:
    sha256_digest = hashlib.sha256(f.read()).hexdigest()

manifest = {
    "filename": dump_path.name,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "size_bytes": dump_path.stat().st_size,
    "sha256": sha256_digest,
    "download_url": dump_url,
    "checksum_url": checksum_url,
}

# Use correct naming: add suffix rather than replacing .gz
manifest_path = dump_path.with_name(dump_path.name + ".manifest.json")
print(f"📝 Writing manifest to {manifest_path}")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

manifest_key = f"dumps/{manifest_path.name}"

print(f"📤 Uploading manifest: {manifest_key}")
client.upload_file(str(manifest_path), "${R2_BUCKET_NAME}", manifest_key, ExtraArgs={"ACL": "public-read"})

# ── Symlink latest.* pointers ─────────────────────────────────────
print("🔁 Updating latest.* symlinks...")
client.copy_object(
    Bucket="${R2_BUCKET_NAME}",
    CopySource={"Bucket": "${R2_BUCKET_NAME}", "Key": dump_key},
    Key="dumps/latest.sql.gz",
    ACL="public-read"
)
client.copy_object(
    Bucket="${R2_BUCKET_NAME}",
    CopySource={"Bucket": "${R2_BUCKET_NAME}", "Key": checksum_key},
    Key="dumps/latest.sql.gz.sha256",
    ACL="public-read"
)
client.copy_object(
    Bucket="${R2_BUCKET_NAME}",
    CopySource={"Bucket": "${R2_BUCKET_NAME}", "Key": manifest_key},
    Key="dumps/latest.json",
    ACL="public-read"
)

print(f"✅ Uploaded: {dump_key}, {checksum_key}, {manifest_key}")
print("✅ Symlinks updated to latest.*"*

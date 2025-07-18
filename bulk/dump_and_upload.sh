#!/bin/bash
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────
MARIADB_HOST="pandects-db.internal"
MARIADB_USER="panda"
MARIADB_DATABASE="mna"

R2_BUCKET_NAME="pandects-bulk"
R2_ENDPOINT="https://34730161d8a80dadcd289d6774ffff3d.r2.cloudflarestorage.com"
PUBLIC_DEV_BASE="https://pub-d1f4ad8b64bd4b89a2d5c5ab58a4ebdf.r2.dev"

# ── Dump DB ─────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")
DUMP_FILE="/tmp/db_dump_${TIMESTAMP}.sql.gz"
CHECKSUM_FILE="${DUMP_FILE}.sha256"

echo "🔄 Dumping MariaDB and compressing…"
mysqldump \
  -h "$MARIADB_HOST" \
  -u "$MARIADB_USER" \
  -p"$MARIADB_PASSWORD" \
  "$MARIADB_DATABASE" \
  --single-transaction \
  --skip-lock-tables \
  --routines \
| gzip > "$DUMP_FILE"

# ── Create Checksum ─────────────────────────────────────────────
echo "🔐 Generating checksum…"
sha256sum "$DUMP_FILE" > "$CHECKSUM_FILE"
echo "Checksum file created at $CHECKSUM_FILE"

# ── Upload to R2 ────────────────────────────────────────────────
echo "☁️ Uploading to R2…"
python3 - <<EOF
import boto3, hashlib, json, time
from pathlib import Path

endpoint           = "${R2_ENDPOINT}"
public_dev_base    = "${PUBLIC_DEV_BASE}"
bucket             = "${R2_BUCKET_NAME}"

session = boto3.session.Session()
client  = session.client(
    service_name='s3',
    aws_access_key_id="${R2_ACCESS_KEY_ID}",
    aws_secret_access_key="${R2_SECRET_ACCESS_KEY}",
    endpoint_url=endpoint,
)

dump_path     = Path("$DUMP_FILE")
checksum_path = Path("$CHECKSUM_FILE")

dump_key     = f"dumps/{dump_path.name}"
checksum_key = f"dumps/{checksum_path.name}"

print(f"📤 Uploading dump: {dump_key}")
client.upload_file(str(dump_path), bucket, dump_key, ExtraArgs={"ACL":"public-read"})

print(f"📤 Uploading checksum: {checksum_key}")
client.upload_file(str(checksum_path), bucket, checksum_key, ExtraArgs={"ACL":"public-read"})

# Build URLs
sha256_digest = hashlib.sha256(dump_path.read_bytes()).hexdigest()
dump_url       = f"{endpoint}/{bucket}/{dump_key}"
checksum_url   = f"{endpoint}/{bucket}/{checksum_key}"
dump_url_dev   = f"{public_dev_base}/{dump_key}"
checksum_url_dev = f"{public_dev_base}/{checksum_key}"

# Generate manifest
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

# Update latest.* pointers
print("🔁 Updating latest.* symlinks…")
for src_key, dst_key in [
    (dump_key,      "dumps/latest.sql.gz"),
    (checksum_key,  "dumps/latest.sql.gz.sha256"),
    (manifest_key,  "dumps/latest.json"),
]:
    client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        ACL="public-read"
    )

print("✅ Symlinks updated to latest.*")
EOF

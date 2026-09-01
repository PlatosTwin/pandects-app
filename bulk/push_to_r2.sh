#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python3"

# ── Config ──────────────────────────────────────────────────────
# Load DB env from backend/.env and only R2 env from bulk/.env.
if [ -f "${REPO_ROOT}/backend/.env" ]; then
    set -a
    source "${REPO_ROOT}/backend/.env"
    set +a
fi

if [ -f "${SCRIPT_DIR}/.env" ]; then
    while IFS='=' read -r key value; do
        case "$key" in
            R2_ACCESS_KEY_ID|R2_SECRET_ACCESS_KEY)
                export "$key=$value"
                ;;
        esac
    done < <(grep -E '^(R2_ACCESS_KEY_ID|R2_SECRET_ACCESS_KEY)=' "${SCRIPT_DIR}/.env" || true)
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

# Public product tables. Shared with bulk/schema_docs/generate_schema_docs.py so
# the public schema docs always cover exactly the downloadable SQL dump.
PUBLIC_TABLES_FILE="${SCRIPT_DIR}/public_tables.txt"
PRIVATE_RESTORE_TABLES_FILE="${SCRIPT_DIR}/private_restore_tables.txt"

for table_file in "$PUBLIC_TABLES_FILE" "$PRIVATE_RESTORE_TABLES_FILE"; do
  if [ ! -f "$table_file" ]; then
    echo "❌ Error: table allowlist not found: ${table_file}."
    exit 1
  fi
done

read_table_file() {
  local table_file="$1"
  while IFS= read -r table; do
    table="${table%%#*}"
    table="$(printf '%s' "$table" | tr -d '[:space:]')"
    [ -n "$table" ] && printf '%s\n' "$table"
  done < "$table_file"
}

PUBLIC_TABLES=()
PRIVATE_RESTORE_TABLES=()
while IFS= read -r table; do
  PUBLIC_TABLES+=("$table")
done < <(read_table_file "$PUBLIC_TABLES_FILE")
while IFS= read -r table; do
  PRIVATE_RESTORE_TABLES+=("$table")
done < <(read_table_file "$PRIVATE_RESTORE_TABLES_FILE")

if [ "${#PUBLIC_TABLES[@]}" -eq 0 ]; then
    echo "❌ Error: no tables found in ${PUBLIC_TABLES_FILE}."
    exit 1
fi

RESTORE_TABLES=("${PUBLIC_TABLES[@]}")
if [ "${#PRIVATE_RESTORE_TABLES[@]}" -gt 0 ]; then
  RESTORE_TABLES+=("${PRIVATE_RESTORE_TABLES[@]}")
fi
if [ "$(printf '%s\n' "${RESTORE_TABLES[@]}" | sort -u | wc -l | tr -d ' ')" -ne "${#RESTORE_TABLES[@]}" ]; then
    echo "❌ Error: duplicate table across public and private restore allowlists."
    exit 1
fi

qualified_tables_csv() {
  local csv=""
  for table in "$@"; do
    if [ -n "$csv" ]; then
      csv+=","
    fi
    csv+="${TARGET_DB}.${table}"
  done
  printf '%s' "$csv"
}

RESTORE_TABLES_LIST="$(qualified_tables_csv "${RESTORE_TABLES[@]}")"
PUBLIC_TABLES_LIST="$(qualified_tables_csv "${PUBLIC_TABLES[@]}")"
PRIVATE_RESTORE_TABLES_LIST=""
if [ "${#PRIVATE_RESTORE_TABLES[@]}" -gt 0 ]; then
  PRIVATE_RESTORE_TABLES_LIST="$(qualified_tables_csv "${PRIVATE_RESTORE_TABLES[@]}")"
fi

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

if [ -z "${MARIADB_HOST:-}" ] || [ -z "${MARIADB_USER:-}" ] || [ -z "${MARIADB_PASSWORD:-}" ] || [ -z "${MARIADB_DATABASE:-}" ]; then
    echo "❌ Error: MariaDB credentials are missing."
    echo "   Expected MARIADB_HOST, MARIADB_USER, MARIADB_PASSWORD, and MARIADB_DATABASE from backend/.env."
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
echo "📚 Public dump tables (${#PUBLIC_TABLES[@]}): ${PUBLIC_TABLES[*]}"
if [ "${#PRIVATE_RESTORE_TABLES[@]}" -gt 0 ]; then
  echo "🔒 Private restore-only tables (${#PRIVATE_RESTORE_TABLES[@]}): ${PRIVATE_RESTORE_TABLES[*]}"
else
  echo "🔒 Private restore-only tables (0): none"
fi

# ── 0. Regenerate Public Schema Docs ────────────────────────────
# Keeps bulk/schema_docs/pandects.dbml and the docs-site schema page in sync
# with what this dump actually contains. Fails (and aborts the push) if a
# dumped table/column is missing a description in table_docs.yml.
echo "📖 [0/4] Regenerating public schema docs..."
if ! "$PYTHON_BIN" -c "import pymysql, yaml" 2>/dev/null; then
    echo "⚠️  Schema-docs dependencies not found. Installing..."
    "$PYTHON_BIN" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi
"$PYTHON_BIN" "${SCRIPT_DIR}/schema_docs/generate_schema_docs.py"

SCHEMA_DOC_PATHS=(
  "bulk/schema_docs/pandects.dbml"
  "docs/docs/guides/bulk-data-schema.md"
)
if ! git -C "$REPO_ROOT" diff --quiet -- "${SCHEMA_DOC_PATHS[@]}" \
   || [ -n "$(git -C "$REPO_ROOT" ls-files --others --exclude-standard -- "${SCHEMA_DOC_PATHS[@]}")" ]; then
    SCHEMA_DOCS_DIRTY=1
else
    SCHEMA_DOCS_DIRTY=0
fi

LOGICAL_DIR="${SESSION_DIR}/logical"
LOGICAL_ARCHIVE="${SESSION_DIR}/logical_backup_${TIMESTAMP}.tar.gz"
LOGICAL_CHECKSUM_FILE="${LOGICAL_ARCHIVE}.sha256"
PRIVATE_TABLE_ROW_COUNTS_FILE="${SESSION_DIR}/private_table_row_counts.json"
RESTORE_TABLE_ROW_COUNTS_FILE="${SESSION_DIR}/restore_table_row_counts.json"

mkdir -p "$LOGICAL_DIR"

# A private serving table is useful in production only when it is complete and
# indexed. Gate publication here so a stale or partially built shadow table can
# never become the promoted restore snapshot.
echo "🔎 [0b/4] Validating private restore tables..."
export PRIVATE_RESTORE_TABLES_FILE PRIVATE_TABLE_ROW_COUNTS_FILE
export RESTORE_TABLES_LIST RESTORE_TABLE_ROW_COUNTS_FILE
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import pymysql


def read_table_file(path: Path) -> list[str]:
    tables: list[str] = []
    for raw_line in path.read_text().splitlines():
        table = raw_line.split("#", 1)[0].strip()
        if table:
            tables.append(table)
    return tables


def quote_identifier(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum():
        raise RuntimeError(f"Unsafe table identifier: {identifier!r}")
    return f"`{identifier}`"


private_tables = read_table_file(Path(os.environ["PRIVATE_RESTORE_TABLES_FILE"]))
restore_tables = [
    entry.rsplit(".", 1)[-1]
    for entry in os.environ["RESTORE_TABLES_LIST"].split(",")
    if entry
]
connection = pymysql.connect(
    host=os.environ["MARIADB_HOST"],
    port=int(os.environ.get("MARIADB_PORT", "3306")),
    user=os.environ["MARIADB_USER"],
    password=os.environ["MARIADB_PASSWORD"],
    database=os.environ["MARIADB_DATABASE"],
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    read_timeout=300,
)
try:
    with connection.cursor() as cursor:
        restore_row_counts: dict[str, int] = {}
        for table in restore_tables:
            cursor.execute(f"SELECT COUNT(*) AS row_count FROM {quote_identifier(table)}")
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError(f"Could not count restore table: {table}")
            restore_row_counts[table] = int(row["row_count"])

        row_counts = {
            table: restore_row_counts[table]
            for table in private_tables
        }

        if "section_text_search" in private_tables:
            cursor.execute(
                """
                SELECT
                    @@innodb_ft_min_token_size AS min_token_size,
                    @@innodb_ft_enable_stopword AS enable_stopword,
                    COALESCE(@@innodb_ft_server_stopword_table, '') AS stopword_table
                """
            )
            fulltext_settings = cursor.fetchone()
            if (
                fulltext_settings is None
                or int(fulltext_settings["min_token_size"]) != 3
                or int(fulltext_settings["enable_stopword"]) != 1
                or str(fulltext_settings["stopword_table"]) != ""
            ):
                raise RuntimeError(
                    "section_text_search requires MariaDB's default InnoDB FULLTEXT "
                    "minimum token size and stopword configuration."
                )
            cursor.execute(
                """
                SELECT COUNT(*) AS index_column_count
                FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'section_text_search'
                  AND INDEX_TYPE = 'FULLTEXT'
                  AND COLUMN_NAME = 'normalized_text'
                """
            )
            index_row = cursor.fetchone()
            if index_row is None or int(index_row["index_column_count"]) < 1:
                raise RuntimeError(
                    "section_text_search must have a FULLTEXT index containing normalized_text."
                )

            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM latest_sections_search) AS source_count,
                    (SELECT COUNT(*) FROM section_text_search) AS target_count,
                    (
                        SELECT COUNT(*)
                        FROM latest_sections_search source
                        LEFT JOIN section_text_search target
                          ON target.section_uuid = source.section_uuid
                        WHERE target.section_uuid IS NULL
                    ) AS missing_count,
                    (
                        SELECT COUNT(*)
                        FROM section_text_search target
                        LEFT JOIN latest_sections_search source
                          ON source.section_uuid = target.section_uuid
                        WHERE source.section_uuid IS NULL
                    ) AS extra_count,
                    (
                        SELECT COUNT(*)
                        FROM latest_sections_search source
                        JOIN sections source_section
                          ON source_section.section_uuid = source.section_uuid
                        JOIN section_text_search target
                          ON target.section_uuid = source.section_uuid
                        WHERE NOT (target.agreement_uuid <=> source.agreement_uuid)
                           OR NOT (target.xml_version <=> source_section.xml_version)
                    ) AS version_mismatch_count,
                    (
                        SELECT COUNT(*)
                        FROM latest_sections_search source
                        JOIN sections source_section
                          ON source_section.section_uuid = source.section_uuid
                        JOIN section_text_search target
                          ON target.section_uuid = source.section_uuid
                        WHERE target.source_xml_sha256 IS NULL
                           OR target.source_xml_sha256 <>
                              UNHEX(SHA2(source_section.xml_content, 256))
                    ) AS source_hash_mismatch_count
                """
            )
            coverage = cursor.fetchone()
            if coverage is None:
                raise RuntimeError("Could not validate section_text_search coverage.")
            failures = {
                key: int(coverage[key])
                for key in (
                    "missing_count",
                    "extra_count",
                    "version_mismatch_count",
                    "source_hash_mismatch_count",
                )
                if int(coverage[key]) != 0
            }
            if int(coverage["source_count"]) != int(coverage["target_count"]):
                failures["row_count_delta"] = (
                    int(coverage["target_count"]) - int(coverage["source_count"])
                )
            if failures:
                raise RuntimeError(
                    "section_text_search does not exactly cover latest_sections_search: "
                    + json.dumps(failures, sort_keys=True)
                )
            row_counts["section_text_search"] = int(coverage["target_count"])
            restore_row_counts["section_text_search"] = int(coverage["target_count"])

    Path(os.environ["PRIVATE_TABLE_ROW_COUNTS_FILE"]).write_text(
        json.dumps(row_counts, sort_keys=True) + "\n"
    )
    Path(os.environ["RESTORE_TABLE_ROW_COUNTS_FILE"]).write_text(
        json.dumps(restore_row_counts, sort_keys=True) + "\n"
    )
finally:
    connection.close()
PY
echo "✅ Private restore table integrity checks passed"

# ── 1. Create Logical Backup (For Fly Restore) ──────────────────
echo "📦 [1/4] Taking Logical Backup (mydumper)..."
mydumper \
  --host="${MARIADB_HOST}" \
  --port="${MARIADB_PORT:-3306}" \
  --user="${MARIADB_USER}" \
  --password="${MARIADB_PASSWORD}" \
  --database="${TARGET_DB}" \
  --tables-list="${RESTORE_TABLES_LIST}" \
  --outputdir="$LOGICAL_DIR" \
  --threads="${MYDUMPER_THREADS:-6}" \
  --rows="${MYDUMPER_ROWS:-100000}" \
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

echo "🔐 [1b/4] Generating logical backup checksum..."
sha256sum "$LOGICAL_ARCHIVE" > "$LOGICAL_CHECKSUM_FILE"
echo "✅ Logical checksum file created: $LOGICAL_CHECKSUM_FILE"

# ── 2. Create SQL Dump (For Public Access) ──────────────────────
echo "📄 [2/4] Taking SQL Dump (for Public Access)..."
mysqldump \
  --host="${MARIADB_HOST}" \
  --port="${MARIADB_PORT:-3306}" \
  --user="${MARIADB_USER}" \
  --password="${MARIADB_PASSWORD}" \
  --single-transaction \
  --quick \
  --lock-tables=false \
  --routines \
  "${TARGET_DB}" \
  "${PUBLIC_TABLES[@]}" \
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
from botocore.config import Config
from pathlib import Path

endpoint        = "${R2_ENDPOINT}"
public_dev_base = "${PUBLIC_DEV_BASE}"
bucket          = "${R2_BUCKET_NAME}"
public_tables_csv = "${PUBLIC_TABLES_LIST}"
private_restore_tables_csv = "${PRIVATE_RESTORE_TABLES_LIST}"
restore_tables_csv = "${RESTORE_TABLES_LIST}"
private_table_row_counts = json.loads(
    Path("${PRIVATE_TABLE_ROW_COUNTS_FILE}").read_text()
)
restore_table_row_counts = json.loads(
    Path("${RESTORE_TABLE_ROW_COUNTS_FILE}").read_text()
)

session = boto3.session.Session()
client  = session.client(
    service_name='s3',
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    endpoint_url=endpoint,
    config=Config(
        connect_timeout=30,
        read_timeout=300,
        retries={"max_attempts": 10, "mode": "standard"},
    ),
)


class ProgressPrinter:
    def __init__(self, label: str, path: Path) -> None:
        self.label = label
        self.path = path
        self.total_bytes = path.stat().st_size
        self.seen_bytes = 0
        self.last_percent = -1
        self.last_line_length = 0
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
                padded_progress = progress.ljust(self.last_line_length)
                sys.stdout.write(padded_progress)
                sys.stdout.flush()
                self.last_percent = percent
                self.last_line_length = len(padded_progress)

    def finish(self) -> None:
        with self.lock:
            if self.last_percent < 100:
                self.seen_bytes = self.total_bytes
                progress = (
                    f"\r   {self.label}: 100% "
                    f"({self._format_bytes(self.total_bytes)}/"
                    f"{self._format_bytes(self.total_bytes)})"
                )
                padded_progress = progress.ljust(self.last_line_length)
                sys.stdout.write(padded_progress)
            else:
                sys.stdout.write("\r".ljust(self.last_line_length))
            sys.stdout.write("\n")
            sys.stdout.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def update_latest_pointer(src_key: str, dst_key: str, acl: str = "public-read") -> None:
    print(f"   ↪ {dst_key} <= {src_key}")
    client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        ACL=acl,
    )

# ── Upload Logical Backup ────────────────────────────────────────
logical_path = Path("${LOGICAL_ARCHIVE}")
logical_checksum_path = Path("${LOGICAL_CHECKSUM_FILE}")
logical_key = f"logical_backups/backup_${TIMESTAMP}.tar.gz"
logical_checksum_key = f"{logical_key}.sha256"

upload_with_progress(logical_path, logical_key, "private", "logical backup")
upload_with_progress(logical_checksum_path, logical_checksum_key, "private", "logical checksum")
print(f"   ✅ Logical backup uploaded: {logical_key}")

# ── Upload SQL Dump and Related Files ─────────────────────────────
dump_path     = Path("${SQL_DUMP_FILE}")
checksum_path = Path("${CHECKSUM_FILE}")

dump_key     = f"dumps/{dump_path.name}"
checksum_key = f"dumps/{checksum_path.name}"

upload_with_progress(dump_path, dump_key, "public-read", "dump")

upload_with_progress(checksum_path, checksum_key, "public-read", "checksum")

logical_sha256 = sha256_file(logical_path)
dump_sha256 = sha256_file(dump_path)

# ── Generate Manifest ─────────────────────────────────────────────
dump_url       = f"{endpoint}/{bucket}/{dump_key}"
checksum_url   = f"{endpoint}/{bucket}/{checksum_key}"
dump_url_dev   = f"{public_dev_base}/{dump_key}"
checksum_url_dev = f"{public_dev_base}/{checksum_key}"

manifest = {
    "filename":          dump_path.name,
    "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "size_bytes":        dump_path.stat().st_size,
    "sha256":            dump_sha256,
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

logical_manifest = {
    "manifest_version": 2,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "logical_key": logical_key,
    "logical_sha256": logical_sha256,
    "logical_size_bytes": logical_path.stat().st_size,
    "logical_checksum_key": logical_checksum_key,
    "public_dump_key": dump_key,
    "public_dump_sha256": dump_sha256,
    "public_dump_size_bytes": dump_path.stat().st_size,
    "tables": restore_tables_csv.split(","),
    "restore_tables": [entry.rsplit(".", 1)[-1] for entry in restore_tables_csv.split(",")],
    "public_tables": [entry.rsplit(".", 1)[-1] for entry in public_tables_csv.split(",")],
    "private_restore_tables": (
        [entry.rsplit(".", 1)[-1] for entry in private_restore_tables_csv.split(",")]
        if private_restore_tables_csv
        else []
    ),
    "private_table_row_counts": private_table_row_counts,
    "restore_table_row_counts": restore_table_row_counts,
}

logical_manifest_path = logical_path.with_suffix(logical_path.suffix + ".manifest.json")
print(f"📝 Writing logical manifest to {logical_manifest_path}")
logical_manifest_path.write_text(json.dumps(logical_manifest, indent=2))

logical_manifest_key = f"logical_backups/{logical_manifest_path.name}"
upload_with_progress(logical_manifest_path, logical_manifest_key, "private", "logical manifest")

# ── Update latest.* Pointers ─────────────────────────────────────
print("🔁 Updating latest.* symlinks...")
for src_key, dst_key in [
    (dump_key,             "dumps/latest.sql.gz"),
    (checksum_key,         "dumps/latest.sql.gz.sha256"),
    (manifest_key,         "dumps/latest.json"),
    (logical_key,          "logical_backups/latest.tar.gz"),
    (logical_checksum_key, "logical_backups/latest.tar.gz.sha256"),
    (logical_manifest_key, "logical_backups/latest.json"),
]:
    acl = "public-read" if dst_key.startswith("dumps/") else "private"
    update_latest_pointer(src_key, dst_key, acl=acl)

print("✅ All uploads successful.")
EOF

# ── Cleanup ─────────────────────────────────────────────────────
echo "🧹 [4/4] Cleaning up local artifacts..."
rm -rf "$SESSION_DIR"

echo "🎉 Sync Complete! Artifacts are on R2."
echo "   - Internal Restore: s3://${R2_BUCKET_NAME}/logical_backups/backup_${TIMESTAMP}.tar.gz"
echo "   - Public Dump:    s3://${R2_BUCKET_NAME}/dumps/latest.sql.gz"

if [ "$SCHEMA_DOCS_DIRTY" -eq 1 ]; then
    echo ""
    echo "⚠️  Schema docs changed with this dump. Commit and push to main so the"
    echo "   docs site and dbdocs.io get updated:"
    printf '     %s\n' "${SCHEMA_DOC_PATHS[@]}"
fi

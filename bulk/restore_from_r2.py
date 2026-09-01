import os
import json
import shutil
import subprocess
import sys
import tarfile
import threading
import re
from pathlib import Path
from typing import cast

from boto3.session import Session as Boto3Session

# Configuration
R2_BUCKET_NAME = "pandects-bulk"
R2_ENDPOINT = "https://7b5e7846d94ee35b35e21999fc4fad5b.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")

# Paths
BACKUP_ARCHIVE = Path("/tmp/logical_backup.tar.gz")
BACKUP_DIR = Path("/tmp/logical_backup")

LOGICAL_LATEST_MANIFEST_KEY = "logical_backups/latest.json"
MANIFEST_VERSION = 2
PUBLIC_TABLES_PATH = Path(__file__).with_name("public_tables.txt")
PRIVATE_RESTORE_TABLES_PATH = Path(__file__).with_name("private_restore_tables.txt")


class ProgressPrinter:
    def __init__(self, label: str, total_bytes: int) -> None:
        self.label = label
        self.total_bytes = total_bytes
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table_allowlist(path: Path) -> list[str]:
    tables: list[str] = []
    for raw_line in path.read_text().splitlines():
        table = raw_line.split("#", 1)[0].strip()
        if table:
            tables.append(table)
    if not tables:
        raise Exception(f"Table allowlist is empty: {path}")
    if len(tables) != len(set(tables)):
        raise Exception(f"Table allowlist contains duplicates: {path}")
    return tables


def validate_restore_manifest(manifest: dict[str, object]) -> None:
    required_fields = {
        "manifest_version",
        "logical_key",
        "logical_sha256",
        "restore_tables",
        "public_tables",
        "private_restore_tables",
        "private_table_row_counts",
        "restore_table_row_counts",
    }
    missing_fields = required_fields - manifest.keys()
    if missing_fields:
        missing_csv = ", ".join(sorted(missing_fields))
        raise Exception(f"Logical backup manifest missing required fields: {missing_csv}")

    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise Exception(
            "Unsupported logical backup manifest version: "
            + f"{manifest['manifest_version']!r}"
        )
    expected_public = set(read_table_allowlist(PUBLIC_TABLES_PATH))
    expected_private = set(read_table_allowlist(PRIVATE_RESTORE_TABLES_PATH))
    expected_restore = expected_public | expected_private
    manifest_public = manifest["public_tables"]
    manifest_private = manifest["private_restore_tables"]
    manifest_restore = manifest["restore_tables"]
    if not all(
        isinstance(table_list, list)
        and all(isinstance(table, str) for table in table_list)
        and len(table_list) == len(set(table_list))
        for table_list in (manifest_public, manifest_private, manifest_restore)
    ):
        raise Exception(
            "Logical backup manifest table fields must be duplicate-free string lists."
        )
    public_tables = cast(list[str], manifest_public)
    private_tables = cast(list[str], manifest_private)
    restore_tables = cast(list[str], manifest_restore)
    if set(public_tables) != expected_public:
        raise Exception("Logical backup manifest public table set does not match this release.")
    if set(private_tables) != expected_private:
        raise Exception("Logical backup manifest private table set does not match this release.")
    if set(restore_tables) != expected_restore:
        raise Exception("Logical backup manifest restore table set does not match this release.")
    private_row_counts = manifest["private_table_row_counts"]
    if not isinstance(private_row_counts, dict):
        raise Exception("Logical backup manifest private_table_row_counts must be an object.")
    if set(private_row_counts) != expected_private or not all(
        isinstance(table, str)
        and isinstance(row_count, int)
        and not isinstance(row_count, bool)
        and row_count >= 0
        for table, row_count in private_row_counts.items()
    ):
        raise Exception(
            "Logical backup manifest private table row counts do not match this release."
        )
    restore_row_counts = manifest["restore_table_row_counts"]
    if not isinstance(restore_row_counts, dict):
        raise Exception("Logical backup manifest restore_table_row_counts must be an object.")
    if set(restore_row_counts) != expected_restore or not all(
        isinstance(table, str)
        and isinstance(row_count, int)
        and not isinstance(row_count, bool)
        and row_count >= 0
        for table, row_count in restore_row_counts.items()
    ):
        raise Exception(
            "Logical backup manifest restore table row counts do not match this release."
        )
    if any(
        restore_row_counts[table] != row_count
        for table, row_count in private_row_counts.items()
    ):
        raise Exception(
            "Logical backup manifest private row counts disagree with restore row counts."
        )


def get_restore_target_manifest(client, bucket):
    print("🔍 Fetching promoted logical backup manifest from R2...", flush=True)
    response = client.get_object(Bucket=bucket, Key=LOGICAL_LATEST_MANIFEST_KEY)
    manifest = json.loads(response["Body"].read())
    if not isinstance(manifest, dict):
        raise Exception("Logical backup manifest must be a JSON object.")
    validate_restore_manifest(manifest)
    return manifest


def validate_private_table_ddl(
    sql_files: list[Path],
    *,
    db_name: str,
    private_tables: list[str],
) -> None:
    for table in private_tables:
        ddl_files = [
            path
            for path in sql_files
            if path.name.startswith(f"{db_name}.{table}-schema")
            or path.name.startswith(f"{table}-schema")
        ]
        if not ddl_files:
            raise Exception(f"Logical backup missing schema DDL for private table: {table}")

        if table == "section_text_search":
            ddl = "\n".join(path.read_text() for path in ddl_files)
            fulltext_normalized_text = re.search(
                r"FULLTEXT(?:\s+(?:KEY|INDEX))?(?:\s+`?[^`\s(]+`?)?\s*"
                r"\([^)]*`?normalized_text`?[^)]*\)",
                ddl,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if (
                "source_xml_sha256" not in ddl
                or fulltext_normalized_text is None
            ):
                raise Exception(
                    "section_text_search backup DDL is missing source_xml_sha256 or a "
                    "FULLTEXT index containing normalized_text."
                )


def validate_restore_archive(
    sql_files: list[Path],
    *,
    db_name: str,
    restore_table_row_counts: dict[str, int],
) -> None:
    """Require a restorable schema and, for non-empty tables, dumped row data."""
    file_names = {path.name for path in sql_files}
    for table, expected_count in restore_table_row_counts.items():
        prefixes = (f"{db_name}.{table}", table)
        has_schema = any(
            f"{prefix}-schema.sql" in file_names
            for prefix in prefixes
        )
        if not has_schema:
            raise Exception(f"Logical backup missing schema DDL for restore table: {table}")

        if expected_count == 0:
            continue
        data_pattern = re.compile(
            rf"^(?:{re.escape(db_name)}\.)?{re.escape(table)}(?:\.\d+)?\.sql$"
        )
        if not any(data_pattern.fullmatch(file_name) for file_name in file_names):
            raise Exception(f"Logical backup missing row data for restore table: {table}")


def strip_definers_in_sql_files(sql_files: list[Path]) -> int:
    # Older myloader builds may not support --skip-definer, so sanitize trigger SQL in place.
    patterns = [
        re.compile(r"/\*![0-9]{5}\s+DEFINER=`[^`]+`@`[^`]+`\*/\s*"),
        re.compile(r"\s+DEFINER=`[^`]+`@`[^`]+`"),
    ]

    rewritten_files = 0
    for path in sql_files:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        file_changed = False

        with path.open("r") as src, temp_path.open("w") as dst:
            for line in src:
                rewritten_line = line
                for pattern in patterns:
                    rewritten_line = pattern.sub(" ", rewritten_line)
                if rewritten_line != line:
                    file_changed = True
                dst.write(rewritten_line)

        if file_changed:
            temp_path.replace(path)
            rewritten_files += 1
        else:
            temp_path.unlink()

    return rewritten_files


def myloader_key_optimization_args(help_text: str) -> list[str]:
    """Pick the key-optimization flag this myloader build advertises, if any.

    Upstream renamed the boolean ``--innodb-optimize-keys`` to the enum-valued
    ``--optimize-keys`` in later releases, and older builds (e.g. Debian
    bookworm's 0.10.1) have neither. Passing an unknown option makes myloader
    exit nonzero, so only emit a spelling that ``myloader --help`` lists.
    ``--optimize-keys-batchsize`` must not count as ``--optimize-keys``.
    """
    if re.search(r"(?<![\w-])--optimize-keys(?![\w-])", help_text):
        return ["--optimize-keys", "AFTER_IMPORT_PER_TABLE"]
    if re.search(r"(?<![\w-])--innodb-optimize-keys(?![\w-])", help_text):
        return ["--innodb-optimize-keys"]
    return []


def probe_myloader_help() -> str:
    """Return ``myloader --help`` output, failing closed if myloader cannot run."""
    try:
        result = subprocess.run(
            ["myloader", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise Exception(f"myloader is not executable: {exc}") from exc
    help_text = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise Exception(
            f"myloader --help exited with status {result.returncode}: {help_text.strip()}"
        )
    if "--directory" not in help_text:
        raise Exception("myloader --help did not advertise --directory; refusing to restore.")
    return help_text


def restore_backup():
    if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
        raise Exception("Missing R2 credentials: R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set")

    db_host = os.environ.get("MARIADB_HOST")
    db_port = os.environ.get("MARIADB_PORT", "3306")
    db_user = os.environ.get("MARIADB_USER")
    db_pass = os.environ.get("MARIADB_PASSWORD")
    db_name = os.environ.get("MARIADB_DATABASE")
    # Production currently runs on a single shared CPU with a tight memory
    # budget. Two workers retain MyLoader's minimum useful concurrency without
    # multiplying the peak memory of concurrent table/index work.
    myloader_threads = os.environ.get("MYLOADER_THREADS", "2")

    if not db_host:
        raise Exception("Missing MARIADB_HOST (e.g., pandects-db.internal)")
    if not db_user:
        raise Exception("Missing MARIADB_USER")
    if not db_pass:
        raise Exception("Missing MARIADB_PASSWORD")
    if not db_name:
        raise Exception("Missing MARIADB_DATABASE")

    print("✅ R2 credentials found", flush=True)

    # Probe the installed myloader before anything destructive: an unknown
    # option would only surface after DROP DATABASE and leave production empty.
    key_optimization_args = myloader_key_optimization_args(probe_myloader_help())
    print(
        "🔎 myloader key optimization: "
        + (" ".join(key_optimization_args) or "not supported by this build"),
        flush=True,
    )

    session = Boto3Session()
    client = session.client(
        service_name="s3",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        endpoint_url=R2_ENDPOINT,
    )

    manifest = get_restore_target_manifest(client, R2_BUCKET_NAME)
    key = manifest["logical_key"]
    head = client.head_object(Bucket=R2_BUCKET_NAME, Key=key)
    archive_size_bytes = head["ContentLength"]
    print(
        f"⬇️  Downloading backup: {key} "
        f"({ProgressPrinter._format_bytes(archive_size_bytes)})",
        flush=True,
    )
    progress = ProgressPrinter("download", archive_size_bytes)
    client.download_file(R2_BUCKET_NAME, key, str(BACKUP_ARCHIVE), Callback=progress)
    progress.finish()

    expected_sha256 = manifest["logical_sha256"]
    actual_sha256 = sha256_file(BACKUP_ARCHIVE)
    if actual_sha256 != expected_sha256:
        raise Exception(
            "Downloaded logical backup checksum mismatch: "
            + f"expected {expected_sha256}, got {actual_sha256}"
        )
    print("✅ Logical backup checksum verified", flush=True)

    print("🧹 Cleaning extraction area...")
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print("📦 Extracting archive...")
    with tarfile.open(BACKUP_ARCHIVE, "r:gz") as tar:
        tar.extractall(path=BACKUP_DIR, filter="data")

    for path in BACKUP_DIR.rglob("._*"):
        if path.is_file():
            path.unlink()
    for path in BACKUP_DIR.rglob(".DS_Store"):
        if path.is_file():
            path.unlink()

    sql_files = list(BACKUP_DIR.glob("*.sql"))

    if not (BACKUP_DIR / "metadata").exists():
        raise Exception("Logical backup missing metadata file; cannot restore.")

    if not sql_files:
        raise Exception("Logical backup contains no .sql files; cannot restore.")

    restore_table_row_counts = cast(
        dict[str, int], manifest["restore_table_row_counts"]
    )
    validate_restore_archive(
        sql_files,
        db_name=db_name,
        restore_table_row_counts=restore_table_row_counts,
    )
    validate_private_table_ddl(
        sql_files,
        db_name=db_name,
        private_tables=list(manifest["private_restore_tables"]),
    )

    print(
        f"✅ Extracted logical backup to {BACKUP_DIR} "
        f"with {len(sql_files)} SQL files",
        flush=True,
    )

    trigger_sql_files = sorted(BACKUP_DIR.glob("*-schema-triggers.sql"))
    rewritten_files = strip_definers_in_sql_files(trigger_sql_files)
    if rewritten_files:
        print(f"🧼 Stripped DEFINER clauses from {rewritten_files} trigger SQL files", flush=True)

    print("🧽 Resetting target database...")
    reset_sql = f"DROP DATABASE IF EXISTS `{db_name}`; CREATE DATABASE `{db_name}`;"
    subprocess.run(
        [
            "mariadb",
            "--protocol=TCP",
            "--host",
            db_host,
            "--port",
            db_port,
            "--user",
            db_user,
            f"--password={db_pass}",
            "-e",
            reset_sql,
        ],
        check=True,
    )

    print(f"📥 Loading logical dump into remote DB (myloader, threads={myloader_threads})...")
    subprocess.run(
        [
            "myloader",
            "--directory",
            str(BACKUP_DIR),
            "--host",
            db_host,
            "--port",
            db_port,
            "--user",
            db_user,
            "--password",
            db_pass,
            "--database",
            db_name,
            "--threads",
            str(myloader_threads),
            *key_optimization_args,
            "--verbose",
            "3",
        ],
        check=True,
    )

    readiness_sql = """
        SELECT CASE WHEN
            EXISTS (
                SELECT 1 FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'section_text_search'
            )
            AND EXISTS (
                SELECT 1 FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'section_text_search'
                  AND COLUMN_NAME = 'normalized_text'
                  AND INDEX_TYPE = 'FULLTEXT'
            )
            AND EXISTS (
                SELECT 1 FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'section_text_search'
                  AND COLUMN_NAME = 'source_xml_sha256'
            )
            AND @@innodb_ft_min_token_size = 3
            AND @@innodb_ft_enable_stopword = 1
            AND COALESCE(@@innodb_ft_server_stopword_table, '') = ''
        THEN 1 ELSE 0 END
    """
    readiness = subprocess.run(
        [
            "mariadb",
            "--protocol=TCP",
            "--host",
            db_host,
            "--port",
            db_port,
            "--user",
            db_user,
            f"--password={db_pass}",
            "--database",
            db_name,
            "--batch",
            "--skip-column-names",
            "--execute",
            readiness_sql,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if readiness.stdout.strip() != "1":
        raise Exception(
            "Restore completed, but section_text_search, its FULLTEXT index, or the "
            "required default InnoDB FULLTEXT settings are not ready."
        )

    for table in restore_table_row_counts:
        if not isinstance(table, str) or not table.replace("_", "").isalnum():
            raise Exception(f"Unsafe restore table name in manifest: {table!r}")
    count_sql = " UNION ALL ".join(
        f"SELECT '{table}', COUNT(*) FROM `{table}`"
        for table in restore_table_row_counts
    )
    actual_counts_result = subprocess.run(
        [
            "mariadb",
            "--protocol=TCP",
            "--host",
            db_host,
            "--port",
            db_port,
            "--user",
            db_user,
            f"--password={db_pass}",
            "--database",
            db_name,
            "--batch",
            "--skip-column-names",
            "--execute",
            count_sql,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_counts = {
        table: int(row_count)
        for line in actual_counts_result.stdout.splitlines()
        for table, row_count in [line.split("\t", 1)]
    }
    for table, expected_count in restore_table_row_counts.items():
        actual_count = actual_counts.get(table)
        if actual_count != expected_count:
            raise Exception(
                f"Restored table {table} row count mismatch: "
                f"expected {expected_count}, got {actual_count}."
            )

    if "section_text_search" in cast(list[str], manifest["private_restore_tables"]):
        integrity_sql = """
            SELECT
                (SELECT COUNT(*) FROM latest_sections_search),
                (SELECT COUNT(*) FROM section_text_search),
                (
                    SELECT COUNT(*)
                    FROM latest_sections_search source
                    LEFT JOIN section_text_search target
                      ON target.section_uuid = source.section_uuid
                    WHERE target.section_uuid IS NULL
                ),
                (
                    SELECT COUNT(*)
                    FROM section_text_search target
                    LEFT JOIN latest_sections_search source
                      ON source.section_uuid = target.section_uuid
                    WHERE source.section_uuid IS NULL
                ),
                (
                    SELECT COUNT(*)
                    FROM latest_sections_search source
                    JOIN sections source_section
                      ON source_section.section_uuid = source.section_uuid
                    JOIN section_text_search target
                      ON target.section_uuid = source.section_uuid
                    WHERE NOT (target.agreement_uuid <=> source.agreement_uuid)
                       OR NOT (target.xml_version <=> source_section.xml_version)
                ),
                (
                    SELECT COUNT(*)
                    FROM latest_sections_search source
                    JOIN sections source_section
                      ON source_section.section_uuid = source.section_uuid
                    JOIN section_text_search target
                      ON target.section_uuid = source.section_uuid
                    WHERE NOT (
                        target.source_xml_sha256
                        <=> UNHEX(SHA2(source_section.xml_content, 256))
                    )
                )
        """
        integrity_result = subprocess.run(
            [
                "mariadb",
                "--protocol=TCP",
                "--host",
                db_host,
                "--port",
                db_port,
                "--user",
                db_user,
                f"--password={db_pass}",
                "--database",
                db_name,
                "--batch",
                "--skip-column-names",
                "--execute",
                integrity_sql,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        integrity_values = [
            int(value) for value in integrity_result.stdout.split()
        ]
        if len(integrity_values) != 6:
            raise Exception("Could not parse restored section_text_search integrity result.")
        source_count, target_count, missing, extra, version_mismatch, hash_mismatch = (
            integrity_values
        )
        if source_count != target_count or any(
            (missing, extra, version_mismatch, hash_mismatch)
        ):
            raise Exception(
                "Restored section_text_search failed source integrity validation: "
                + json.dumps(
                    {
                        "source_count": source_count,
                        "target_count": target_count,
                        "missing_count": missing,
                        "extra_count": extra,
                        "version_mismatch_count": version_mismatch,
                        "source_hash_mismatch_count": hash_mismatch,
                    },
                    sort_keys=True,
                )
            )

    print("✅ Restore Complete. The database is ready.")


if __name__ == "__main__":
    sys.stdout.flush()
    print("🚀 Starting restore process...", flush=True)
    try:
        restore_backup()
    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)

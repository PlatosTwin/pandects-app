import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import boto3

# Configuration
R2_BUCKET_NAME = "pandects-bulk"
R2_ENDPOINT = "https://7b5e7846d94ee35b35e21999fc4fad5b.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")

# Paths
BACKUP_ARCHIVE = Path("/tmp/logical_backup.tar.gz")
BACKUP_DIR = Path("/tmp/logical_backup")

LOGICAL_PREFIX = "logical_backups/backup_"


def get_restore_target_key(client, bucket):
    print("🔍 Finding latest logical backup in R2...", flush=True)
    response = client.list_objects_v2(Bucket=bucket, Prefix=LOGICAL_PREFIX)
    if "Contents" not in response:
        raise Exception("No backups found in 'logical_backups/' prefix")

    latest = sorted(response["Contents"], key=lambda x: x["LastModified"])[-1]
    return latest["Key"]


def restore_backup():
    if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
        raise Exception("Missing R2 credentials: R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set")

    db_host = os.environ.get("MARIADB_HOST")
    db_port = os.environ.get("MARIADB_PORT", "3306")
    db_user = os.environ.get("MARIADB_USER")
    db_pass = os.environ.get("MARIADB_PASSWORD")
    db_name = os.environ.get("MARIADB_DATABASE")
    myloader_threads = os.environ.get("MYLOADER_THREADS", "6")

    if not db_host:
        raise Exception("Missing MARIADB_HOST (e.g., pandects-db.internal)")
    if not db_user:
        raise Exception("Missing MARIADB_USER")
    if not db_pass:
        raise Exception("Missing MARIADB_PASSWORD")
    if not db_name:
        raise Exception("Missing MARIADB_DATABASE")

    print("✅ R2 credentials found", flush=True)

    session = boto3.session.Session()
    client = session.client(
        service_name="s3",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        endpoint_url=R2_ENDPOINT,
    )

    key = get_restore_target_key(client, R2_BUCKET_NAME)
    print(f"⬇️  Downloading backup: {key}", flush=True)
    client.download_file(R2_BUCKET_NAME, key, str(BACKUP_ARCHIVE))

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

    if not (BACKUP_DIR / "metadata").exists():
        raise Exception("Logical backup missing metadata file; cannot restore.")

    if not list(BACKUP_DIR.glob("*.sql")):
        raise Exception("Logical backup contains no .sql files; cannot restore.")

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

    print("📥 Loading logical dump into remote DB (myloader)...")
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
            "--verbose",
            "3",
        ],
        check=True,
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

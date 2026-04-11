from __future__ import annotations

import importlib.util
import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUSH_TO_R2_PATH = _REPO_ROOT / "bulk" / "push_to_r2.sh"
_RESTORE_FROM_R2_PATH = _REPO_ROOT / "bulk" / "restore_from_r2.py"
_RESTORE_SPEC = importlib.util.spec_from_file_location("bulk_restore_from_r2", _RESTORE_FROM_R2_PATH)
if _RESTORE_SPEC is None or _RESTORE_SPEC.loader is None:
    raise RuntimeError("Could not load bulk/restore_from_r2.py")
_RESTORE_MODULE = importlib.util.module_from_spec(_RESTORE_SPEC)
_RESTORE_SPEC.loader.exec_module(_RESTORE_MODULE)


class BulkSyncMetadataTests(unittest.TestCase):
    def test_push_script_loads_db_env_from_backend_and_only_r2_env_from_bulk(self) -> None:
        script = _PUSH_TO_R2_PATH.read_text()
        self.assertIn('source "${REPO_ROOT}/backend/.env"', script)
        self.assertIn('grep -E \'^(R2_ACCESS_KEY_ID|R2_SECRET_ACCESS_KEY)=\' "${SCRIPT_DIR}/.env"', script)
        self.assertNotIn('source "${SCRIPT_DIR}/.env"', script)

    def test_push_script_promotes_logical_latest_metadata(self) -> None:
        script = _PUSH_TO_R2_PATH.read_text()
        self.assertIn('logical_backups/latest.tar.gz', script)
        self.assertIn('logical_backups/latest.tar.gz.sha256', script)
        self.assertIn('logical_backups/latest.json', script)
        self.assertIn('"logical_sha256"', script)

    def test_restore_script_uses_promoted_logical_manifest(self) -> None:
        script = _RESTORE_FROM_R2_PATH.read_text()
        self.assertIn('LOGICAL_LATEST_MANIFEST_KEY = "logical_backups/latest.json"', script)
        self.assertIn("client.get_object(Bucket=bucket, Key=LOGICAL_LATEST_MANIFEST_KEY)", script)
        self.assertIn('manifest["logical_key"]', script)
        self.assertIn('manifest["logical_sha256"]', script)
        self.assertNotIn("list_objects_v2", script)

    def test_restore_backup_uses_manifest_and_verifies_checksum_before_db_reset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive_source = tmp_path / "logical_backup.tar.gz"
            extract_root = tmp_path / "extract"
            backup_archive = tmp_path / "downloaded.tar.gz"

            inner_dir = tmp_path / "logical"
            inner_dir.mkdir()
            (inner_dir / "metadata").write_text("Started dump at: 2026-04-10 12:00:00\n")
            (inner_dir / "agreements.sql").write_text("CREATE TABLE agreements (id INT);\n")
            with tarfile.open(archive_source, "w:gz") as tar:
                tar.add(inner_dir / "metadata", arcname="metadata")
                tar.add(inner_dir / "agreements.sql", arcname="agreements.sql")

            archive_sha256 = _RESTORE_MODULE.sha256_file(archive_source)
            manifest = {
                "logical_key": "logical_backups/backup_2026-04-10_12-00-00.tar.gz",
                "logical_sha256": archive_sha256,
            }

            class _FakeBody:
                def read(self) -> bytes:
                    return json.dumps(manifest).encode("utf-8")

            class _FakeClient:
                def __init__(self) -> None:
                    self.calls: list[tuple[object, ...]] = []

                def get_object(self, Bucket: str, Key: str) -> dict[str, object]:
                    self.calls.append(("get_object", Bucket, Key))
                    return {"Body": _FakeBody()}

                def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
                    self.calls.append(("head_object", Bucket, Key))
                    return {"ContentLength": archive_source.stat().st_size}

                def download_file(self, Bucket: str, Key: str, Filename: str, Callback=None) -> None:
                    self.calls.append(("download_file", Bucket, Key, Filename))
                    Path(Filename).write_bytes(archive_source.read_bytes())
                    if Callback is not None:
                        Callback(archive_source.stat().st_size)

            fake_client = _FakeClient()
            subprocess_calls: list[list[str]] = []

            def _fake_subprocess_run(args: list[str], check: bool) -> None:
                self.assertTrue(check)
                subprocess_calls.append(args)

            class _FakeSession:
                def client(self, **kwargs):  # type: ignore[no-untyped-def]
                    return fake_client

            with (
                patch.object(_RESTORE_MODULE, "BACKUP_ARCHIVE", backup_archive),
                patch.object(_RESTORE_MODULE, "BACKUP_DIR", extract_root),
                patch.object(_RESTORE_MODULE, "R2_ACCESS_KEY_ID", "key"),
                patch.object(_RESTORE_MODULE, "R2_SECRET_ACCESS_KEY", "secret"),
                patch.object(_RESTORE_MODULE.boto3.session, "Session", return_value=_FakeSession()),
                patch.object(_RESTORE_MODULE.subprocess, "run", side_effect=_fake_subprocess_run),
                patch.dict(
                    _RESTORE_MODULE.os.environ,
                    {
                        "MARIADB_HOST": "pandects-db.internal",
                        "MARIADB_PORT": "3306",
                        "MARIADB_USER": "panda",
                        "MARIADB_PASSWORD": "pw",
                        "MARIADB_DATABASE": "pdx",
                        "MYLOADER_THREADS": "6",
                    },
                    clear=False,
                ),
            ):
                _RESTORE_MODULE.restore_backup()

            self.assertEqual(
                fake_client.calls[0],
                ("get_object", _RESTORE_MODULE.R2_BUCKET_NAME, _RESTORE_MODULE.LOGICAL_LATEST_MANIFEST_KEY),
            )
            self.assertEqual(
                fake_client.calls[1],
                ("head_object", _RESTORE_MODULE.R2_BUCKET_NAME, manifest["logical_key"]),
            )
            self.assertEqual(fake_client.calls[2][0], "download_file")
            self.assertEqual(fake_client.calls[2][2], manifest["logical_key"])
            self.assertEqual(len(subprocess_calls), 2)
            self.assertEqual(subprocess_calls[0][0], "mariadb")
            self.assertIn("DROP DATABASE IF EXISTS `pdx`", subprocess_calls[0][-1])
            self.assertEqual(subprocess_calls[1][0], "myloader")
            self.assertTrue((extract_root / "metadata").exists())
            self.assertTrue((extract_root / "agreements.sql").exists())


if __name__ == "__main__":
    unittest.main()

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

    def test_push_script_separates_public_and_private_restore_tables(self) -> None:
        script = _PUSH_TO_R2_PATH.read_text()
        self.assertIn('PRIVATE_RESTORE_TABLES_FILE=', script)
        self.assertIn('table allowlist not found:', script)
        self.assertIn('--tables-list="${RESTORE_TABLES_LIST}"', script)
        self.assertIn('"${PUBLIC_TABLES[@]}"', script)
        self.assertIn('"private_restore_tables"', script)

    def test_push_script_gates_private_table_integrity_and_records_counts(self) -> None:
        script = _PUSH_TO_R2_PATH.read_text()
        self.assertIn("source_section.xml_version", script)
        self.assertIn("target.source_xml_sha256", script)
        self.assertIn("source_hash_mismatch_count", script)
        self.assertIn('"private_table_row_counts": private_table_row_counts', script)
        self.assertIn('"restore_table_row_counts": restore_table_row_counts', script)

    def test_restore_script_uses_promoted_logical_manifest(self) -> None:
        script = _RESTORE_FROM_R2_PATH.read_text()
        self.assertIn('LOGICAL_LATEST_MANIFEST_KEY = "logical_backups/latest.json"', script)
        self.assertIn("client.get_object(Bucket=bucket, Key=LOGICAL_LATEST_MANIFEST_KEY)", script)
        self.assertIn('manifest["logical_key"]', script)
        self.assertIn('manifest["logical_sha256"]', script)
        self.assertNotIn("list_objects_v2", script)

    def test_myloader_key_optimization_prefers_enum_flag(self) -> None:
        help_text = (
            "  --directory                Directory of the dump to import\n"
            "  --optimize-keys            Creates the table without the indexes unless "
            "SKIP is selected. Options: AFTER_IMPORT_PER_TABLE, AFTER_IMPORT_ALL_TABLES and SKIP\n"
            "  --optimize-keys-batchsize  Limits the amount of indexes per ALTER TABLE\n"
        )
        self.assertEqual(
            _RESTORE_MODULE.myloader_key_optimization_args(help_text),
            ["--optimize-keys", "AFTER_IMPORT_PER_TABLE"],
        )

    def test_myloader_key_optimization_falls_back_to_boolean_flag(self) -> None:
        help_text = (
            "  --directory                Directory of the dump to import\n"
            "  --innodb-optimize-keys     Creates the table without the indexes and "
            "it adds them at the end\n"
        )
        self.assertEqual(
            _RESTORE_MODULE.myloader_key_optimization_args(help_text),
            ["--innodb-optimize-keys"],
        )

    def test_myloader_key_optimization_omits_flag_when_unsupported(self) -> None:
        # Debian bookworm's mydumper 0.10.1 advertises neither spelling.
        help_text = (
            "  -d, --directory              Directory of the dump to import\n"
            "  -q, --queries-per-transaction  Number of queries per transaction\n"
            "  -o, --overwrite-tables       Drop tables if they already exist\n"
            "  -B, --database               An alternative database to restore into\n"
            "  -e, --enable-binlog          Enable binary logging of the restore data\n"
        )
        self.assertEqual(_RESTORE_MODULE.myloader_key_optimization_args(help_text), [])

    def test_myloader_key_optimization_ignores_batchsize_only_match(self) -> None:
        help_text = (
            "  --directory                Directory of the dump to import\n"
            "  --optimize-keys-batchsize  Limits the amount of indexes per ALTER TABLE\n"
        )
        self.assertEqual(_RESTORE_MODULE.myloader_key_optimization_args(help_text), [])

    def test_probe_myloader_help_fails_closed_when_not_executable(self) -> None:
        with patch.object(
            _RESTORE_MODULE.subprocess,
            "run",
            side_effect=FileNotFoundError("myloader"),
        ):
            with self.assertRaisesRegex(Exception, "myloader is not executable"):
                _RESTORE_MODULE.probe_myloader_help()

    def test_probe_myloader_help_fails_closed_on_nonzero_exit(self) -> None:
        class _Result:
            returncode = 127
            stdout = ""
            stderr = "boom"

        with patch.object(_RESTORE_MODULE.subprocess, "run", return_value=_Result()):
            with self.assertRaisesRegex(Exception, "exited with status 127"):
                _RESTORE_MODULE.probe_myloader_help()

    def test_probe_myloader_help_fails_closed_on_unrecognized_output(self) -> None:
        class _Result:
            returncode = 0
            stdout = "not myloader\n"
            stderr = ""

        with patch.object(_RESTORE_MODULE.subprocess, "run", return_value=_Result()):
            with self.assertRaisesRegex(Exception, "did not advertise --directory"):
                _RESTORE_MODULE.probe_myloader_help()

    def test_restore_backup_uses_manifest_and_verifies_checksum_before_db_reset(self) -> None:
        help_text = (
            "  --directory      Directory of the dump to import\n"
            "  --optimize-keys  Options: AFTER_IMPORT_PER_TABLE, AFTER_IMPORT_ALL_TABLES and SKIP\n"
        )
        fake_client, subprocess_calls, extract_root, manifest = self._run_restore_backup(
            myloader_help=help_text
        )

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
        self.assertEqual(len(subprocess_calls), 6)
        self.assertEqual(subprocess_calls[0], ["myloader", "--help"])
        self.assertEqual(subprocess_calls[1][0], "mariadb")
        self.assertIn("DROP DATABASE IF EXISTS `pdx`", subprocess_calls[1][-1])
        self.assertEqual(subprocess_calls[2][0], "myloader")
        self.assertIn("--optimize-keys", subprocess_calls[2])
        optimize_index = subprocess_calls[2].index("--optimize-keys")
        self.assertEqual(subprocess_calls[2][optimize_index + 1], "AFTER_IMPORT_PER_TABLE")
        self.assertNotIn("--innodb-optimize-keys", subprocess_calls[2])
        self.assertEqual(subprocess_calls[3][0], "mariadb")
        self.assertIn("information_schema.STATISTICS", subprocess_calls[3][-1])
        self.assertEqual(subprocess_calls[4][0], "mariadb")
        self.assertIn("SELECT 'section_text_search', COUNT(*)", subprocess_calls[4][-1])
        self.assertEqual(subprocess_calls[5][0], "mariadb")
        self.assertIn("source_xml_sha256", subprocess_calls[5][-1])
        # extract_root existence is asserted inside _run_restore_backup while the
        # temporary directory is still alive.
        self.assertIsInstance(extract_root, Path)

    def test_restore_backup_uses_boolean_flag_when_only_that_is_advertised(self) -> None:
        help_text = (
            "  --directory             Directory of the dump to import\n"
            "  --innodb-optimize-keys  Creates the table without the indexes\n"
        )
        _, subprocess_calls, _, _ = self._run_restore_backup(myloader_help=help_text)

        myloader_call = subprocess_calls[2]
        self.assertEqual(myloader_call[0], "myloader")
        self.assertIn("--innodb-optimize-keys", myloader_call)
        self.assertNotIn("--optimize-keys", myloader_call)
        self.assertNotIn("AFTER_IMPORT_PER_TABLE", myloader_call)

    def test_restore_backup_omits_key_flag_for_mydumper_0_10(self) -> None:
        help_text = (
            "  -d, --directory          Directory of the dump to import\n"
            "  -o, --overwrite-tables   Drop tables if they already exist\n"
        )
        _, subprocess_calls, _, _ = self._run_restore_backup(myloader_help=help_text)

        myloader_call = subprocess_calls[2]
        self.assertEqual(myloader_call[0], "myloader")
        self.assertNotIn("--optimize-keys", myloader_call)
        self.assertNotIn("--innodb-optimize-keys", myloader_call)
        self.assertNotIn("AFTER_IMPORT_PER_TABLE", myloader_call)
        self.assertEqual(myloader_call[-2:], ["--verbose", "3"])

    def test_restore_backup_aborts_before_drop_when_myloader_missing(self) -> None:
        with self.assertRaisesRegex(Exception, "myloader is not executable"):
            self._run_restore_backup(myloader_help=None)

        subprocess_calls = self._last_subprocess_calls
        self.assertEqual(subprocess_calls, [["myloader", "--help"]])
        self.assertFalse(
            any("DROP DATABASE" in " ".join(call) for call in subprocess_calls)
        )

    def _run_restore_backup(self, *, myloader_help: str | None):  # type: ignore[no-untyped-def]
        """Drive restore_backup() against fakes.

        ``myloader_help`` is what ``myloader --help`` prints; ``None`` makes the
        probe raise FileNotFoundError as if the binary were absent. Returns
        (fake_client, subprocess_calls, extract_root, manifest). The extract
        directory is torn down when the call returns, so path assertions must
        happen inside the ``with`` block; that is why existence checks run here.
        """
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive_source = tmp_path / "logical_backup.tar.gz"
            extract_root = tmp_path / "extract"
            backup_archive = tmp_path / "downloaded.tar.gz"

            inner_dir = tmp_path / "logical"
            inner_dir.mkdir()
            (inner_dir / "metadata").write_text("Started dump at: 2026-04-10 12:00:00\n")
            public_tables = _RESTORE_MODULE.read_table_allowlist(
                _RESTORE_MODULE.PUBLIC_TABLES_PATH
            )
            private_tables = _RESTORE_MODULE.read_table_allowlist(
                _RESTORE_MODULE.PRIVATE_RESTORE_TABLES_PATH
            )
            restore_tables = public_tables + private_tables
            for table in restore_tables:
                ddl = f"CREATE TABLE {table} (id INT);\n"
                if table == "section_text_search":
                    ddl = (
                        "CREATE TABLE section_text_search ("
                        "section_uuid CHAR(36) PRIMARY KEY, "
                        "source_xml_sha256 BINARY(32), "
                        "normalized_text LONGTEXT NOT NULL, "
                        "FULLTEXT KEY ft_section_text_search_normalized_text "
                        "(normalized_text));\n"
                    )
                (inner_dir / f"pdx.{table}-schema.sql").write_text(ddl)
                (inner_dir / f"pdx.{table}.00000.sql").write_text(
                    f"INSERT INTO {table} VALUES (1);\n"
                )
            with tarfile.open(archive_source, "w:gz") as tar:
                for path in inner_dir.iterdir():
                    tar.add(path, arcname=path.name)

            archive_sha256 = _RESTORE_MODULE.sha256_file(archive_source)
            manifest = {
                "manifest_version": _RESTORE_MODULE.MANIFEST_VERSION,
                "logical_key": "logical_backups/backup_2026-04-10_12-00-00.tar.gz",
                "logical_sha256": archive_sha256,
                "public_tables": public_tables,
                "private_restore_tables": private_tables,
                "restore_tables": public_tables + private_tables,
                "private_table_row_counts": {"section_text_search": 1},
                "restore_table_row_counts": {
                    table: 1 for table in restore_tables
                },
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

            class _FakeCompletedProcess:
                def __init__(self, stdout: str = "1\n", returncode: int = 0) -> None:
                    self.stdout = stdout
                    self.stderr = ""
                    self.returncode = returncode

            self._last_subprocess_calls = subprocess_calls

            def _fake_subprocess_run(args: list[str], check: bool, **kwargs):  # type: ignore[no-untyped-def]
                subprocess_calls.append(args)
                if args == ["myloader", "--help"]:
                    self.assertFalse(check)
                    if myloader_help is None:
                        raise FileNotFoundError("myloader")
                    return _FakeCompletedProcess(myloader_help)
                self.assertTrue(check)
                command = args[-1] if args and args[0] == "mariadb" else ""
                if " UNION ALL " in command:
                    return _FakeCompletedProcess(
                        "".join(f"{table}\t1\n" for table in restore_tables)
                    )
                if "(SELECT COUNT(*) FROM latest_sections_search)" in command:
                    return _FakeCompletedProcess("1\t1\t0\t0\t0\t0\n")
                return _FakeCompletedProcess()

            class _FakeSession:
                def client(self, **kwargs):  # type: ignore[no-untyped-def]
                    return fake_client

            with (
                patch.object(_RESTORE_MODULE, "BACKUP_ARCHIVE", backup_archive),
                patch.object(_RESTORE_MODULE, "BACKUP_DIR", extract_root),
                patch.object(_RESTORE_MODULE, "R2_ACCESS_KEY_ID", "key"),
                patch.object(_RESTORE_MODULE, "R2_SECRET_ACCESS_KEY", "secret"),
                patch.object(_RESTORE_MODULE, "Boto3Session", return_value=_FakeSession()),
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

            self.assertTrue((extract_root / "metadata").exists())
            self.assertTrue((extract_root / "pdx.agreements-schema.sql").exists())
            return fake_client, subprocess_calls, extract_root, manifest

    def test_restore_manifest_rejects_missing_private_table(self) -> None:
        public_tables = _RESTORE_MODULE.read_table_allowlist(
            _RESTORE_MODULE.PUBLIC_TABLES_PATH
        )
        with self.assertRaisesRegex(Exception, "private table set"):
            _RESTORE_MODULE.validate_restore_manifest(
                {
                    "manifest_version": _RESTORE_MODULE.MANIFEST_VERSION,
                    "logical_key": "logical_backups/backup.tar.gz",
                    "logical_sha256": "abc",
                    "public_tables": public_tables,
                    "private_restore_tables": [],
                    "restore_tables": public_tables,
                    "private_table_row_counts": {},
                    "restore_table_row_counts": {
                        table: 1 for table in public_tables
                    },
                }
            )

    def test_restore_rejects_private_table_ddl_without_fulltext(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ddl_path = Path(tmpdir) / "pdx.section_text_search-schema.sql"
            ddl_path.write_text(
                "CREATE TABLE section_text_search (normalized_text LONGTEXT NOT NULL);"
            )

            with self.assertRaisesRegex(Exception, "missing source_xml_sha256 or a FULLTEXT"):
                _RESTORE_MODULE.validate_private_table_ddl(
                    [ddl_path],
                    db_name="pdx",
                    private_tables=["section_text_search"],
                )

    def test_restore_rejects_fulltext_on_the_wrong_column(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ddl_path = Path(tmpdir) / "pdx.section_text_search-schema.sql"
            ddl_path.write_text(
                "CREATE TABLE section_text_search ("
                "source_xml_sha256 BINARY(32), normalized_text LONGTEXT NOT NULL, "
                "other_text LONGTEXT, FULLTEXT KEY ft_wrong (other_text));"
            )

            with self.assertRaisesRegex(Exception, "FULLTEXT index containing normalized_text"):
                _RESTORE_MODULE.validate_private_table_ddl(
                    [ddl_path],
                    db_name="pdx",
                    private_tables=["section_text_search"],
                )

    def test_restore_archive_requires_every_table_schema_before_reset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            sql_file = Path(tmpdir) / "pdx.agreements-schema.sql"
            sql_file.write_text("CREATE TABLE agreements (id INT);")

            with self.assertRaisesRegex(Exception, "schema DDL.*sections"):
                _RESTORE_MODULE.validate_restore_archive(
                    [sql_file],
                    db_name="pdx",
                    restore_table_row_counts={"agreements": 0, "sections": 0},
                )

    def test_restore_archive_requires_data_for_nonempty_table(self) -> None:
        with TemporaryDirectory() as tmpdir:
            schema_file = Path(tmpdir) / "pdx.agreements-schema.sql"
            schema_file.write_text("CREATE TABLE agreements (id INT);")

            with self.assertRaisesRegex(Exception, "row data.*agreements"):
                _RESTORE_MODULE.validate_restore_archive(
                    [schema_file],
                    db_name="pdx",
                    restore_table_row_counts={"agreements": 1},
                )

    def test_restore_manifest_rejects_duplicate_table_entries(self) -> None:
        public_tables = _RESTORE_MODULE.read_table_allowlist(
            _RESTORE_MODULE.PUBLIC_TABLES_PATH
        )
        private_tables = _RESTORE_MODULE.read_table_allowlist(
            _RESTORE_MODULE.PRIVATE_RESTORE_TABLES_PATH
        )
        with self.assertRaisesRegex(Exception, "duplicate-free"):
            _RESTORE_MODULE.validate_restore_manifest(
                {
                    "manifest_version": _RESTORE_MODULE.MANIFEST_VERSION,
                    "logical_key": "logical_backups/backup.tar.gz",
                    "logical_sha256": "abc",
                    "public_tables": public_tables + [public_tables[0]],
                    "private_restore_tables": private_tables,
                    "restore_tables": public_tables + private_tables,
                    "private_table_row_counts": {"section_text_search": 1},
                    "restore_table_row_counts": {
                        table: 1 for table in public_tables + private_tables
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()

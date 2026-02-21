# pyright: reportAny=false
from pathlib import Path
import re
import unittest


class StageQueueAlignmentTests(unittest.TestCase):
    @staticmethod
    def _read(rel_path: str) -> str:
        root = Path(__file__).resolve().parents[1]
        return (root / rel_path).read_text()

    def test_pre_processing_uses_canonical_queue(self) -> None:
        src = self._read("src/etl/defs/b_pre_processing_asset.py")
        self.assertIn("canonical_pre_processing_queue_sql", src)

    def test_tagging_uses_canonical_queue(self) -> None:
        src = self._read("src/etl/defs/c_tagging_asset.py")
        self.assertIn("canonical_tagging_queue_sql", src)

    def test_xml_assets_use_canonical_queues(self) -> None:
        src = self._read("src/etl/defs/f_xml_asset.py")
        self.assertIn("canonical_fresh_xml_build_queue_sql", src)
        self.assertIn("canonical_fresh_xml_verify_queue_sql", src)
        self.assertIn("when coalesce(p.gold_label, p.source_page_type) = 'body' then", src.lower())
        self.assertIn("else p.processed_page_content", src.lower())

    def test_verify_assets_select_latest_xml_only(self) -> None:
        fresh_src = self._read("src/etl/defs/f_xml_asset.py")
        repair_src = self._read("src/etl/defs/f_xml_repair_cycle_asset.py")
        fresh_pattern = re.compile(
            r"SELECT agreement_uuid, version, xml\s+FROM \{xml_table\}\s+WHERE agreement_uuid IN :auuids\s+AND latest = 1",
            re.MULTILINE,
        )
        repair_pattern = re.compile(
            r"SELECT agreement_uuid, version, xml\s+FROM \{xml_table\}\s+WHERE agreement_uuid IN :auuids\s+AND latest = 1",
            re.MULTILINE,
        )
        self.assertRegex(fresh_src, fresh_pattern)
        self.assertRegex(repair_src, repair_pattern)

    def test_sections_use_canonical_queue(self) -> None:
        src = self._read("src/etl/defs/g_sections_asset.py")
        self.assertIn("canonical_fresh_sections_queue_sql", src)

    def test_repair_assets_use_canonical_queues(self) -> None:
        repair_src = self._read("src/etl/defs/d_ai_repair_asset.py")
        cycle_src = self._read("src/etl/defs/f_xml_repair_cycle_asset.py")
        self.assertIn("canonical_ai_repair_enqueue_queue_sql", repair_src)
        self.assertIn("canonical_post_repair_build_queue_sql", cycle_src)
        self.assertIn("canonical_post_repair_verify_queue_sql", cycle_src)
        self.assertIn("when coalesce(p.gold_label, p.source_page_type) = 'body' then", cycle_src.lower())
        self.assertIn("else p.processed_page_content", cycle_src.lower())

    def test_scoped_xml_and_repair_assets_require_batched_scope(self) -> None:
        fresh_src = self._read("src/etl/defs/f_xml_asset.py")
        repair_src = self._read("src/etl/defs/f_xml_repair_cycle_asset.py")
        sections_src = self._read("src/etl/defs/g_sections_asset.py")
        ai_repair_src = self._read("src/etl/defs/d_ai_repair_asset.py")
        self.assertIn("ensure_batched_scope(context, pipeline_config, asset_name=\"xml_verify_asset\")", fresh_src)
        self.assertIn("ensure_batched_scope(context, pipeline_config, asset_name=\"post_repair_build_xml_asset\")", repair_src)
        self.assertIn("ensure_batched_scope(context, pipeline_config, asset_name=\"post_repair_verify_xml_asset\")", repair_src)
        self.assertIn("ensure_batched_scope(context, pipeline_config, asset_name=\"ai_repair_enqueue_asset\")", ai_repair_src)
        self.assertIn("ensure_batched_scope(context, pipeline_config, asset_name=log_prefix)", sections_src)
        self.assertIn("agreement_version_batch_key", fresh_src)
        self.assertIn("agreement_version_batch_key", repair_src)


if __name__ == "__main__":
    _ = unittest.main()

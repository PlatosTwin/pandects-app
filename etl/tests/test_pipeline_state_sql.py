# pyright: reportAny=false
import unittest

from etl.utils.pipeline_state_sql import (
    canonical_ai_repair_enqueue_queue_sql,
    canonical_fresh_xml_build_queue_sql,
    canonical_stage_state_sql,
    canonical_tagging_queue_sql,
)


class PipelineStateSqlTests(unittest.TestCase):
    def test_stage_sql_has_required_xml_red_rules(self) -> None:
        sql = canonical_stage_state_sql("pdx")
        self.assertIn("WHEN latest_xml_status IS NULL THEN '3_xml'", sql)
        self.assertIn("WHEN latest_xml_status = 'invalid' THEN '3_xml'", sql)
        self.assertIn("WHEN latest_xml_status IS NULL THEN 'red'", sql)
        self.assertIn("WHEN latest_xml_status = 'invalid' THEN 'red'", sql)
        self.assertIn("WHERE COALESCE(LOWER(a.status), '') <> 'invalid'", sql)

    def test_stage_sql_marks_no_body_as_preproc_red(self) -> None:
        sql = canonical_stage_state_sql("pdx")
        self.assertIn(
            "WHEN body_page_count = 0 OR tagged_body_page_count < body_page_count THEN '1_pre_processing'",
            sql,
        )
        self.assertIn("WHEN body_page_count = 0 THEN 'red'", sql)

    def test_tagging_queue_uses_body_gap_and_non_gated(self) -> None:
        sql = canonical_tagging_queue_sql("pdx")
        self.assertIn("body_page_count > 0", sql)
        self.assertIn("tagged_body_page_count < body_page_count", sql)
        self.assertIn("page_is_gated = 0", sql)

    def test_fresh_xml_build_queue_handles_stale_verified_regardless_of_repair_attempt(self) -> None:
        sql = canonical_fresh_xml_build_queue_sql("pdx")
        self.assertIn("has_latest_xml = 0", sql)
        self.assertIn("latest_xml_status = 'verified'", sql)
        self.assertIn("has_stale_body_tags = 1", sql)

    def test_ai_repair_queue_selects_latest_invalid_xml_rows(self) -> None:
        sql = canonical_ai_repair_enqueue_queue_sql("pdx")
        self.assertIn("SELECT", sql)
        self.assertIn("x.version AS xml_version", sql)
        self.assertIn("s.latest_xml_ai_repair_attempted AS ai_repair_attempted", sql)
        self.assertIn("r.reason_code", sql)
        self.assertIn("r.page_uuid", sql)
        self.assertIn("x.latest = 1", sql)
        self.assertIn("latest_xml_status = 'invalid'", sql)
        self.assertIn("JOIN pdx.xml_status_reasons r", sql)
        self.assertIn("r.reason_code IN :reason_codes", sql)
        self.assertIn("r.page_uuid IS NOT NULL", sql)


if __name__ == "__main__":
    _ = unittest.main()

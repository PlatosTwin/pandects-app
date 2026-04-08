# pyright: reportAny=false
import unittest

from etl.defs.jobs import defs


class DagsterAssetGraphTests(unittest.TestCase):
    def test_asset_keys_match_canonical_stage_names(self) -> None:
        repo = defs.get_repository_def()
        asset_keys = {".".join(key.path) for key in repo.asset_graph.get_all_asset_keys()}

        expected_keys = {
            "01_staging_asset",
            "01-01_regular_ingest_staging_asset",
            "02_pre_processing_asset",
            "02-01_regular_ingest_pre_processing_asset",
            "03_tagging_asset",
            "03-01_regular_ingest_tagging_asset",
            "04-01_build_xml",
            "04-01_regular_ingest_build_xml",
            "04-02_verify_xml",
            "04-02_regular_ingest_verify_xml",
            "05-01_ai_repair_enqueue_asset",
            "05-01_regular_ingest_ai_repair_enqueue_asset",
            "05-02_ai_repair_poll_asset",
            "05-02_regular_ingest_ai_repair_poll_asset",
            "05-03_reconcile_tags",
            "05-03_regular_ingest_reconcile_tags",
            "05-04_post_repair_build_xml",
            "05-04_regular_ingest_post_repair_build_xml",
            "05-05_post_repair_verify_xml",
            "05-05_regular_ingest_post_repair_verify_xml",
            "06_sections_asset",
            "06-01_sections_from_fresh_xml",
            "06-01_regular_ingest_sections_from_fresh_xml",
            "06-02_sections_from_repair_xml",
            "06-02_regular_ingest_sections_from_repair_xml",
            "07_taxonomy_asset",
            "07-01_regular_ingest_taxonomy_llm_asset",
            "08_tax_module_asset",
            "08-01_tax_module_from_fresh_xml",
            "08-02_tax_module_from_repair_xml",
            "08-03_regular_ingest_tax_module_asset",
            "09_regular_ingest_taxonomy_gold_backfill_asset",
            "10_tx_metadata_asset",
            "10-01_regular_ingest_tx_metadata_offline_asset",
            "10-02_regular_ingest_tx_metadata_web_search_asset",
            "11_embed_sections",
            "99_gating",
        }

        self.assertEqual(asset_keys, expected_keys)

    def test_explicit_jobs_and_manual_assets_are_resolved(self) -> None:
        repo = defs.get_repository_def()
        jobs = {
            job.name: {".".join(key.path) for key in job.asset_layer.executable_asset_keys}
            for job in repo.get_all_jobs()
        }

        self.assertEqual(
            jobs["regular_ingest"],
            {
                "01-01_regular_ingest_staging_asset",
                "02-01_regular_ingest_pre_processing_asset",
                "03-01_regular_ingest_tagging_asset",
                "04-01_regular_ingest_build_xml",
                "04-02_regular_ingest_verify_xml",
                "05-01_regular_ingest_ai_repair_enqueue_asset",
                "05-02_regular_ingest_ai_repair_poll_asset",
                "05-03_regular_ingest_reconcile_tags",
                "05-04_regular_ingest_post_repair_build_xml",
                "05-05_regular_ingest_post_repair_verify_xml",
                "06-01_regular_ingest_sections_from_fresh_xml",
                "06-02_regular_ingest_sections_from_repair_xml",
                "07-01_regular_ingest_taxonomy_llm_asset",
                "08-03_regular_ingest_tax_module_asset",
                "09_regular_ingest_taxonomy_gold_backfill_asset",
                "10-01_regular_ingest_tx_metadata_offline_asset",
                "10-02_regular_ingest_tx_metadata_web_search_asset",
            },
        )
        self.assertTrue({"10_tx_metadata_asset", "11_embed_sections", "99_gating"}.issubset(jobs["__ASSET_JOB"]))


if __name__ == "__main__":
    _ = unittest.main()

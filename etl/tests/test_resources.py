# pyright: reportAny=false
import unittest
from typing import cast
from unittest.mock import patch

from etl.defs.resources import (
    AIRepairAttemptPriority,
    DBResource,
    PipelineConfig,
    QueueRunMode,
    TaggingModel,
    TaxonomyMode,
    get_resources,
)


class DBResourceTests(unittest.TestCase):
    def test_get_engine_enables_stale_connection_protection(self) -> None:
        db = DBResource(
            host="127.0.0.1",
            port="3306",
            user="user",
            password="password",
            database="pdx",
        )

        with patch("etl.defs.resources.create_engine") as mock_create_engine:
            _ = db.get_engine()

        mock_create_engine.assert_called_once_with(
            "mariadb+mysqldb://user:password@127.0.0.1:3306/pdx",
            pool_pre_ping=True,
            pool_recycle=3600,
        )


class TaggingModelTests(unittest.TestCase):
    def test_model_is_cached(self) -> None:
        model = TaggingModel()
        fake_inference = object()

        with patch("etl.defs.resources.NERInference", return_value=fake_inference) as inference_cls:
            first = model.model()
            second = model.model()

        self.assertIs(first, fake_inference)
        self.assertIs(second, fake_inference)
        inference_cls.assert_called_once()


class PipelineConfigResourceTests(unittest.TestCase):
    def test_get_resources_accepts_new_queue_and_openai_batch_keys(self) -> None:
        with patch(
            "etl.defs.resources._load_yaml_config",
            return_value={
                "queue_run_mode": "drain",
                "resume_openai_batches": False,
                "xml_enable_llm_verification": True,
                "ai_repair_attempt_priority": "attempted_first",
                "enable_section_taxonomy": True,
                "taxonomy_mode": "ml",
                "taxonomy_section_title_regex": "^governing",
                "taxonomy_llm_model": "gpt-5.4-mini",
                "taxonomy_llm_sections_per_request": 7,
                "tax_module_agreement_batch_size": 11,
                "enable_tax_taxonomy": True,
                "tax_module_llm_model": "gpt-5.4-mini",
                "tax_module_llm_clauses_per_request": 3,
            },
        ):
            resources = get_resources()

        pipeline_config = cast(PipelineConfig, resources["pipeline_config"])
        self.assertEqual(pipeline_config.queue_run_mode, QueueRunMode.DRAIN)
        self.assertFalse(pipeline_config.resume_openai_batches)
        self.assertTrue(pipeline_config.xml_enable_llm_verification)
        self.assertEqual(
            pipeline_config.ai_repair_attempt_priority,
            AIRepairAttemptPriority.ATTEMPTED_FIRST,
        )
        self.assertTrue(pipeline_config.enable_section_taxonomy)
        self.assertEqual(pipeline_config.taxonomy_mode, TaxonomyMode.ML)
        self.assertEqual(pipeline_config.taxonomy_section_title_regex, "^governing")
        self.assertEqual(pipeline_config.taxonomy_llm_model, "gpt-5.4-mini")
        self.assertEqual(pipeline_config.taxonomy_llm_sections_per_request, 7)
        self.assertEqual(pipeline_config.tax_module_agreement_batch_size, 11)
        self.assertTrue(pipeline_config.enable_tax_taxonomy)
        self.assertEqual(pipeline_config.tax_module_llm_model, "gpt-5.4-mini")
        self.assertEqual(pipeline_config.tax_module_llm_clauses_per_request, 3)

    def test_pipeline_config_defaults_taxonomy_to_llm(self) -> None:
        config = PipelineConfig()
        self.assertFalse(config.xml_enable_llm_verification)
        self.assertTrue(config.enable_section_taxonomy)
        self.assertEqual(config.taxonomy_mode, TaxonomyMode.LLM)
        self.assertIsNone(config.taxonomy_section_title_regex)
        self.assertEqual(config.taxonomy_llm_model, "gpt-5.4-mini")
        self.assertEqual(config.taxonomy_llm_sections_per_request, 5)
        self.assertEqual(config.tax_module_agreement_batch_size, 25)
        self.assertFalse(config.enable_tax_taxonomy)
        self.assertEqual(config.tax_module_llm_model, "gpt-5.4-mini")
        self.assertEqual(config.tax_module_llm_clauses_per_request, 5)

    def test_get_resources_rejects_legacy_scope_keys(self) -> None:
        with patch(
            "etl.defs.resources._load_yaml_config",
            return_value={
                "scope": "batched",
                "resume_open_batches": True,
            },
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Unknown keys in resources.pipeline_config.config: resume_open_batches, scope",
            ):
                _ = get_resources()


if __name__ == "__main__":
    _ = unittest.main()

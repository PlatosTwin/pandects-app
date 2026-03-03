import unittest

from etl.models.ner.article_audit import build_article_failure_records


class NerArticleAuditTests(unittest.TestCase):
    def test_boundary_mismatch_is_reported_once(self) -> None:
        raw_text = "ARTICLE I\nDefinitions\n"
        token_offsets = [
            (0, 7),
            (8, 9),
            (10, 21),
        ]
        gold_tags = ["B-ARTICLE", "E-ARTICLE", "O"]
        pred_tags = ["O", "B-ARTICLE", "E-ARTICLE"]

        records = build_article_failure_records(
            doc_id=7,
            pred_tags=pred_tags,
            gold_tags=gold_tags,
            raw_text=raw_text,
            token_offsets=token_offsets,
            context_chars=40,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["failure_type"], "boundary_mismatch")
        self.assertEqual(record["gold_token_start"], 0)
        self.assertEqual(record["pred_token_start"], 1)
        self.assertEqual(record["overlap_token_count"], 1)

    def test_false_negative_and_false_positive_are_separate(self) -> None:
        raw_text = "ARTICLE II\nCovenants\n"
        token_offsets = [
            (0, 7),
            (8, 10),
            (11, 20),
        ]
        gold_tags = ["B-ARTICLE", "E-ARTICLE", "O"]
        pred_tags = ["O", "O", "O"]

        records = build_article_failure_records(
            doc_id=11,
            pred_tags=pred_tags,
            gold_tags=gold_tags,
            raw_text=raw_text,
            token_offsets=token_offsets,
            context_chars=40,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["failure_type"], "false_negative")
        self.assertEqual(records[0]["gold_text"], "ARTICLE II")
        self.assertIsNone(records[0]["pred_text"])

        fp_records = build_article_failure_records(
            doc_id=12,
            pred_tags=gold_tags,
            gold_tags=["O", "O", "O"],
            raw_text=raw_text,
            token_offsets=token_offsets,
            context_chars=40,
        )
        self.assertEqual(len(fp_records), 1)
        self.assertEqual(fp_records[0]["failure_type"], "false_positive")
        self.assertEqual(fp_records[0]["pred_text"], "ARTICLE II")
        self.assertIsNone(fp_records[0]["gold_text"])


if __name__ == "__main__":
    _ = unittest.main()

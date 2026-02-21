# pyright: reportAny=false
import unittest

from etl.domain.d_ai_repair import RepairDecision, build_jsonl_lines_for_page


class AiRepairDomainTests(unittest.TestCase):
    def test_full_mode_request_id_includes_xml_version(self) -> None:
        decision = RepairDecision(mode="full", windows=[(0, 12)], token_map=[])
        lines, metas = build_jsonl_lines_for_page(
            page_uuid="11111111-1111-1111-1111-111111111111",
            text="Sample text.",
            decision=decision,
            model="gpt-5.1",
            uncertain_spans=[],
            xml_version=7,
        )

        self.assertEqual(len(lines), 1)
        self.assertEqual(len(metas), 1)
        self.assertEqual(
            lines[0]["custom_id"],
            "11111111-1111-1111-1111-111111111111::full::7",
        )
        self.assertEqual(
            metas[0]["request_id"],
            "11111111-1111-1111-1111-111111111111::full::7",
        )


if __name__ == "__main__":
    _ = unittest.main()

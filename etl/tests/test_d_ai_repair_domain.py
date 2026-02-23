# pyright: reportAny=false, reportPrivateUsage=false
import unittest

from etl.domain.d_ai_repair import (
    RepairDecision,
    _user_prompt_full,
    build_jsonl_lines_for_page,
)


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

    def test_full_mode_user_prompt_is_raw_page_text_only(self) -> None:
        page_text = "Line 1\nLine 2"
        prompt = _user_prompt_full(
            page_uuid="11111111-1111-1111-1111-111111111111",
            text=page_text,
        )
        self.assertEqual(prompt, page_text)
        self.assertNotIn("PAGE_UUID=", prompt)
        self.assertNotIn("Task: Insert", prompt)


if __name__ == "__main__":
    _ = unittest.main()
